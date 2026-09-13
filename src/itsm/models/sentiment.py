"""Requester-sentiment scoring.

Sentiment on ITSM tickets is not product-review sentiment. Almost every ticket
is negative in the everyday sense - something is broken. What matters
operationally is *escalation risk*: a requester who is blocked, has already
chased twice and writes in capitals is a candidate for proactive handling.
The useful signal is therefore frustration intensity, not polarity.

Two backends share one interface:

``lexicon`` (default)
    A compact FR/EN lexicon with ITSM-specific markers, negation handling and
    intensifiers. No download, no GPU, fully auditable - which is why it is the
    default in a constrained environment, and a solid baseline in any case.

``transformer``
    A multilingual sentiment model from the Hugging Face hub (default:
    ``cardiffnlp/twitter-xlm-roberta-base-sentiment``). Needs the ``[hf]``
    extra and a first-run download.

Both are evaluated on the same weakly labelled set in
``itsm.pipelines.eval_sentiment``; see ``docs/03-modeling.md`` for the caveats
of weak labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from itsm.logging import get_logger
from itsm.text.normalize import strip_accents

logger = get_logger("models.sentiment")

Label = Literal["negative", "neutral", "positive"]

#: Negative markers, weighted by intensity. Keys are accent-free and lowercase.
_NEGATIVE_LEXICON: dict[str, float] = {
    # French
    "bloque": 1.0,
    "bloquee": 1.0,
    "inacceptable": 2.0,
    "inadmissible": 2.0,
    "scandaleux": 2.0,
    "impossible": 0.8,
    "toujours pas": 1.2,
    "encore une fois": 1.2,
    "relance": 1.0,
    "relances": 1.3,
    "aucune reponse": 1.5,
    "personne": 0.8,
    "lamentable": 2.0,
    "catastrophe": 1.8,
    "insupportable": 1.8,
    "ras le bol": 2.0,
    "delai": 0.6,
    "retard": 0.8,
    "urgent": 0.7,
    "tres urgent": 1.4,
    "penible": 1.2,
    "ne fonctionne plus": 0.9,
    "ne marche pas": 0.9,
    "perdu": 0.8,
    "echec": 0.7,
    # English
    "blocked": 1.0,
    "unacceptable": 2.0,
    "ridiculous": 1.8,
    "frustrated": 1.6,
    "frustrating": 1.6,
    "angry": 1.8,
    "still not": 1.2,
    "again": 0.6,
    "no response": 1.5,
    "nobody": 0.8,
    "third time": 1.5,
    "second time": 1.0,
    "asap": 0.9,
    "escalate": 1.4,
    "escalation": 1.4,
    "useless": 1.6,
    "broken": 0.7,
    "cannot work": 1.5,
    "unable to work": 1.5,
    "waiting": 0.6,
    "delay": 0.6,
    "terrible": 1.6,
    "worst": 1.8,
    "disappointed": 1.4,
}

#: Positive markers. Politeness formulas are down-weighted: "merci d'avance"
#: is a convention, not an expression of satisfaction.
_POSITIVE_LEXICON: dict[str, float] = {
    "merci": 0.4,
    "merci beaucoup": 0.8,
    "parfait": 1.2,
    "super": 1.0,
    "excellent": 1.4,
    "rapide": 0.6,
    "efficace": 0.9,
    "pas urgent": 0.6,
    "tres bon": 1.0,
    "bravo": 1.4,
    "thanks": 0.4,
    "thank you": 0.5,
    "great": 1.0,
    "perfect": 1.2,
    "appreciate": 0.9,
    "helpful": 1.0,
    "no urgency": 0.6,
    "no rush": 0.6,
    "awesome": 1.3,
}

_NEGATIONS = ("pas", "plus", "jamais", "aucun", "aucune", "ni", "not", "no", "never")
_INTENSIFIERS = {
    "tres": 1.4,
    "vraiment": 1.4,
    "totalement": 1.5,
    "completement": 1.5,
    "very": 1.4,
    "really": 1.4,
    "completely": 1.5,
    "extremely": 1.6,
}


@dataclass
class SentimentResult:
    """Sentiment of one ticket."""

    label: Label
    score: float
    """Signed intensity in ``[-1, 1]``; negative means frustrated."""
    backend: str
    evidence: list[str]
    """Matched markers, for auditability."""


class LexiconSentiment:
    """Rule-based FR/EN frustration scorer.

    Args:
        negative_threshold: Score below which a ticket is labelled negative.
        positive_threshold: Score above which a ticket is labelled positive.
    """

    backend = "lexicon"

    def __init__(self, negative_threshold: float = -0.18, positive_threshold: float = 0.18) -> None:
        self.negative_threshold = negative_threshold
        self.positive_threshold = positive_threshold

    @staticmethod
    def _shouting(text: str) -> float:
        r"""Detect a shouted message, not a capitalised product name.

        A token test (``\\b[A-Z]{4,}\\b``) would fire on ``MATLAB``, ``ORION``
        or ``ST22`` - which appear in perfectly calm tickets. The share of
        upper-case letters over a long enough span does not.
        """
        letters = [character for character in text if character.isalpha()]
        if len(letters) < 20:
            return 0.0
        upper_ratio = sum(character.isupper() for character in letters) / len(letters)
        return 1.0 if upper_ratio > 0.6 else 0.0

    def _score_one(self, text: str) -> tuple[float, list[str]]:
        value = strip_accents(str(text or "")).lower()
        shouting = self._shouting(str(text or ""))
        exclamations = min(value.count("!"), 3) * 0.2

        tokens = re.findall(r"[a-z']+", value)
        positive = negative = 0.0
        evidence: list[str] = []

        for lexicon, sign in ((_NEGATIVE_LEXICON, -1.0), (_POSITIVE_LEXICON, 1.0)):
            for marker, weight in lexicon.items():
                if marker not in value:
                    continue
                position = value.index(marker)
                prefix = tokens[max(0, len(re.findall(r"[a-z']+", value[:position])) - 2) :]
                multiplier = 1.0
                for word in prefix[:2]:
                    multiplier *= _INTENSIFIERS.get(word, 1.0)
                    if word in _NEGATIONS and sign > 0:
                        multiplier *= -1.0  # "pas parfait"
                contribution = weight * multiplier
                if sign < 0:
                    negative += max(contribution, 0.0)
                else:
                    positive += contribution
                evidence.append(marker)

        raw = positive - negative - shouting - exclamations
        # Squash to [-1, 1]; 3 accumulated points is already extreme.
        score = float(max(-1.0, min(1.0, raw / 3.0)))
        return score, evidence

    def analyze(self, texts: list[str]) -> list[SentimentResult]:
        """Score a batch of texts."""
        results = []
        for text in texts:
            score, evidence = self._score_one(text)
            if score <= self.negative_threshold:
                label: Label = "negative"
            elif score >= self.positive_threshold:
                label = "positive"
            else:
                label = "neutral"
            results.append(
                SentimentResult(
                    label=label, score=round(score, 4), backend=self.backend, evidence=evidence[:6]
                )
            )
        return results


class TransformerSentiment:
    """Multilingual sentiment via a Hugging Face pipeline.

    Args:
        model_name: Hub identifier of a sequence-classification model.
        batch_size: Inference batch size.
        max_length: Truncation length; ticket bodies can be long.
    """

    backend = "transformer"

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        batch_size: int = 32,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._pipeline: Any | None = None

    def _load(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline  # optional dependency

            logger.info("loading sentiment model %s", self.model_name)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                truncation=True,
                max_length=self.max_length,
            )
        return self._pipeline

    @staticmethod
    def _map_label(raw: str) -> Label:
        value = raw.lower()
        if value in {"negative", "label_0", "1 star", "2 stars"}:
            return "negative"
        if value in {"positive", "label_2", "4 stars", "5 stars"}:
            return "positive"
        return "neutral"

    def analyze(self, texts: list[str]) -> list[SentimentResult]:
        """Score a batch of texts."""
        outputs = self._load()(list(texts), batch_size=self.batch_size)
        results: list[SentimentResult] = []
        for output in outputs:
            label = self._map_label(str(output["label"]))
            signed = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}[label]
            results.append(
                SentimentResult(
                    label=label,
                    score=round(signed * float(output["score"]), 4),
                    backend=self.backend,
                    evidence=[],
                )
            )
        return results


def build_sentiment(
    backend: Literal["lexicon", "transformer"] = "lexicon", model_name: str | None = None
) -> LexiconSentiment | TransformerSentiment:
    """Return the sentiment analyser matching the configured backend."""
    if backend == "transformer":
        return TransformerSentiment(model_name) if model_name else TransformerSentiment()
    return LexiconSentiment()
