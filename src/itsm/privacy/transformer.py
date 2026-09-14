"""Transformer PII detection - the open-source model option.

A multilingual NER model from the Hugging Face hub reads the *sentence*, not
the *format*, which is exactly the half of the problem the rule engine cannot
solve: it finds a person named with no trigger word and no known first name,
including names the organisation has never seen.

**It is not used alone.** A NER model is trained on PER/ORG/LOC and knows
nothing about IBANs, matricules or corporate mailboxes; asking it for them
yields nothing. So this backend keeps the structured layer of
:class:`~itsm.privacy.rules.RuleBasedRecognizer` for formats and uses the model
for names. That hybrid is the honest engineering answer, and the per-entity
evaluation shows why: the model wins on ``PERSON``, the rules win everywhere
else.

Default model: ``Davlan/distilbert-base-multilingual-cased-ner-hrl`` - small,
CPU-friendly, and covers both languages of the corpus. For a French-only
deployment ``Jean-Baptiste/camembert-ner`` is usually stronger; set
``ITSM_NER_MODEL`` to switch. Requires the ``hf`` extra and a one-time
download, which is the trade against the rule engine's zero-dependency
deployment.
"""

from __future__ import annotations

from typing import Any

from itsm.logging import get_logger
from itsm.privacy.entities import EntityType, PiiSpan, merge_spans
from itsm.privacy.rules import RuleBasedRecognizer

logger = get_logger("privacy.transformer")

DEFAULT_MODEL = "Davlan/distilbert-base-multilingual-cased-ner-hrl"

#: NER tag -> entity type. ``ORG`` is deliberately dropped: in ticket text the
#: organisations are the employer and its vendors ("SAP", "Microsoft"), which
#: are not personal data and whose redaction would destroy the signal the
#: classifier lives on.
TAG_MAP: dict[str, EntityType] = {
    "PER": EntityType.PERSON,
    "PERSON": EntityType.PERSON,
    "LOC": EntityType.ADDRESS,
    "LOCATION": EntityType.ADDRESS,
}


class TransformerRecognizer:
    """Hybrid detector: a Hugging Face NER model for names, rules for formats.

    Args:
        model_name: Hub identifier of a token-classification model.
        min_score: Minimum model confidence for a name to be kept.
        with_structural: Keep the rule engine's structured-identifier layer.
            Turning it off isolates the model's contribution, which is how the
            comparison in ``docs/07-privacy.md`` was produced.
        map_locations: Treat ``LOC`` predictions as addresses. Off by default:
            in ITSM text most locations are office sites ("Toulouse"), which
            are not personal data - the structured address pattern already
            catches real postal addresses.
        batch_size: Inference batch size.
        device: Torch device; ``None`` lets transformers decide.
    """

    name = "transformer"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        min_score: float = 0.5,
        with_structural: bool = True,
        map_locations: bool = False,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.min_score = min_score
        self.with_structural = with_structural
        self.map_locations = map_locations
        self.batch_size = batch_size
        self.device = device
        self._pipeline: Any | None = None
        self._structural = RuleBasedRecognizer(detect_persons=False)

    def _load(self) -> Any:
        """Load the pipeline on first use, with an actionable error."""
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as error:  # pragma: no cover - depends on the extra
            raise RuntimeError(
                "the transformer backend needs the optional dependencies: "
                "`uv sync --extra hf`. The rule-based backend needs nothing."
            ) from error

        logger.info("loading NER model %s", self.model_name)
        kwargs: dict[str, Any] = {"aggregation_strategy": "simple"}
        if self.device is not None:
            kwargs["device"] = self.device
        self._pipeline = pipeline("token-classification", model=self.model_name, **kwargs)
        return self._pipeline

    def _to_spans(self, text: str, entities: list[dict[str, Any]]) -> list[PiiSpan]:
        """Convert pipeline output into spans, filtering tags and scores."""
        spans: list[PiiSpan] = []
        for entity in entities:
            tag = str(entity.get("entity_group", "")).upper().removeprefix("B-").removeprefix("I-")
            mapped = TAG_MAP.get(tag)
            if mapped is None:
                continue
            if mapped is EntityType.ADDRESS and not self.map_locations:
                continue
            score = float(entity.get("score", 0.0))
            if score < self.min_score:
                continue

            start, end = int(entity["start"]), int(entity["end"])
            # The pipeline occasionally returns a span with leading whitespace
            # or trailing punctuation; tighten it so offsets stay comparable
            # with the reference.
            surface = text[start:end]
            stripped = surface.strip(" .,;:!?()")
            if not stripped:
                continue
            start += surface.index(stripped)
            spans.append(
                PiiSpan(
                    start=start,
                    end=start + len(stripped),
                    entity=mapped,
                    text=stripped,
                    score=round(score, 4),
                    source=f"{self.name}:{tag}",
                )
            )
        return spans

    def detect(self, text: str) -> list[PiiSpan]:
        """Return the disjoint PII spans found in ``text``."""
        return self.detect_batch([text])[0]

    def detect_batch(self, texts: list[str]) -> list[list[PiiSpan]]:
        """Detect over a batch - the only efficient way to use a transformer."""
        cleaned = [text or "" for text in texts]
        non_empty = [index for index, text in enumerate(cleaned) if text.strip()]
        results: list[list[PiiSpan]] = [[] for _ in cleaned]
        if not non_empty:
            return results

        outputs = self._load()([cleaned[index] for index in non_empty], batch_size=self.batch_size)
        # A single input yields a flat list; normalise to a list of lists.
        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]

        for index, entities in zip(non_empty, outputs, strict=True):
            spans = self._to_spans(cleaned[index], list(entities))
            if self.with_structural:
                spans.extend(self._structural.detect(cleaned[index]))
            results[index] = merge_spans(spans)
        return results
