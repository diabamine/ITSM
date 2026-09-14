"""The anonymisation step: detect, then redact, over a whole corpus.

One class ties the two halves together so callers - the pipeline, the API, the
app - never assemble a detector and a redactor by hand and never disagree about
the order of operations.

The backend is chosen by name, and the choice is the point of this module:

``rules``
    :class:`~itsm.privacy.rules.RuleBasedRecognizer` - offline, no download,
    unbeatable on formats, weaker on unseen names.
``transformer``
    :class:`~itsm.privacy.transformer.TransformerRecognizer` - a Hugging Face
    NER model for names, plus the same rules for formats.
``presidio``
    :class:`~itsm.privacy.presidio.PresidioRecognizer` - Microsoft Presidio's
    analyser, standalone: its own recognisers and nothing else.

All three are scored by the same evaluation, so the trade is measured rather
than argued (``docs/07-privacy.md``). ``with_structural`` decides whether a
model-based backend runs alone or composed with the rule layer; pass ``False``
to hold every backend to the same terms, which is what
``itsm privacy eval --isolate`` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import pandas as pd

from itsm.logging import get_logger
from itsm.privacy.entities import PiiSpan
from itsm.privacy.redact import Policy, Redactor
from itsm.privacy.rules import RuleBasedRecognizer

logger = get_logger("privacy.anonymiser")

Backend = Literal["rules", "transformer", "presidio"]

#: Free-text columns anonymised by default. Structured columns are not touched:
#: they are categorical values from a closed vocabulary, not free text.
TEXT_COLUMNS: tuple[str, ...] = ("designation", "description")


class Recognizer(Protocol):
    """What every detector must provide."""

    name: str

    def detect(self, text: str) -> list[PiiSpan]:
        """Return disjoint PII spans found in one text."""
        ...

    def detect_batch(self, texts: list[str]) -> list[list[PiiSpan]]:
        """Return spans for a batch of texts."""
        ...


def build_recognizer(
    backend: Backend = "rules",
    model_name: str | None = None,
    language: str | None = None,
    with_structural: bool | None = None,
) -> Recognizer:
    """Instantiate the requested detector.

    Args:
        backend: ``rules`` (offline), ``transformer`` (needs the ``hf`` extra)
            or ``presidio`` (needs the ``presidio`` extra).
        model_name: Override of the model this backend loads - the Hugging Face
            NER model for ``transformer``, the spaCy pipeline for ``presidio``.
            The two are not interchangeable, which is why
            :meth:`itsm.config.Settings.pii_model` picks the right one.
        language: Language the analyser is configured for, ``presidio`` only.
        with_structural: Force the rule engine's structured layer on or off
            underneath a model-based backend. ``None`` keeps each backend's own
            default - composed for ``transformer``, standalone for
            ``presidio``. Pass ``False`` to benchmark every backend in
            isolation, which is what ``itsm privacy eval --isolate`` does.

    Returns:
        A ready-to-use recogniser. Models are loaded lazily, on the first
        detection call, not here.
    """
    options: dict[str, Any] = (
        {} if with_structural is None else {"with_structural": with_structural}
    )
    if backend == "transformer":
        from itsm.privacy.transformer import TransformerRecognizer

        if model_name:
            return TransformerRecognizer(model_name, **options)
        return TransformerRecognizer(**options)
    if backend == "presidio":
        from itsm.privacy.presidio import DEFAULT_LANGUAGE, PresidioRecognizer

        return PresidioRecognizer(
            language=language or DEFAULT_LANGUAGE, spacy_model=model_name, **options
        )
    return RuleBasedRecognizer()


@dataclass
class AnonymisationOutcome:
    """Result of anonymising a corpus, for one text column."""

    column: str
    original: list[str]
    anonymised: list[str]
    spans: list[list[PiiSpan]]
    mappings: list[dict[str, str]]

    @property
    def n_spans(self) -> int:
        """Total number of spans replaced."""
        return sum(len(item) for item in self.spans)


class Anonymiser:
    """Detect and replace personal data across ticket text.

    Args:
        backend: Detector to use.
        policy: Per-entity policy overrides.
        model_name: Model override for the backend that loads one - the NER
            model for ``transformer``, the spaCy pipeline for ``presidio``.
        language: Language the analyser is configured for, ``presidio`` only.
        with_structural: Force the rule layer on or off under a model-based
            backend; ``None`` keeps the backend's own default.
    """

    def __init__(
        self,
        backend: Backend = "rules",
        policy: dict[Any, Policy] | None = None,
        model_name: str | None = None,
        language: str | None = None,
        with_structural: bool | None = None,
    ) -> None:
        self.backend = backend
        self.recognizer = build_recognizer(backend, model_name, language, with_structural)
        self.redactor = Redactor(policy=policy)

    def anonymise_text(self, text: str) -> tuple[str, list[PiiSpan], dict[str, str]]:
        """Anonymise one string.

        Returns:
            ``(anonymised_text, detected_spans, replacements)``.
        """
        spans = self.recognizer.detect(text or "")
        result = self.redactor.apply(text or "", spans)
        return result.text, spans, result.replacements

    def anonymise_column(self, texts: list[str]) -> AnonymisationOutcome:
        """Anonymise a list of texts, batching the detection."""
        cleaned = [str(text) if text is not None else "" for text in texts]
        spans = self.recognizer.detect_batch(cleaned)
        anonymised: list[str] = []
        mappings: list[dict[str, str]] = []

        for text, detected in zip(cleaned, spans, strict=True):
            result = self.redactor.apply(text, detected)
            anonymised.append(result.text)
            mappings.append(result.replacements)

        return AnonymisationOutcome(
            column="", original=cleaned, anonymised=anonymised, spans=spans, mappings=mappings
        )

    def anonymise_frame(
        self,
        tickets: pd.DataFrame,
        columns: tuple[str, ...] = TEXT_COLUMNS,
        keep_original: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, AnonymisationOutcome]]:
        """Anonymise every free-text column of a ticket frame.

        Args:
            tickets: Validated ticket frame.
            columns: Free-text columns to rewrite.
            keep_original: Also keep the raw text under ``<column>_raw``. Off
                by default - an "anonymised" extract that still carries the
                original alongside protects nobody, so keeping it must be an
                explicit, auditable decision.

        Returns:
            ``(frame, outcomes)`` where the frame's text columns are rewritten.
        """
        frame = tickets.copy()
        outcomes: dict[str, AnonymisationOutcome] = {}

        for column in columns:
            if column not in frame.columns:
                continue
            outcome = self.anonymise_column(frame[column].tolist())
            outcome.column = column
            outcomes[column] = outcome
            if keep_original:
                frame[f"{column}_raw"] = outcome.original
            frame[column] = outcome.anonymised
            logger.info(
                "anonymised %s: %d spans over %d rows (%s)",
                column,
                outcome.n_spans,
                len(frame),
                self.backend,
            )
        return frame, outcomes
