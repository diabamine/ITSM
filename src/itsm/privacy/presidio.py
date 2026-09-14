"""Presidio PII detection - the standard analyser option.

Microsoft Presidio is the reference open-source PII toolkit: an
``AnalyzerEngine`` that runs a spaCy pipeline for the entities that need
language understanding and a set of pattern recognisers for the ones that have
a shape. The rule engine here is authored for this corpus and the transformer
backend is one NER model; Presidio is a framework other teams already audit and
extend, which is frequently the argument that decides a security review.

Two things it does better than either existing backend:

* **Validated structured identifiers.** ``IBAN_CODE`` and ``CREDIT_CARD`` are
  checksum-verified rather than shape-matched, so a 16-digit order number that
  the regex in :mod:`~itsm.privacy.rules` accepts is rejected here.
* **A context layer.** Presidio raises a match's score when surrounding words
  support it - the same idea as the rule engine's triggers, generalised and
  maintained upstream.

**This backend runs Presidio alone.** The transformer backend composes itself
with the rule engine's structured layer; this one deliberately does not,
because a composed detector cannot be benchmarked - every number it produces is
the union of two systems, and a gap in one is invisible while the other covers
it. Isolated, ``presidio`` answers the question a benchmark is actually asked:
*what does this tool find on its own?* The composition is still available
(``with_structural=True``) once the choice has been made rather than while it
is being made.

The consequence is stated rather than hidden, because it is exactly what the
benchmark should surface: Presidio's predefined recognisers know nothing about
a French *matricule*, a NIR, or the street-address shapes of this corpus, so
standalone it will miss them. That is a finding, not a defect to paper over.

**One language at a time.** An ``AnalyzerEngine`` is built around one spaCy
pipeline, and nothing in a ticket declares its language. On this bilingual
corpus that is a real cost, not a detail: French text analysed by an English
pipeline loses the French-only names, and the per-entity evaluation in
``docs/07-privacy.md`` is where that shows up. Set ``ITSM_PRESIDIO_LANGUAGE``
and, if the default is not installed, ``ITSM_PRESIDIO_SPACY_MODEL``.

Requires the ``presidio`` extra and a one-time spaCy model download - the trade
against the rule engine, which needs neither.
"""

from __future__ import annotations

from typing import Any

from itsm.logging import get_logger
from itsm.privacy.entities import EntityType, PiiSpan, merge_spans
from itsm.privacy.rules import RuleBasedRecognizer

logger = get_logger("privacy.presidio")

DEFAULT_LANGUAGE = "en"

#: spaCy pipeline used for a language when none is configured. The large
#: models are named because the small ones cost noticeably more ``PERSON``
#: recall; either works, and ``ITSM_PRESIDIO_SPACY_MODEL`` overrides both.
DEFAULT_SPACY_MODELS: dict[str, str] = {
    "en": "en_core_web_lg",
    "fr": "fr_core_news_lg",
    "de": "de_core_news_lg",
    "es": "es_core_news_lg",
}

#: Presidio entity name -> our entity type. Everything absent from this map is
#: dropped, and three omissions are deliberate:
#:
#: * ``DATE_TIME`` - a bare date is not personal data. Only a date introduced
#:   by a birth trigger is, and the rule layer already gates on that.
#: * ``ORGANIZATION`` / ``NRP`` - in ticket text these are the employer and its
#:   vendors ("SAP", "Microsoft"); redacting them would destroy the vocabulary
#:   the classifier lives on.
#: * ``URL`` - an intranet link is routing information, not an identifier.
ENTITY_MAP: dict[str, EntityType] = {
    "PERSON": EntityType.PERSON,
    "EMAIL_ADDRESS": EntityType.EMAIL,
    "PHONE_NUMBER": EntityType.PHONE,
    "IBAN_CODE": EntityType.IBAN,
    "CREDIT_CARD": EntityType.CREDIT_CARD,
    "IP_ADDRESS": EntityType.IP_ADDRESS,
    "LOCATION": EntityType.ADDRESS,
}

#: Trailing characters an analyser routinely swallows into a span. Trimming
#: them keeps offsets comparable with the reference spans, which are tight.
_EDGE_CHARS = " \t\n.,;:!?()[]<>\"'"


def _tighten(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink ``[start, end)`` onto its non-padding content.

    Returns:
        The tightened offsets, or ``None`` if nothing is left.
    """
    surface = text[start:end]
    stripped = surface.strip(_EDGE_CHARS)
    if not stripped:
        return None
    offset = start + surface.index(stripped)
    return offset, offset + len(stripped)


class PresidioRecognizer:
    """Presidio ``AnalyzerEngine`` for language and formats, rules for the rest.

    Args:
        language: Language code the analyser is configured for. One engine
            serves one language; see the module docstring.
        spacy_model: spaCy pipeline to load. Defaults to the entry in
            :data:`DEFAULT_SPACY_MODELS` for ``language``.
        min_score: Minimum Presidio confidence for a detection to be kept.
            Passed to the engine as its score threshold, so filtered matches
            are never built into spans.
        with_structural: Add the rule engine's structured-identifier layer for
            the entities Presidio has no recogniser for (NIR, matricule,
            address, date of birth). **Off by default**: this backend exists to
            be measured on its own, and a composed detector cannot be. Turn it
            on for a deployment that has already chosen Presidio and wants the
            local identifiers covered too.
        map_locations: Treat ``LOCATION`` predictions as addresses. Off by
            default: in ITSM text most locations are office sites
            ("Toulouse"), which are not personal data - the structured address
            pattern already catches real postal addresses.
    """

    name = "presidio"

    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        spacy_model: str | None = None,
        min_score: float = 0.4,
        with_structural: bool = False,
        map_locations: bool = False,
    ) -> None:
        self.language = language
        self.spacy_model = spacy_model or DEFAULT_SPACY_MODELS.get(language)
        self.min_score = min_score
        self.with_structural = with_structural
        self.map_locations = map_locations
        self._engine: Any | None = None
        self._structural = RuleBasedRecognizer(detect_persons=False)

    @property
    def entities(self) -> list[str]:
        """Presidio entity names to ask for - the mapped ones, nothing else."""
        return [
            name
            for name, entity in ENTITY_MAP.items()
            if self.map_locations or entity is not EntityType.ADDRESS
        ]

    def _load(self) -> Any:
        """Build the analyser on first use, with an actionable error."""
        if self._engine is not None:
            return self._engine
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
        except ImportError as error:  # pragma: no cover - depends on the extra
            raise RuntimeError(
                "the presidio backend needs the optional dependencies: "
                "`uv sync --extra presidio`. The rule-based backend needs nothing."
            ) from error

        if self.spacy_model is None:
            raise RuntimeError(
                f"no default spaCy pipeline is known for language {self.language!r}; "
                "set ITSM_PRESIDIO_SPACY_MODEL to the pipeline to load."
            )

        logger.info("loading presidio analyser (%s, %s)", self.language, self.spacy_model)
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": self.language, "model_name": self.spacy_model}],
            }
        )
        try:
            nlp_engine = provider.create_engine()
        except OSError as error:  # pragma: no cover - depends on the download
            raise RuntimeError(
                f"spaCy pipeline {self.spacy_model!r} is not installed: run "
                f"`uv run python -m spacy download {self.spacy_model}`."
            ) from error

        self._engine = AnalyzerEngine(
            nlp_engine=nlp_engine, supported_languages=[self.language]
        )
        return self._engine

    def _to_spans(self, text: str, results: list[Any]) -> list[PiiSpan]:
        """Convert analyser results into spans, filtering and tightening them."""
        spans: list[PiiSpan] = []
        for result in results:
            tag = str(result.entity_type).upper()
            mapped = ENTITY_MAP.get(tag)
            if mapped is None:
                continue
            if mapped is EntityType.ADDRESS and not self.map_locations:
                continue

            tight = _tighten(text, int(result.start), int(result.end))
            if tight is None:
                continue
            start, end = tight
            spans.append(
                PiiSpan(
                    start=start,
                    end=end,
                    entity=mapped,
                    text=text[start:end],
                    score=round(float(result.score), 4),
                    source=f"{self.name}:{tag}",
                )
            )
        return spans

    def detect(self, text: str) -> list[PiiSpan]:
        """Return the disjoint PII spans found in ``text``."""
        if not text or not text.strip():
            return []

        results = self._load().analyze(
            text=text,
            language=self.language,
            entities=self.entities,
            score_threshold=self.min_score,
        )
        spans = self._to_spans(text, list(results))
        if self.with_structural:
            spans.extend(self._structural.detect(text))
        return merge_spans(spans)

    def detect_batch(self, texts: list[str]) -> list[list[PiiSpan]]:
        """Detect over a batch.

        Sequential on purpose: ``AnalyzerEngine.analyze`` takes one document
        and spaCy is invoked per document either way, so there is no batched
        inference win to collect here - unlike the transformer backend.
        """
        return [self.detect(text or "") for text in texts]
