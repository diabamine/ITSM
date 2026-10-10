"""GLiNER PII detection - the zero-shot option.

GLiNER asks for entity types *by name* at inference time. Where the transformer
backend is stuck with whatever its checkpoint was trained on (PER/ORG/LOC) and
Presidio is stuck with its recogniser catalogue, GLiNER is handed a list of
strings - ``["person", "employee id", "social security number"]`` - and looks
for those. For an ITSM corpus that is the interesting property: the entity
vocabulary in :mod:`itsm.privacy.entities` was chosen for this domain, and this
is the only backend that can be *told* about it rather than approximated to it.

The trade is the usual zero-shot one: it will find a *matricule* because the
prompt said "employee id", but with none of the guarantees a regex gives. An
IBAN here is whatever the model thinks looks like one - no checksum, no
validation - so a high score is not the same claim as a Presidio
``IBAN_CODE`` match.

**It runs alone.** Same reason as the Presidio backend: a composed detector
cannot be benchmarked, because a gap in one layer is covered by the other and
the score measures the pair. ``with_structural=True`` composes the rule layer
back in, for a deployment that has already chosen GLiNER.

Two generations are supported, because both are in use:

``gliner2``
    The newer package. Tried first.
``gliner``
    The original. ``urchade/gliner_multi_pii-v1`` is multilingual and tuned for
    exactly this task, which makes it the better default for an FR/EN corpus.

.. warning::
   **This adapter has not been executed against either library.** It is written
   against their documented interfaces and normalises defensively - it accepts
   the span shapes both are documented to return and recovers character offsets
   by search when a shape carries only the surface string. Treat the first run
   as part of the experiment: if it raises, the shape it received is in the
   error message, and :func:`_normalise` is the one function to fix.

The checkpoint is fetched from the hub on first use.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from itsm.logging import get_logger
from itsm.privacy.entities import EntityType, PiiSpan, merge_spans
from itsm.privacy.rules import RuleBasedRecognizer

logger = get_logger("privacy.gliner")

DEFAULT_MODEL = "urchade/gliner_multi_pii-v1"

#: Natural-language entity names handed to the model, mapped back onto our
#: vocabulary. The wording matters - it *is* the prompt - and these were chosen
#: to read like the labels the public PII checkpoints were tuned on.
PROMPT_LABELS: dict[str, EntityType] = {
    "person": EntityType.PERSON,
    "full name": EntityType.PERSON,
    "email address": EntityType.EMAIL,
    "phone number": EntityType.PHONE,
    "employee id": EntityType.EMPLOYEE_ID,
    "social security number": EntityType.NATIONAL_ID,
    "iban": EntityType.IBAN,
    "credit card number": EntityType.CREDIT_CARD,
    "postal address": EntityType.ADDRESS,
    "date of birth": EntityType.DATE_OF_BIRTH,
    "ip address": EntityType.IP_ADDRESS,
}

#: Trailing characters a span may swallow; trimmed so offsets stay comparable
#: with the reference, which is tight.
_EDGE_CHARS = " \t\n.,;:!?()[]<>\"'"


def _tighten(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink ``[start, end)`` onto its non-padding content, or ``None``."""
    surface = text[start:end]
    stripped = surface.strip(_EDGE_CHARS)
    if not stripped:
        return None
    offset = start + surface.index(stripped)
    return offset, offset + len(stripped)


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> tuple[Any, str]:
    """Load the checkpoint once per process, newer package first.

    Returns:
        ``(model, flavour)`` - the flavour decides which entry point to call,
        since the two generations name it differently.
    """
    try:
        from gliner2 import GLiNER2

        logger.info("loading gliner2 model %s", model_name)
        return GLiNER2.from_pretrained(model_name), "gliner2"
    except ImportError:
        from gliner import GLiNER

        logger.info("loading gliner model %s", model_name)
        return GLiNER.from_pretrained(model_name), "gliner"


def _locate(text: str, value: str, taken: set[int]) -> tuple[int, int] | None:
    """Find ``value`` in ``text``, skipping offsets already used.

    Needed because a zero-shot library may return the surface string without
    offsets, and the ground truth is scored on character offsets. A value
    occurring twice must not be reported at the same position twice.
    """
    start = 0
    while True:
        found = text.find(value, start)
        if found < 0:
            return None
        if found not in taken:
            return found, found + len(value)
        start = found + 1


class GlinerRecognizer:
    """Zero-shot PII detector over the project's own entity vocabulary.

    Args:
        model_name: Hub identifier. Defaults to a multilingual PII checkpoint.
        min_score: Minimum model confidence for a span to be kept.
        with_structural: Add the rule engine's structured layer underneath.
            **Off by default** - this backend exists to be measured on its own.
        labels: Override the prompt labels. Changing the wording changes the
            results, which is itself worth an experiment.
        batch_size: Documents per forward pass in :meth:`detect_batch`.
    """

    name = "gliner"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        min_score: float = 0.4,
        with_structural: bool = False,
        labels: dict[str, EntityType] | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.min_score = min_score
        self.with_structural = with_structural
        self.labels = dict(labels or PROMPT_LABELS)
        self.batch_size = batch_size
        self._model: Any | None = None
        self._flavour = ""
        self._structural = RuleBasedRecognizer(detect_persons=False)

    def _load(self) -> Any:
        """Return the model, loading it at most once per process."""
        if self._model is None:
            self._model, self._flavour = _load_model(self.model_name)
        return self._model

    def _predict(self, text: str) -> Any:
        """Call whichever library was loaded, with its own entry point."""
        model = self._load()
        names = list(self.labels)
        if self._flavour == "gliner2":
            return model.extract_entities(text, names)
        return model.predict_entities(text, names, threshold=self.min_score)

    def _normalise(self, text: str, raw: Any) -> list[PiiSpan]:
        """Turn whatever the library returned into spans over ``text``.

        Three documented shapes are accepted:

        * a list of dicts carrying ``start``/``end`` - used directly;
        * a list of dicts carrying only ``text``/``label`` - offsets recovered
          by search;
        * a mapping of label to values (GLiNER2's schema output) - same.

        Anything else raises with the shape in the message, because guessing
        would silently produce an empty result and a flattering leak rate.
        """
        items: list[tuple[str, str, float, int | None, int | None]] = []

        if isinstance(raw, dict):
            payload = raw.get("entities", raw)
            if not isinstance(payload, dict):
                return self._normalise(text, payload)
            for label, values in payload.items():
                for value in values if isinstance(values, list) else [values]:
                    if isinstance(value, dict):
                        items.append(
                            (
                                str(label),
                                str(value.get("text", "")),
                                float(value.get("score", 1.0)),
                                value.get("start"),
                                value.get("end"),
                            )
                        )
                    else:
                        items.append((str(label), str(value), 1.0, None, None))
        elif isinstance(raw, list):
            for entry in raw:
                if not isinstance(entry, dict):
                    raise RuntimeError(
                        f"unexpected gliner span shape {type(entry).__name__}: {entry!r}"
                    )
                items.append(
                    (
                        str(entry.get("label", entry.get("entity", ""))),
                        str(entry.get("text", "")),
                        float(entry.get("score", 1.0)),
                        entry.get("start"),
                        entry.get("end"),
                    )
                )
        else:
            raise RuntimeError(f"unexpected gliner output shape: {type(raw).__name__}")

        spans: list[PiiSpan] = []
        taken: set[int] = set()
        for label, value, score, start, end in items:
            entity = self.labels.get(label.lower().strip())
            if entity is None or score < self.min_score:
                continue

            if start is None or end is None:
                if not value:
                    continue
                located = _locate(text, value, taken)
                if located is None:
                    continue
                start, end = located

            tight = _tighten(text, int(start), int(end))
            if tight is None:
                continue
            taken.add(tight[0])
            spans.append(
                PiiSpan(
                    start=tight[0],
                    end=tight[1],
                    entity=entity,
                    text=text[tight[0] : tight[1]],
                    score=round(float(score), 4),
                    source=f"{self.name}:{label}",
                )
            )
        return spans

    def detect(self, text: str) -> list[PiiSpan]:
        """Return the disjoint PII spans found in ``text``."""
        if not text or not text.strip():
            return []
        spans = self._normalise(text, self._predict(text))
        if self.with_structural:
            spans.extend(self._structural.detect(text))
        return merge_spans(spans)

    def detect_batch(self, texts: list[str]) -> list[list[PiiSpan]]:
        """Detect over a batch, in one forward pass where the library allows it.

        ``gliner`` exposes ``batch_predict_entities``; one call over 64
        documents is several times faster than 64 calls, because the cost is
        dominated by per-call model overhead rather than sequence length. The
        per-document path stays as a fallback: the batch entry point is absent
        from ``gliner2`` and from older ``gliner`` releases, and a benchmark
        that crashes on a version difference is worse than a slow one.
        """
        cleaned = [text or "" for text in texts]
        model = self._load()
        batched = getattr(model, "batch_predict_entities", None)
        if self._flavour != "gliner" or batched is None:
            return [self.detect(text) for text in cleaned]

        names = list(self.labels)
        results: list[list[PiiSpan]] = []
        for start in range(0, len(cleaned), self.batch_size):
            chunk = cleaned[start : start + self.batch_size]
            try:
                raw = batched(chunk, names, threshold=self.min_score)
            except (TypeError, NotImplementedError):  # pragma: no cover - version drift
                logger.info("batch_predict_entities rejected the call; falling back")
                return [self.detect(text) for text in cleaned]
            for text, entities in zip(chunk, raw, strict=True):
                spans = self._normalise(text, entities)
                if self.with_structural:
                    spans.extend(self._structural.detect(text))
                results.append(merge_spans(spans))
        return results
