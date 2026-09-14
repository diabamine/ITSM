"""Measuring how good an anonymisation actually is.

Anonymisation is usually shipped on a demo ("look, it found the name"). This
module scores it, and it does so on the axes that carry consequences - which
are not the axes a generic NER benchmark reports.

**Three levels of recall, because they mean different things.**

``strict``
    Exact character boundaries and the right entity type. The academic number.
``partial``
    Boundaries overlap enough (IoU) and the type is right. Detecting
    ``Dupont`` where the truth is ``Jean Dupont`` lands here - and it is *not*
    a success, because ``Jean`` survives in the output.
``coverage`` (any type, any overlap)
    Was the personal data touched at all? Mislabelling an email as a person
    still removes it, so **this is the privacy-relevant recall** and the one to
    read first.

**Residual PII is the ground truth of the ground truth.** Span arithmetic can
flatter a detector whose boundaries are sloppy, so the final check ignores
spans entirely and searches the *output* text for the original values. If
``Jean Dupont`` is still in there, nothing else matters. That is
:func:`residual_pii_rate`.

**Over-redaction is the cost side.** A detector that redacts everything scores
a perfect recall and destroys the corpus, so false positives are reported both
as a precision and as a share of characters removed - and the pipeline
additionally retrains the classifier on the anonymised text to put a number on
what the redaction cost downstream.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from itsm.privacy.entities import DIRECT_IDENTIFIERS, EntityType, PiiSpan

#: Minimum intersection-over-union for a prediction to "partially" match.
DEFAULT_IOU = 0.5


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio, or 0.0 when there is nothing to divide."""
    return round(numerator / denominator, 4) if denominator else 0.0


def match_spans(
    reference: Sequence[PiiSpan], predicted: Sequence[PiiSpan], iou_threshold: float = DEFAULT_IOU
) -> dict[str, Any]:
    """Match one document's predictions against its reference spans.

    Matching is greedy by overlap, one prediction per reference: two
    predictions covering the same reference count once, and the extra one is a
    false positive rather than a free true positive.

    Args:
        reference: Ground-truth spans.
        predicted: Detected spans.
        iou_threshold: Overlap required for a ``partial`` match.

    Returns:
        Counts per level plus the lists of missed and spurious spans.
    """
    used: set[int] = set()
    strict = partial = covered = 0
    missed: list[PiiSpan] = []

    for truth in reference:
        best_index, best_iou = -1, 0.0
        for index, prediction in enumerate(predicted):
            if index in used or not truth.overlaps(prediction):
                continue
            overlap = truth.iou(prediction)
            if overlap > best_iou:
                best_index, best_iou = index, overlap

        if best_index < 0:
            missed.append(truth)
            continue

        used.add(best_index)
        prediction = predicted[best_index]
        covered += 1
        if prediction.entity is truth.entity:
            if best_iou >= iou_threshold:
                partial += 1
            if prediction.start == truth.start and prediction.end == truth.end:
                strict += 1

    spurious = [span for index, span in enumerate(predicted) if index not in used]
    return {
        "n_reference": len(reference),
        "n_predicted": len(predicted),
        "strict": strict,
        "partial": partial,
        "covered": covered,
        "missed": missed,
        "spurious": spurious,
    }


def score_detection(
    references: Sequence[Sequence[PiiSpan]],
    predictions: Sequence[Sequence[PiiSpan]],
    texts: Sequence[str] | None = None,
    iou_threshold: float = DEFAULT_IOU,
) -> dict[str, Any]:
    """Aggregate detection quality over a corpus.

    Args:
        references: Ground-truth spans, one list per document.
        predictions: Detected spans, aligned with ``references``.
        texts: The documents, used to normalise over-redaction by length.
        iou_threshold: Overlap required for a ``partial`` match.

    Returns:
        Corpus metrics, per-entity metrics, and the most frequent misses and
        false positives - the two lists an engineer actually iterates on.
    """
    totals: Counter[str] = Counter()
    per_type: dict[str, Counter[str]] = defaultdict(Counter)
    missed_samples: Counter[str] = Counter()
    spurious_samples: Counter[str] = Counter()

    for reference, predicted in zip(references, predictions, strict=True):
        result = match_spans(reference, predicted, iou_threshold)
        for key in ("n_reference", "n_predicted", "strict", "partial", "covered"):
            totals[key] += int(result[key])

        for span in reference:
            per_type[str(span.entity)]["reference"] += 1
        for span in predicted:
            per_type[str(span.entity)]["predicted"] += 1

        # Per-type hits: re-match inside each type so a type's recall is not
        # inflated by a cross-type overlap.
        for entity in {span.entity for span in reference} | {span.entity for span in predicted}:
            typed = match_spans(
                [s for s in reference if s.entity is entity],
                [s for s in predicted if s.entity is entity],
                iou_threshold,
            )
            per_type[str(entity)]["partial_hits"] += int(typed["partial"])

        for span in result["missed"]:
            totals["missed"] += 1
            if span.entity in DIRECT_IDENTIFIERS:
                totals["missed_direct"] += 1
            missed_samples[f"{span.entity}: {span.text}"] += 1
        for span in result["spurious"]:
            totals["spurious"] += 1
            totals["spurious_chars"] += span.length
            spurious_samples[f"{span.entity}: {span.text}"] += 1

    for span_list in references:
        for span in span_list:
            if span.entity in DIRECT_IDENTIFIERS:
                totals["reference_direct"] += 1

    n_reference = totals["n_reference"]
    n_predicted = totals["n_predicted"]
    total_chars = sum(len(text) for text in texts) if texts else 0

    entity_metrics = {
        entity: {
            "support": counts["reference"],
            "predicted": counts["predicted"],
            "recall": _safe_ratio(counts["partial_hits"], counts["reference"]),
            "precision": _safe_ratio(counts["partial_hits"], counts["predicted"]),
        }
        for entity, counts in sorted(per_type.items())
    }

    return {
        "n_documents": len(references),
        "n_reference_spans": n_reference,
        "n_predicted_spans": n_predicted,
        "recall_strict": _safe_ratio(totals["strict"], n_reference),
        "recall_partial": _safe_ratio(totals["partial"], n_reference),
        "recall_coverage": _safe_ratio(totals["covered"], n_reference),
        "precision_partial": _safe_ratio(totals["partial"], n_predicted),
        "precision_coverage": _safe_ratio(totals["covered"], n_predicted),
        "f1_partial": _safe_ratio(2 * totals["partial"], n_reference + n_predicted),
        "missed_spans": totals["missed"],
        "missed_direct_identifiers": totals["missed_direct"],
        "leak_rate_direct_identifiers": _safe_ratio(
            totals["missed_direct"], totals["reference_direct"]
        ),
        "false_positive_spans": totals["spurious"],
        "over_redacted_chars": totals["spurious_chars"],
        "over_redacted_char_share": _safe_ratio(totals["spurious_chars"], total_chars),
        "per_entity": entity_metrics,
        "top_misses": missed_samples.most_common(10),
        "top_false_positives": spurious_samples.most_common(10),
    }


#: Minimum length of a token worth treating as a partial leak on its own.
_MIN_TOKEN = 4


#: Types where a surviving token really is a leak. A street or city word from
#: an address recurs innocently all over a corpus ("Paris"), so counting it
#: would drown the signal; a surviving surname never does.
_PARTIAL_CHECKED = frozenset({EntityType.PERSON, EntityType.EMAIL})


def _residual_tokens(value: str, text: str, entity: EntityType | None = None) -> list[str]:
    """Return the value's own tokens that survive verbatim in ``text``.

    ``Jean <PERSON>`` contains none of ``Jean Dupont`` as a whole, yet the
    first name is right there. Whole-value search alone would call that a
    success, so multi-token identifiers are also checked token by token.
    """
    tokens = re.findall(rf"[\w.@+-]{{{_MIN_TOKEN},}}", value)
    if len(tokens) < 2:
        return []
    return [token for token in tokens if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text)]


def residual_pii_rate(
    anonymised: Sequence[str], references: Sequence[Sequence[PiiSpan]]
) -> dict[str, Any]:
    """Search the *output* for the original values - the end-to-end check.

    Span metrics measure the detector; this measures the deliverable. Two
    levels, because a detector can fail at either:

    ``residual_rate``
        The whole value is still in the output. An outright miss.
    ``residual_partial_rate``
        Part of a multi-token value survives - the classic sloppy-boundary
        failure, where ``Jean Dupont`` is redacted from ``Dupont`` onwards and
        ``Jean`` stays behind. Span arithmetic scores that as a hit; a reader
        of the output would still know who it is.

    Args:
        anonymised: The rewritten texts.
        references: Ground-truth spans of the corresponding originals.

    Returns:
        Leak counts overall, for direct identifiers, and per entity type, with
        examples.
    """
    leaked: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    examples: list[dict[str, str]] = []

    for text, spans in zip(anonymised, references, strict=True):
        for span in spans:
            value = span.text.strip()
            if len(value) < 3:  # too short to assert anything meaningful
                continue
            direct = span.entity in DIRECT_IDENTIFIERS
            totals[str(span.entity)] += 1
            totals["all"] += 1
            if direct:
                totals["direct"] += 1

            if value in text:
                leaked[str(span.entity)] += 1
                leaked["all"] += 1
                if direct:
                    leaked["direct"] += 1
                if len(examples) < 10:
                    examples.append({"entity": str(span.entity), "value": value, "kind": "full"})
                continue

            surviving = (
                _residual_tokens(value, text, span.entity)
                if span.entity in _PARTIAL_CHECKED
                else []
            )
            if surviving:
                leaked["partial"] += 1
                if direct:
                    leaked["partial_direct"] += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "entity": str(span.entity),
                            "value": " ".join(surviving),
                            "kind": "partial",
                        }
                    )

    return {
        "checked_values": totals["all"],
        "residual_values": leaked["all"],
        "residual_rate": _safe_ratio(leaked["all"], totals["all"]),
        "residual_rate_direct_identifiers": _safe_ratio(leaked["direct"], totals["direct"]),
        "residual_partial_values": leaked["partial"],
        "residual_partial_rate": _safe_ratio(leaked["partial"], totals["all"]),
        "residual_partial_rate_direct_identifiers": _safe_ratio(
            leaked["partial_direct"], totals["direct"]
        ),
        "per_entity": {
            entity: {
                "checked": totals[entity],
                "residual": leaked[entity],
                "rate": _safe_ratio(leaked[entity], totals[entity]),
            }
            for entity in sorted(set(totals) - {"all", "direct"})
        },
        "examples": examples,
    }


def spans_from_payload(payload: Mapping[str, Any], field: str) -> list[PiiSpan]:
    """Rebuild the ground-truth spans of one field from the stored JSON."""
    return [PiiSpan.from_dict(item) for item in payload.get(field, [])]


def entity_distribution(references: Sequence[Sequence[PiiSpan]]) -> dict[str, int]:
    """Count reference spans per entity type - the support of the evaluation."""
    counter: Counter[str] = Counter()
    for spans in references:
        for span in spans:
            counter[str(span.entity)] += 1
    return dict(sorted(counter.items(), key=lambda item: -item[1]))


__all__ = [
    "DEFAULT_IOU",
    "EntityType",
    "entity_distribution",
    "match_spans",
    "residual_pii_rate",
    "score_detection",
    "spans_from_payload",
]
