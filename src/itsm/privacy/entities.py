"""Entity vocabulary shared by the detectors, the redactor and the scorer.

One character-offset span type, one entity enumeration, and the span algebra
(overlap, merge, sort) that every component needs. Keeping it here means a
detector can be swapped without touching the evaluation, and vice versa.

Offsets are **character offsets into the original string**, half-open
``[start, end)``, exactly like Python slicing. That choice matters: token
offsets would tie the ground truth to one tokeniser, and a rule engine and a
transformer never tokenise the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class EntityType(StrEnum):
    """Categories of personal data an ITSM free-text field realistically holds.

    The list is deliberately operational rather than exhaustive: these are the
    identifiers that actually turn up in a ticket body, ranked by how badly
    they identify someone.
    """

    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    EMPLOYEE_ID = "EMPLOYEE_ID"
    NATIONAL_ID = "NATIONAL_ID"
    IBAN = "IBAN"
    CREDIT_CARD = "CREDIT_CARD"
    ADDRESS = "ADDRESS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    IP_ADDRESS = "IP_ADDRESS"


#: Direct identifiers: one value is enough to name a person. A miss here is a
#: breach, which is why the evaluation reports their recall separately.
DIRECT_IDENTIFIERS = frozenset(
    {
        EntityType.PERSON,
        EntityType.EMAIL,
        EntityType.PHONE,
        EntityType.NATIONAL_ID,
        EntityType.IBAN,
        EntityType.CREDIT_CARD,
    }
)


@dataclass(frozen=True)
class PiiSpan:
    """One piece of personal data located in a text.

    Attributes:
        start: First character offset, inclusive.
        end: Last character offset, exclusive.
        entity: What kind of identifier it is.
        text: The matched substring, kept for reporting and error analysis.
        score: Detector confidence in ``[0, 1]``. Rule matches are 1.0 when
            the pattern is unambiguous (an IBAN checksum) and lower when the
            match rests on a heuristic (a capitalised word after "contacter").
        source: Which detector produced it, for error analysis.
    """

    start: int
    end: int
    entity: EntityType
    text: str = ""
    score: float = 1.0
    source: str = ""

    def __post_init__(self) -> None:
        """Reject impossible spans early: a bad offset corrupts everything."""
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span offsets: [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        """Number of characters covered."""
        return self.end - self.start

    def overlaps(self, other: PiiSpan) -> bool:
        """Return whether two spans share at least one character."""
        return self.start < other.end and other.start < self.end

    def iou(self, other: PiiSpan) -> float:
        """Intersection over union of the two character ranges.

        Used by the evaluation to decide whether a prediction "found" a
        reference span when the boundaries differ slightly - detecting
        ``Dupont`` where the truth is ``Jean Dupont`` is a partial success,
        not a miss and not a hit.
        """
        intersection = min(self.end, other.end) - max(self.start, other.start)
        if intersection <= 0:
            return 0.0
        union = max(self.end, other.end) - min(self.start, other.start)
        return intersection / union

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view, used to persist the ground truth."""
        payload = asdict(self)
        payload["entity"] = str(self.entity)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PiiSpan:
        """Rebuild a span from its serialised form."""
        return cls(
            start=int(payload["start"]),
            end=int(payload["end"]),
            entity=EntityType(str(payload["entity"])),
            text=str(payload.get("text", "")),
            score=float(payload.get("score", 1.0)),
            source=str(payload.get("source", "")),
        )


def merge_spans(spans: list[PiiSpan]) -> list[PiiSpan]:
    """Resolve overlapping detections into a non-overlapping, sorted list.

    Detectors disagree on boundaries and fire twice on the same text (an email
    contains a name; a rule and a model both flag ``Jean Dupont``). Replacing a
    substring is only well defined on disjoint spans, so overlaps are resolved
    by keeping the **longest** span, then the highest score: over-redacting a
    character is a utility cost, leaving one is a privacy incident.

    Args:
        spans: Possibly overlapping spans, in any order.

    Returns:
        Disjoint spans sorted by ``start``.
    """
    ordered = sorted(spans, key=lambda s: (-s.length, -s.score, s.start))
    kept: list[PiiSpan] = []
    for span in ordered:
        if not any(span.overlaps(other) for other in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)


def apply_replacements(text: str, replacements: list[tuple[PiiSpan, str]]) -> str:
    """Substitute spans in a text, right to left so offsets stay valid.

    Args:
        text: The original string.
        replacements: Pairs of (span, replacement). Spans must be disjoint.

    Returns:
        The rewritten string.
    """
    result = text
    for span, replacement in sorted(replacements, key=lambda pair: pair[0].start, reverse=True):
        result = result[: span.start] + replacement + result[span.end :]
    return result
