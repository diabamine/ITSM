"""Replacing detected spans with an entity tag.

Every identifier becomes the name of its kind: ``Camille Dubois`` becomes
``<PERSON>``, a mobile number becomes ``<PHONE>``. Nothing of the value
survives, and nothing plausible takes its place.

**Why tags rather than realistic fake values.** A surrogate that looks like a
real name is indistinguishable from a real name to everyone downstream: the
agent reading the ticket, the analyst reading an export, the reviewer auditing
the output. A leak and a correct redaction look identical, which is precisely
the property you do not want in a privacy control. ``<PERSON>`` is unmistakable
and self-documenting - you can see at a glance that the field was processed,
and how.

The cost is stated rather than hidden: an entity tag carries **no identity**,
so two tickets from the same requester are no longer linkable through the
text. If cross-ticket linkage matters for an analysis, it has to come from a
structured key (the caller id column), not from the body - which is where it
should have come from in the first place.

Per-entity policy stays configurable: :class:`Policy` also offers ``KEEP`` for
a type a given deployment judges non-identifying, and the decision is data
rather than code.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from itsm.privacy.entities import EntityType, PiiSpan, apply_replacements


class Policy(StrEnum):
    """What to do with a detected span."""

    REDACT = "redact"
    """Replace with the entity tag; the value is destroyed."""
    KEEP = "keep"
    """Leave untouched; for a type a deployment judges non-identifying."""


#: Default policy per entity type: everything is redacted. Keeping a type is a
#: deliberate, auditable decision a deployment makes explicitly.
DEFAULT_POLICY: dict[EntityType, Policy] = dict.fromkeys(EntityType, Policy.REDACT)


def tag_for(entity: EntityType) -> str:
    """Return the placeholder written in place of an ``entity`` value."""
    return f"<{entity}>"


@dataclass
class RedactionResult:
    """Outcome of redacting one text."""

    text: str
    """The rewritten text."""
    spans: list[PiiSpan] = field(default_factory=list)
    """Spans that were detected in the original text."""
    replacements: dict[str, str] = field(default_factory=dict)
    """Original surface form -> tag, for this text only."""

    @property
    def n_replaced(self) -> int:
        """Number of substitutions applied."""
        return len(self.replacements)


class Redactor:
    """Replace detected spans with their entity tag.

    Args:
        policy: Per-entity overrides of :data:`DEFAULT_POLICY`.
    """

    def __init__(self, policy: dict[EntityType, Policy] | None = None) -> None:
        self.policy = {**DEFAULT_POLICY, **(policy or {})}

    def replacement_for(self, entity: EntityType, value: str) -> str:
        """Return what replaces one detected value under the policy."""
        if self.policy.get(entity, Policy.REDACT) is Policy.KEEP:
            return value
        return tag_for(entity)

    def apply(self, text: str, spans: Iterable[PiiSpan]) -> RedactionResult:
        """Rewrite ``text``, replacing every span according to the policy.

        Args:
            text: Original text.
            spans: Disjoint spans, as returned by a recogniser.

        Returns:
            The result, carrying the rewritten text and the substitutions made.
        """
        ordered = sorted(spans, key=lambda span: span.start)
        replacements: list[tuple[PiiSpan, str]] = []
        mapping: dict[str, str] = {}

        for span in ordered:
            surface = text[span.start : span.end]
            tag = self.replacement_for(span.entity, surface)
            if tag == surface:
                continue
            replacements.append((span, tag))
            mapping[surface] = tag

        return RedactionResult(
            text=apply_replacements(text, replacements), spans=ordered, replacements=mapping
        )
