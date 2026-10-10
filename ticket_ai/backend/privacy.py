"""Light masking of personal data before text is sent to an external API.

Regular expressions catch what has a **format**: emails, phone numbers, IBANs,
card numbers. They cannot catch **names**, which have none. That limitation is
real: a ticket saying "contacter Camille Dubois" still carries a name after this
step.

For a production deployment, replace :func:`redact_pii` with a proper detector
(Presidio, GLiNER...). The signature is the only contract the rest of the code
relies on, so the swap is local.
"""

from __future__ import annotations

import re

#: Order matters: the most specific patterns run first, so an IBAN is not
#: half-eaten by the phone pattern.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,4})?\b")),
    ("CREDIT_CARD", re.compile(r"\b\d{4}(?:[ -]\d{4}){3}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+\d{1,3}[ .-]?|0)[1-9](?:[ .-]?\d{2}){4}(?!\d)")),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace formatted personal data with an entity tag.

    Args:
        text: Free text.

    Returns:
        ``(masked_text, entity_types_found)``. Types are listed once each, in
        the order of :data:`_PATTERNS`.
    """
    found: list[str] = []
    for entity, pattern in _PATTERNS:
        text, count = pattern.subn(f"<{entity}>", text)
        if count:
            found.append(entity)
    return text, found
