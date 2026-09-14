"""Rule-based PII detection - the offline, no-download option.

Nothing here is downloaded, imported from a model hub or trained: the engine is
regular expressions, a small authored gazetteer and a set of contextual
triggers. That is a deliberate capability, not a fallback. In a constrained
environment a detector that needs no network, no GPU and no model licence is
often the only one that can be deployed, and on the *structured* identifiers -
email, phone, IBAN, card, social-security number - it beats a NER model
outright, because those are formats rather than language.

The engine has two layers, and they fail differently:

**Structured identifiers** (email, phone, IBAN, card, national ID, employee
number, IP, postal address). Matched by shape. Precision is essentially 1.0 -
a string matching the IBAN pattern is an IBAN - and recall is limited only by
formats nobody anticipated.

**Person names.** There is no pattern for "this is a name", so three weaker
signals are combined, each with its own confidence:

1. *Contextual triggers* - "contacter", "de la part de", "raised by",
   "M."/"Mme" - followed by a capitalised sequence. High precision: whoever
   wrote "merci de contacter X" was naming a person.
2. *A gazetteer of first names*. A capitalised token in the list, followed by
   another capitalised token, is a first name and a surname.
3. *Bare capitalised sequences* are **not** flagged. In ticket text they are
   overwhelmingly product names, sites and acronyms (``Outlook``, ``Paris``,
   ``SAP``), and flagging them would shred the very vocabulary the classifier
   relies on.

The consequence is stated plainly rather than hidden: this engine misses names
that appear with no trigger and no known first name (the transformer backend
exists for exactly that case), and the evaluation in
:mod:`itsm.privacy.evaluate` quantifies the gap per entity type.
"""

from __future__ import annotations

import re

from itsm.data import catalog as cat
from itsm.privacy.entities import EntityType, PiiSpan, merge_spans

#: Capitalised word sequence, accent-aware, one to three tokens long.
_NAME_CHUNK = r"[A-ZÀ-ÖØ-Þ][\w'À-ɏ-]+(?:\s+[A-ZÀ-ÖØ-Þ][\w'À-ɏ-]+){0,2}"

#: Words that introduce a person, in both languages of the corpus.
_PERSON_TRIGGERS = (
    r"M\.|Mme|Mlle|Monsieur|Madame|Dr\.?",
    r"contacter|appeler|joindre|rappeler|prevenir",
    r"de la part de|pour le compte de|au nom de|ouvert par|saisi par|concerne",
    r"poste de|machine de|compte de|boite (?:aux lettres )?de|mot de passe de",
    # "phone" is deliberately absent: in ITSM text it is a device nine times
    # out of ten ("new phone", "softphone"), and as a trigger it turned every
    # sentence starting after it into a person.
    r"contact|call|reach",
    r"raised by|on behalf of|reported by|requested by|assigned to",
    r"workstation of|mailbox transfer for|password reset for|account creation for",
)
# Two subtleties, both found by measuring false positives:
#   * the trigger is matched case-insensitively, the name is *not* - a global
#     re.IGNORECASE would defeat the capitalisation requirement and label
#     "saisi par erreur" a person, hence the scoped (?i:...) group;
#   * (?<!\w) anchors the trigger to a word start, without which "telephone"
#     ends in the trigger "phone" and "5pm." ends in the trigger "M.".
_PERSON_TRIGGER = re.compile(
    rf"(?<!\w)(?:(?i:{'|'.join(_PERSON_TRIGGERS)}))\s+(?P<name>{_NAME_CHUNK})"
)

#: Triggers that qualify a date as a date of birth. A bare date is not PII.
_BIRTH_TRIGGER = re.compile(
    r"(?:ne(?:e)?\s+(?:de\s+)?le|date de naissance|born(?: on)?|d\.?o\.?b\.?)\s*:?\s*"
    r"(?P<date>\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2})",
    flags=re.IGNORECASE,
)

#: Triggers that qualify a bare number as an employee identifier.
_EMPLOYEE_TRIGGER = re.compile(
    r"(?:matricule|employee (?:number|id)|staff (?:number|id)|identifiant agent)\s*:?\s*"
    r"(?P<id>(?:EMP-)?[A-Z]?\d{4,8})",
    flags=re.IGNORECASE,
)

#: (entity, pattern, score). Order is irrelevant: overlaps are resolved by
#: :func:`~itsm.privacy.entities.merge_spans`, longest span first.
_STRUCTURED: tuple[tuple[EntityType, re.Pattern[str], float], ...] = (
    (EntityType.EMAIL, re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), 1.0),
    # IBAN: two letters, two check digits, then grouped alphanumerics.
    (EntityType.IBAN, re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{3,4}){4,7}\b"), 1.0),
    # French social-security number: 1|2 YY MM DD CCC KKK CC, spaces optional.
    (
        EntityType.NATIONAL_ID,
        re.compile(r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}(?:\s?\d{2})?\b"),
        0.95,
    ),
    (EntityType.CREDIT_CARD, re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b"), 0.95),
    # French mobile/landline, and international numbers.
    (EntityType.PHONE, re.compile(r"\+33\s?\d(?:[\s.-]?\d{2}){4}\b"), 1.0),
    (EntityType.PHONE, re.compile(r"\b0\d(?:[\s.-]?\d{2}){4}\b"), 0.95),
    (EntityType.PHONE, re.compile(r"\+\d{1,3}\s?\d{2,4}\s?\d{5,8}\b"), 0.9),
    (EntityType.IP_ADDRESS, re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 1.0),
    (EntityType.EMPLOYEE_ID, re.compile(r"\bEMP-\d{4,6}\b"), 1.0),
    (EntityType.EMPLOYEE_ID, re.compile(r"\bM\d{5}\b"), 0.8),
    # Street address, French then English shape.
    (
        EntityType.ADDRESS,
        re.compile(
            rf"\b\d{{1,4}}\s+(?:{'|'.join(cat.STREET_TYPES_FR)})\s+[\w'À-ɏ\s-]{{3,40}}?"
            r"(?:,\s?\d{5}\s+[A-ZÀ-Þ][\w-]+)?(?=[.,;]|\s*$)",
            flags=re.IGNORECASE,
        ),
        0.9,
    ),
    (
        EntityType.ADDRESS,
        re.compile(
            # A lowercase particle is allowed inside the street name: real
            # addresses read "45 de Gaulle Street" as readily as "45 Mill Road".
            r"\b\d{1,4}\s+(?:(?:de|du|la|le|van|von)\s+)?[A-ZÀ-Þ][\w'-]+"
            r"(?:\s+(?:[A-ZÀ-Þ][\w'-]+|de|du|la|le))*\s+"
            r"(?:Street|Road|Avenue|Lane)(?:,\s+[A-ZÀ-Þ][\w-]+)?\b"
        ),
        0.9,
    ),
)


def _gazetteer() -> frozenset[str]:
    """Authored first-name list, lowercased.

    Written by hand rather than scraped: the point of this backend is that it
    carries no external artefact. A deployment extends it with the first names
    of the organisation's own directory, which is both legitimate and far more
    effective than any generic list.
    """
    names = (*cat.FIRST_NAMES_FR, *cat.FIRST_NAMES_EN)
    return frozenset(name.lower() for name in names)


FIRST_NAMES = _gazetteer()


#: Capitalised tokens that are never people in this corpus. Built from the
#: business catalog so it follows the estate rather than a hard-coded guess.
def _build_stoplist() -> frozenset[str]:
    """Collect product, site and vendor words that must never be redacted."""
    words: set[str] = set()
    for site in cat.SITES:
        words.add(site.lower())
    for spec in cat.CATEGORIES:
        for values in spec.slots.values():
            for value in values:
                words.update(part.lower() for part in re.split(r"[\s/]+", value) if part)
        words.update(part.lower() for part in re.split(r"[\s&/]+", spec.name) if part)
        words.add(spec.assignment_group.lower())
    words.update({"outlook", "teams", "onedrive", "windows", "office", "wi-fi", "vpn", "sap"})
    return frozenset(words)


STOPLIST = _build_stoplist()


def _is_plausible_name(chunk: str) -> bool:
    """Reject a capitalised chunk that is clearly not a person."""
    tokens = chunk.split()
    if not tokens or len(tokens) > 3:
        return False
    return not any(token.lower() in STOPLIST for token in tokens)


def _trim(chunk: str, start: int) -> tuple[str, int, int]:
    """Trim trailing punctuation from a match, returning the tight span."""
    stripped = chunk.rstrip(" .,;:!?)")
    return stripped, start, start + len(stripped)


class RuleBasedRecognizer:
    """Offline PII detector: patterns, triggers and an authored gazetteer.

    Args:
        detect_persons: Disable to use only the structured-identifier layer -
            useful when a transformer handles names and the rules handle
            formats, which is how :class:`~itsm.privacy.transformer.TransformerRecognizer`
            composes the two.
        min_score: Drop detections below this confidence.
    """

    name = "rules"

    def __init__(self, detect_persons: bool = True, min_score: float = 0.0) -> None:
        self.detect_persons = detect_persons
        self.min_score = min_score

    def _structured(self, text: str) -> list[PiiSpan]:
        spans: list[PiiSpan] = []
        for entity, pattern, score in _STRUCTURED:
            for match in pattern.finditer(text):
                value, start, end = _trim(match.group(0), match.start())
                if not value:
                    continue
                spans.append(
                    PiiSpan(
                        start=start,
                        end=end,
                        entity=entity,
                        text=value,
                        score=score,
                        source=self.name,
                    )
                )

        for pattern, entity in (
            (_BIRTH_TRIGGER, EntityType.DATE_OF_BIRTH),
            (_EMPLOYEE_TRIGGER, EntityType.EMPLOYEE_ID),
        ):
            group = "date" if entity is EntityType.DATE_OF_BIRTH else "id"
            for match in pattern.finditer(text):
                spans.append(
                    PiiSpan(
                        start=match.start(group),
                        end=match.end(group),
                        entity=entity,
                        text=match.group(group),
                        score=0.95,
                        source=self.name,
                    )
                )
        return spans

    def _persons(self, text: str) -> list[PiiSpan]:
        spans: list[PiiSpan] = []

        for match in _PERSON_TRIGGER.finditer(text):
            value, start, end = _trim(match.group("name"), match.start("name"))
            tokens = value.split()
            # A trigger followed by a single unknown capitalised word is almost
            # always the next sentence starting ("... new phone. Please ..."),
            # not a name. Demand either two tokens or a known first name.
            if len(tokens) < 2 and (not tokens or tokens[0].lower() not in FIRST_NAMES):
                continue
            if value and _is_plausible_name(value):
                spans.append(
                    PiiSpan(
                        start=start,
                        end=end,
                        entity=EntityType.PERSON,
                        text=value,
                        score=0.8,
                        source=f"{self.name}:trigger",
                    )
                )

        for match in re.finditer(_NAME_CHUNK, text):
            value, start, end = _trim(match.group(0), match.start())
            tokens = value.split()
            if len(tokens) < 2 or tokens[0].lower() not in FIRST_NAMES:
                continue
            # Keep the first name plus one surname; a third token is usually
            # the next sentence's first word. Plausibility is checked on the
            # *trimmed* name, not on the raw chunk - otherwise a trailing
            # "Workstation" would veto the perfectly good name before it.
            trimmed = " ".join(tokens[:2])
            if not _is_plausible_name(trimmed):
                continue
            spans.append(
                PiiSpan(
                    start=start,
                    end=start + len(trimmed),
                    entity=EntityType.PERSON,
                    text=trimmed,
                    score=0.9,
                    source=f"{self.name}:gazetteer",
                )
            )
        return spans

    def detect(self, text: str) -> list[PiiSpan]:
        """Return the disjoint PII spans found in ``text``."""
        if not text:
            return []
        spans = self._structured(text)
        if self.detect_persons:
            spans.extend(self._persons(text))
        return merge_spans([span for span in spans if span.score >= self.min_score])

    def detect_batch(self, texts: list[str]) -> list[list[PiiSpan]]:
        """Detect over a batch; trivially sequential for the rule engine."""
        return [self.detect(text) for text in texts]
