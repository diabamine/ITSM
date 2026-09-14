"""Detection and redaction of personal data in ticket text.

Ticket free text is full of names, mailboxes and phone numbers that nobody
decided to collect. This package removes them before any analysis runs, offers
three independently benchmarkable detectors - an offline rule engine, a Hugging
Face NER model and Microsoft Presidio's analyser - replaces each identifier
with its entity tag, and, crucially, **measures** the result: see
:mod:`itsm.privacy.evaluate` and ``docs/07-privacy.md``.
"""

from itsm.privacy.anonymiser import Anonymiser, Backend, Recognizer, build_recognizer
from itsm.privacy.entities import DIRECT_IDENTIFIERS, EntityType, PiiSpan, merge_spans
from itsm.privacy.evaluate import match_spans, residual_pii_rate, score_detection
from itsm.privacy.redact import DEFAULT_POLICY, Policy, Redactor, tag_for
from itsm.privacy.rules import RuleBasedRecognizer

__all__ = [
    "DEFAULT_POLICY",
    "DIRECT_IDENTIFIERS",
    "Anonymiser",
    "Backend",
    "EntityType",
    "PiiSpan",
    "Policy",
    "Recognizer",
    "Redactor",
    "RuleBasedRecognizer",
    "build_recognizer",
    "match_spans",
    "merge_spans",
    "residual_pii_rate",
    "score_detection",
    "tag_for",
]
