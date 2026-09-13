"""Normalisation of ticket designations.

Ticket titles are short, noisy, bilingual and full of identifiers. Left raw,
a TF-IDF vocabulary explodes with hostnames, ticket references and error codes
that are unique to a single ticket and therefore carry no generalisable
signal. The normaliser trades that noise for a handful of placeholder tokens
(``<url>``, ``<code>``, ``<num>``...), which keeps *the fact that* a code was
present - often informative - while dropping its unique value.

The whole module is dependency-free (no spaCy, no NLTK download) so that the
analysis runs offline and deterministically.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from sklearn.base import BaseEstimator, TransformerMixin

#: Boilerplate that service-desk forms and mail gateways prepend or append.
_BOILERPLATE = re.compile(
    r"\b(bonjour|madame|monsieur|cordialement|merci d avance|best regards|regards|hello|"
    r"hi team|envoye de mon iphone|sent from my iphone|ticket cree automatiquement|"
    r"automatically created)\b",
    flags=re.IGNORECASE,
)

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"https?://\S+|www\.\S+"), " <url> "),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), " <email> "),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), " <ip> "),
    (re.compile(r"\b(?:inc|req|chg|prb|tsk)\s?\d{4,}\b", re.IGNORECASE), " <ticketref> "),
    # Error codes: hex codes, ORA-xxxxx, HTTP 5xx, SAP transactions.
    (re.compile(r"\b0x[0-9a-f]{4,}\b", re.IGNORECASE), " <code> "),
    (re.compile(r"\b[a-z]{2,5}-\d{3,6}\b", re.IGNORECASE), " <code> "),
    (re.compile(r"\bhttp\s?[45]\d{2}\b", re.IGNORECASE), " <code> "),
    (re.compile(r"\berr[_a-z]*_[a-z_]+\b", re.IGNORECASE), " <code> "),
    # Hostnames / asset tags such as PC-FR-01234 or SRVPARIS12.
    (re.compile(r"\b[a-z]{2,}[-_]?[a-z]{0,3}\d{2,}\b", re.IGNORECASE), " <asset> "),
    (re.compile(r"\b\d+([.,]\d+)?\b"), " <num> "),
    (re.compile(r"[^\w<>\s]+"), " "),
    (re.compile(r"\s+"), " "),
)

#: French and English stop words, plus service-desk filler that dominates
#: every family and therefore separates nothing ("probleme", "issue"...).
STOPWORDS_FR_EN: frozenset[str] = frozenset(
    [
        "a",
        "ai",
        "alors",
        "au",
        "aucun",
        "aussi",
        "autre",
        "avec",
        "avoir",
        "avons",
        "c",
        "ca",
        "car",
        "ce",
        "cela",
        "ces",
        "cet",
        "cette",
        "ceux",
        "chaque",
        "comme",
        "d",
        "dans",
        "de",
        "des",
        "du",
        "elle",
        "elles",
        "en",
        "encore",
        "est",
        "et",
        "etre",
        "eu",
        "fait",
        "faire",
        "il",
        "ils",
        "j",
        "je",
        "l",
        "la",
        "le",
        "les",
        "leur",
        "lui",
        "ma",
        "mais",
        "me",
        "meme",
        "mes",
        "moi",
        "mon",
        "n",
        "ne",
        "nos",
        "notre",
        "nous",
        "on",
        "ou",
        "par",
        "pas",
        "plus",
        "pour",
        "qu",
        "que",
        "quel",
        "quelle",
        "qui",
        "sa",
        "sans",
        "se",
        "ses",
        "si",
        "son",
        "sont",
        "sur",
        "ta",
        "te",
        "tes",
        "toi",
        "ton",
        "tous",
        "tout",
        "tu",
        "un",
        "une",
        "vos",
        "votre",
        "vous",
        "y",
        "etait",
        "ete",
        "depuis",
        "toujours",
        "tres",
        "peut",
        "plusieurs",
        "apres",
        "about",
        "after",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "out",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "too",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "bonjour",
        "merci",
        "svp",
        "stp",
        "please",
        "thanks",
        "thank",
        "hello",
        "regards",
        "cordialement",
        "probleme",
        "problem",
        "issue",
        "incident",
        "ticket",
        "demande",
        "request",
        "urgent",
        "svp",
        "aide",
        "help",
        "support",
    ]
)


def strip_accents(text: str) -> str:
    """Remove diacritics so that ``probleme`` and ``problème`` match."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(text: str | None) -> str:
    """Normalise one designation.

    Lowercases, strips accents and boilerplate, and replaces identifiers by
    placeholder tokens.

    Args:
        text: Raw designation; ``None`` is treated as empty.

    Returns:
        The normalised string, possibly empty.

    Examples:
        >>> normalize_text("VPN client returns error 0x80070005 on PC12345")
        'vpn client returns error <code> on <asset>'
    """
    if not text:
        return ""
    value = strip_accents(str(text)).lower()
    value = _BOILERPLATE.sub(" ", value)
    for pattern, replacement in _PATTERNS:
        value = pattern.sub(replacement, value)
    return value.strip()


def tokenize(text: str, drop_stopwords: bool = True, min_length: int = 2) -> list[str]:
    """Split a normalised string into content tokens."""
    tokens = text.split()
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_FR_EN]
    return [t for t in tokens if len(t) >= min_length or t.startswith("<")]


class TextNormalizer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer wrapper around :func:`normalize_text`.

    Placed first in every pipeline so that the exact same preprocessing is
    persisted with the model and applied at inference time.
    """

    def fit(self, x: Iterable[str], y: object = None) -> TextNormalizer:  # noqa: D102
        return self

    def transform(self, x: Iterable[str]) -> list[str]:  # noqa: D102
        return [normalize_text(value) for value in x]
