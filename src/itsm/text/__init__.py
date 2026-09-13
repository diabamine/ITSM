"""Text preparation: normalisation and vectorisation of ticket designations."""

from itsm.text.normalize import STOPWORDS_FR_EN, TextNormalizer, normalize_text
from itsm.text.vectorize import EmbeddingVectorizer, build_vectorizer

__all__ = [
    "STOPWORDS_FR_EN",
    "EmbeddingVectorizer",
    "TextNormalizer",
    "build_vectorizer",
    "normalize_text",
]
