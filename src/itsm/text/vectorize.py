"""Vectorisation backends for ticket text.

Two interchangeable representations, selected by
:data:`~itsm.config.Settings.embedding_backend`:

``tfidf`` (default)
    Word 1-2 grams union char 3-5 grams. The char channel is what makes the
    representation robust to the typos, agglutinations and FR/EN code-switching
    of real ticket titles. Runs on CPU, offline, in seconds.

``embeddings``
    A multilingual sentence-transformer. Better at grouping paraphrases that
    share no vocabulary ("VPN drops" / "connexion coupee"), at the cost of a
    model download and ~100x the CPU time. Optional extra ``[hf]``.

Both expose the scikit-learn ``fit``/``transform`` API, so every downstream
component (classifier, clustering) is agnostic to the choice.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from itsm.text.normalize import STOPWORDS_FR_EN


def build_tfidf(
    min_df: int = 3,
    max_features: int | None = 60_000,
    use_char_ngrams: bool = True,
) -> FeatureUnion | TfidfVectorizer:
    """Build the default TF-IDF representation.

    Args:
        min_df: Minimum document frequency; drops hapax noise.
        max_features: Cap on the word vocabulary.
        use_char_ngrams: Add the character n-gram channel.

    Returns:
        A fitted-on-demand scikit-learn vectoriser.
    """
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        stop_words=sorted(STOPWORDS_FR_EN),
    )
    if not use_char_ngrams:
        return word
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Sentence-transformer encoder with a scikit-learn interface.

    The model is loaded lazily on first use and cached on the instance, so an
    unused pipeline never pays the import cost. Vectors are L2-normalised,
    which makes the Euclidean geometry used by KMeans equivalent to cosine
    similarity.

    Args:
        model_name: Hugging Face model identifier.
        batch_size: Encoding batch size.
        device: Torch device; ``None`` lets sentence-transformers decide.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # optional dependency

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def fit(self, x: Iterable[str], y: object = None) -> EmbeddingVectorizer:  # noqa: D102
        self._load()
        return self

    def transform(self, x: Iterable[str]) -> np.ndarray:  # noqa: D102
        model = self._load()
        texts = [str(v) for v in x]
        return np.asarray(
            model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        )

    def __getstate__(self) -> dict[str, Any]:
        """Drop the loaded weights when pickling; they are re-downloaded."""
        state = self.__dict__.copy()
        state["_model"] = None
        return state


def build_vectorizer(
    backend: Literal["tfidf", "embeddings"] = "tfidf",
    model_name: str | None = None,
    **tfidf_kwargs: Any,
) -> BaseEstimator:
    """Return the vectoriser matching the configured backend."""
    if backend == "embeddings":
        return EmbeddingVectorizer(model_name=model_name) if model_name else EmbeddingVectorizer()
    return build_tfidf(**tfidf_kwargs)
