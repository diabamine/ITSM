"""Finding the past tickets most similar to a new one (a simple k-NN).

Similarity blends two things, on purpose:

    similarity = w * cosine(text embeddings) + (1 - w) * share of matching categories

* the **text** (title + description) captures what the user describes, in their
  own words and in either language;
* the **categories** (service, group, incident type, affected system, site) capture
  context that the text does not always state.

Both terms live in ``[0, 1]``, so the blend is easy to read and to explain: a
score of 0.82 means "very close wording, and most of the context agrees". The
weight ``w`` is the only knob.

There is no vector database: with a few thousand to a few hundred thousand
tickets a brute-force dot product is milliseconds, and one less moving part.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from backend.schema import SIMILARITY_FEATURES, Col, fold

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class TextEmbedder:
    """Turn texts into unit-length vectors, with a sentence-transformer or TF-IDF.

    ``auto`` prefers a multilingual sentence-transformer, which understands that
    "connexion coupée" and "VPN drops" are the same problem, and falls back to
    TF-IDF + SVD (offline, no download, vocabulary-based) when the package or the
    model weights are unavailable.

    Args:
        backend: ``"auto"``, ``"sbert"`` or ``"tfidf"``.
        model_name: Sentence-transformers model for the ``sbert`` backend.
        cache_dir: Where corpus embeddings are cached between runs (sbert only,
            since encoding a few thousand texts takes a minute on CPU).
    """

    def __init__(self, backend: str = "auto", model_name: str = DEFAULT_MODEL, cache_dir: Path | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._sbert = None
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self.backend = self._resolve(backend)

    def _resolve(self, requested: str) -> str:
        if requested == "tfidf":
            return "tfidf"
        try:
            from sentence_transformers import SentenceTransformer

            self._sbert = SentenceTransformer(self.model_name)
            return "sbert"
        except Exception as error:  # missing package, or no network to fetch the weights
            if requested == "sbert":
                raise
            logger.warning("sentence-transformers unavailable (%s); using TF-IDF", error)
            return "tfidf"

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        """Embed the corpus (and, for TF-IDF, learn its vocabulary)."""
        if self.backend == "sbert":
            return self._encode_cached(list(texts))
        self._vectorizer = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        matrix = self._vectorizer.fit_transform(texts)
        components = max(2, min(128, matrix.shape[1] - 1, matrix.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=components, random_state=0)
        return normalize(self._svd.fit_transform(matrix))

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Embed new texts into the space learnt by :meth:`fit_transform`."""
        if self.backend == "sbert":
            return self._encode(list(texts))
        assert self._vectorizer is not None and self._svd is not None, "call fit_transform first"
        return normalize(self._svd.transform(self._vectorizer.transform(texts)))

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._sbert.encode(  # type: ignore[union-attr]
                texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
            )
        )

    def _encode_cached(self, texts: list[str]) -> np.ndarray:
        """Encode a corpus, reusing a previous run's result when nothing changed."""
        if self.cache_dir is None:
            return self._encode(texts)
        digest = hashlib.sha1((self.model_name + "\n".join(texts)).encode()).hexdigest()[:16]
        path = self.cache_dir / f"embeddings_{digest}.npy"
        if path.exists():
            return np.load(path)
        vectors = self._encode(texts)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors


class SimilarTicketIndex:
    """Index of past tickets, searchable with a new ticket as the query.

    Args:
        tickets: The history to search.
        embedder: Text embedder (it is fitted on this history).
        text_weight: ``w`` in the similarity formula above.
        features: Categorical columns blended into the score; columns missing
            from ``tickets`` are ignored.
    """

    def __init__(
        self,
        tickets: pd.DataFrame,
        embedder: TextEmbedder,
        text_weight: float = 0.7,
        features: Sequence[str] = SIMILARITY_FEATURES,
    ) -> None:
        self.tickets = tickets.reset_index(drop=True)
        self.embedder = embedder
        self.text_weight = text_weight
        self.features = [f for f in features if f in self.tickets.columns]
        self._vectors = embedder.fit_transform(self._texts(self.tickets))
        self._categories = np.array(
            [[fold(v) if pd.notna(v) else "" for v in self.tickets[f]] for f in self.features],
            dtype=object,
        ).T

    @staticmethod
    def _texts(frame: pd.DataFrame) -> list[str]:
        title = frame[Col.TITLE].fillna("").astype(str)
        body = frame[Col.DESCRIPTION].fillna("").astype(str)
        return (title + ". " + body).tolist()

    def search(self, query: Mapping[str, object], k: int = 5) -> pd.DataFrame:
        """Return the ``k`` most similar past tickets, best first.

        Args:
            query: A ticket as a mapping. ``Description courte`` and ``Description``
                feed the text term; any column in :attr:`features` feeds the
                category term (empty values simply do not match).
            k: How many tickets to return.

        Returns:
            The matching rows plus three columns: ``Similarity`` (the blend),
            ``Text match`` and ``Category match`` (its two terms).
        """
        text = f"{query.get(Col.TITLE, '')}. {query.get(Col.DESCRIPTION, '')}"
        text_match = np.clip(self._vectors @ self.embedder.transform([text])[0], 0.0, 1.0)

        wanted = np.array([fold(query.get(f, "") or "") for f in self.features], dtype=object)
        matches = (self._categories == wanted) & (wanted != "")
        category_match = matches.mean(axis=1) if self.features else np.zeros(len(self.tickets))

        score = self.text_weight * text_match + (1 - self.text_weight) * category_match
        # Ties are common (identical wording, identical context): prefer the most recent.
        opened = self.tickets[Col.OPENED].astype("int64").to_numpy()
        top = np.lexsort((-opened, -score))[:k]
        result = self.tickets.iloc[top].copy()
        result["Similarity"] = score[top]
        result["Text match"] = text_match[top]
        result["Category match"] = category_match[top]
        return result
