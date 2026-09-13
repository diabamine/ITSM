"""2D exploratory map: sentence embeddings -> UMAP -> HDBSCAN.

Companion to :mod:`itsm.analysis.clustering`, which fits a production
``DesignationClusterer`` (TF-IDF/SVD/KMeans, persisted so the API can assign a
new ticket in microseconds). This module answers a different question - *what
does the ticket corpus look like* - and is built for a human looking at a
scatter plot rather than for serving:

* text is the **designation and the free-text description concatenated**, not
  the designation alone, so the map also reflects what the requester wrote in
  the body;
* a sentence-transformer (or, offline, TF-IDF + SVD) embeds each ticket;
* **UMAP reduces straight to 2 components**, and **HDBSCAN clusters in that
  same 2D space** - the plot the analyst looks at is exactly the space the
  clusters were found in, not a separate high-dimensional model whose
  boundaries the picture only suggests;
* HDBSCAN's own noise label (``-1``) surfaces tickets that do not belong to any
  dense family, which a forced partition (KMeans) cannot express.

Nothing here is persisted: the parameters worth exploring (sample size,
``n_neighbors``, ``min_dist``, ``min_cluster_size``) are meant to be turned
live, and refitting on a few thousand points takes a couple of seconds once the
embeddings themselves are cached (the expensive step - see
``itsm.app.state.embedding_features``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from itsm.analysis.clustering import compute_metrics
from itsm.analysis.labeling import class_tfidf, cluster_name
from itsm.logging import get_logger
from itsm.text.normalize import normalize_text
from itsm.text.vectorize import EmbeddingVectorizer, build_tfidf

logger = get_logger("analysis.embedding_map")

TextBackend = Literal["tfidf", "embeddings"]


@dataclass
class EmbeddingMap:
    """A 2D layout of tickets with the density clusters found in it."""

    coords: np.ndarray  # (n, 2), the UMAP layout
    labels: np.ndarray  # (n,), HDBSCAN cluster id, -1 = noise
    probabilities: np.ndarray  # (n,), HDBSCAN membership strength
    metrics: dict[str, float]
    cluster_names: dict[int, str] = field(default_factory=dict)
    cluster_terms: dict[int, list[tuple[str, float]]] = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        """Number of non-noise clusters found."""
        return len(self.cluster_names)

    def representative_indices(self, n: int = 5) -> dict[int, list[int]]:
        """Return, per cluster, the ``n`` rows HDBSCAN is most confident about."""
        output: dict[int, list[int]] = {}
        for label in self.cluster_names:
            members = np.flatnonzero(self.labels == label)
            order = members[np.argsort(self.probabilities[members])[::-1]]
            output[label] = [int(i) for i in order[:n]]
        return output


def combined_text(
    frame: pd.DataFrame, columns: tuple[str, ...] = ("designation", "description")
) -> list[str]:
    """Concatenate the designation and the free-text body into one string.

    Missing columns and null values are treated as empty, so this also works on
    sources that never populate ``description``.
    """
    parts = [
        frame[column].fillna("").astype(str)
        if column in frame.columns
        else pd.Series([""] * len(frame))
        for column in columns
    ]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined.str.cat(part, sep=" ")
    return joined.str.strip().tolist()


def embed(
    texts: list[str], backend: TextBackend = "tfidf", embedding_model: str | None = None
) -> np.ndarray:
    """Encode ``texts`` into a dense vector per ticket.

    ``embeddings`` uses a multilingual sentence-transformer (offline-unfriendly,
    needs the ``[hf]`` extra); ``tfidf`` falls back to word+char TF-IDF reduced
    with SVD, which runs anywhere and is what the demo uses by default.
    """
    if backend == "embeddings":
        vectorizer = (
            EmbeddingVectorizer(model_name=embedding_model)
            if embedding_model
            else (EmbeddingVectorizer())
        )
        return np.asarray(vectorizer.fit_transform(texts))

    pipeline = Pipeline(
        [
            ("vectorize", build_tfidf()),
            ("reduce", TruncatedSVD(n_components=100, random_state=42)),
            ("scale", Normalizer(copy=False)),
        ]
    )
    return np.asarray(pipeline.fit_transform(texts))


def project(
    features: np.ndarray,
    texts: list[str],
    categories: pd.Series | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 25,
    min_samples: int | None = None,
    random_state: int = 42,
) -> EmbeddingMap:
    """Reduce ``features`` to 2D with UMAP, then cluster that layout with HDBSCAN.

    Args:
        features: Dense per-ticket vectors, as returned by :func:`embed`.
        texts: The same tickets' raw text, used only to name clusters.
        categories: Optional declared category, to score agreement with the
            discovered clusters.
        n_neighbors: UMAP's local-neighbourhood size; higher favours global
            structure over fine detail.
        min_dist: UMAP's minimum distance between points in the embedding;
            lower packs same-cluster points tighter.
        min_cluster_size: HDBSCAN's smallest group size worth calling a cluster.
        min_samples: HDBSCAN conservatism; ``None`` defaults to
            ``min_cluster_size``, HDBSCAN's own default.
        random_state: Seed, fixes the UMAP layout.

    Returns:
        An :class:`EmbeddingMap`.

    Raises:
        RuntimeError: if ``umap-learn`` / ``hdbscan`` are not installed.
    """
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - both are core dependencies
        raise RuntimeError("the 2D map needs `umap-learn`: `uv sync`.") from exc
    try:
        import hdbscan
    except ImportError as exc:  # pragma: no cover - both are core dependencies
        raise RuntimeError("the 2D map needs `hdbscan`: `uv sync`.") from exc

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=random_state,
    )
    coords = np.asarray(reducer.fit_transform(features))

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = np.asarray(clusterer.fit_predict(coords))
    probabilities = np.asarray(clusterer.probabilities_)
    logger.info(
        "projected %d tickets to 2D: %d clusters, %.1f%% noise",
        len(labels),
        len(set(labels) - {-1}),
        100 * float((labels == -1).mean()),
    )

    metrics = compute_metrics(coords, labels, categories)

    normalized = [normalize_text(t) for t in texts]
    terms = class_tfidf(normalized, labels)
    names = {label: cluster_name(ranked) for label, ranked in terms.items()}

    return EmbeddingMap(
        coords=coords,
        labels=labels,
        probabilities=probabilities,
        metrics=metrics,
        cluster_names=names,
        cluster_terms=terms,
    )
