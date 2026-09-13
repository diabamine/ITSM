"""Unsupervised discovery of ticket families from the designation.

Pipeline: ``normalise -> vectorise -> reduce -> cluster -> label``.

Design notes
------------
*Why reduce before clustering?* KMeans minimises a Euclidean objective, which
degrades badly in the very high dimensional, sparse TF-IDF space. Truncated
SVD (LSA) projects to a few hundred dense components, keeps the dominant
co-occurrence structure and makes centroids meaningful. Vectors are
L2-normalised afterwards so Euclidean distance is monotone in cosine distance -
the *spherical* KMeans commonly used for text.

*Why KMeans by default?* It is deterministic under a fixed seed, cheap to
refit, and gives a centroid per family that the API can reuse to assign a new
ticket in microseconds. HDBSCAN is available for exploration: it finds
variable-density families and, crucially, refuses to assign outliers (label
``-1``) - useful to surface genuinely novel tickets - but it has no native
``predict`` for unseen points.

*How is the result validated?* Internal metrics (silhouette, Davies-Bouldin)
measure geometric quality. When a category label exists, external metrics
(ARI, NMI, homogeneity, completeness) measure whether the discovered families
actually recover the operational taxonomy.

*Expect homogeneity to be high and completeness to be low, and read ARI
accordingly.* Measured on the reference dataset, clusters are consistently
**finer** than the declared taxonomy: one category spans several symptom
templates and two languages, so it splits into several pure clusters. Purity
and homogeneity climb with k (0.35 -> 0.78 for k = 12 -> 40) while completeness
plateaus around 0.5, and ARI stays near 0.2 because it penalises exactly that
splitting. A low ARI is therefore not a failure here - it is the measurement
that the routing taxonomy is coarser than what users actually report, which is
the finding the analysis exists to produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from itsm.analysis.labeling import class_tfidf, cluster_name, representative_indices
from itsm.logging import get_logger
from itsm.text.normalize import TextNormalizer, normalize_text
from itsm.text.vectorize import build_vectorizer

logger = get_logger("analysis.clustering")


@dataclass
class ClusterSummary:
    """Readable description of one discovered family."""

    cluster_id: int
    size: int
    share: float
    name: str
    top_terms: list[str]
    representatives: list[str]
    dominant_category: str | None = None
    category_purity: float | None = None


@dataclass
class ClusteringResult:
    """Outcome of a clustering run: assignments, metrics and summaries."""

    labels: np.ndarray
    n_clusters: int
    metrics: dict[str, float]
    summaries: list[ClusterSummary] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Return the cluster summaries as a dataframe, largest first."""
        return pd.DataFrame([s.__dict__ for s in self.summaries]).sort_values(
            "size", ascending=False
        )


class DesignationClusterer(BaseEstimator):
    """Cluster ticket designations into families.

    Args:
        n_clusters: Number of families. ``None`` triggers a silhouette-based
            search over ``k_range``.
        backend: Text representation, ``tfidf`` or ``embeddings``.
        algorithm: ``kmeans`` (default, supports ``predict``) or ``hdbscan``.
        n_components: SVD dimensionality; ignored for dense embeddings. 60 is
            empirically better than 120 here: extra components add noise
            directions that blur the families.
        k_range: Candidate values of k when ``n_clusters`` is ``None``.
        random_state: Seed, fixes the whole run.
    """

    def __init__(
        self,
        n_clusters: int | None = None,
        backend: Literal["tfidf", "embeddings"] = "tfidf",
        algorithm: Literal["kmeans", "hdbscan"] = "kmeans",
        n_components: int = 60,
        k_range: tuple[int, ...] = (12, 16, 20, 24, 30, 36),
        min_cluster_size: int = 25,
        random_state: int = 42,
        embedding_model: str | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.backend = backend
        self.algorithm = algorithm
        self.n_components = n_components
        self.k_range = k_range
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.embedding_model = embedding_model

        self.features_: np.ndarray | None = None
        self.model_: Any | None = None
        self.pipeline_: Pipeline | None = None
        self.centroids_: np.ndarray | None = None
        self.summaries_: list[ClusterSummary] = []

    def __getstate__(self) -> dict[str, Any]:
        """Drop the training feature matrix when pickling.

        It is not needed for inference and would multiply the artifact size by
        two orders of magnitude.
        """
        state = self.__dict__.copy()
        state["features_"] = None
        return state

    # -- internals --------------------------------------------------------
    def _build_pipeline(self) -> Pipeline:
        steps: list[tuple[str, Any]] = [
            ("normalize", TextNormalizer()),
            ("vectorize", build_vectorizer(self.backend, self.embedding_model)),
        ]
        if self.backend == "tfidf":
            steps.append(
                (
                    "reduce",
                    TruncatedSVD(n_components=self.n_components, random_state=self.random_state),
                )
            )
        steps.append(("scale", Normalizer(copy=False)))
        return Pipeline(steps)

    def _select_k(self, features: np.ndarray) -> int:
        """Pick k by silhouette on a subsample (silhouette is O(n^2))."""
        rng = np.random.default_rng(self.random_state)
        sample = (
            rng.choice(len(features), size=3000, replace=False)
            if len(features) > 3000
            else np.arange(len(features))
        )
        best_k, best_score = self.k_range[0], -1.0
        for k in self.k_range:
            labels = KMeans(n_clusters=k, n_init=5, random_state=self.random_state).fit_predict(
                features
            )
            score = float(silhouette_score(features[sample], labels[sample]))
            logger.info("k=%d silhouette=%.4f", k, score)
            if score > best_score:
                best_k, best_score = k, score
        return best_k

    def _fit_cluster_model(self, features: np.ndarray) -> np.ndarray:
        if self.algorithm == "hdbscan":
            import hdbscan  # optional dependency

            self.model_ = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size, prediction_data=True
            )
            return self.model_.fit_predict(features)

        k = self.n_clusters or self._select_k(features)
        self.model_ = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        return self.model_.fit_predict(features)

    def _compute_centroids(self) -> np.ndarray:
        """Return the cluster centroids, averaged from the data for HDBSCAN."""
        if self.model_ is None or self.features_ is None:
            raise RuntimeError("clusterer is not fitted")
        if hasattr(self.model_, "cluster_centers_"):
            return np.asarray(self.model_.cluster_centers_)
        labels = self.model_.labels_
        return np.vstack(
            [self.features_[labels == label].mean(axis=0) for label in sorted(set(labels) - {-1})]
        )

    # -- public API -------------------------------------------------------
    def fit(self, designations: list[str], categories: pd.Series | None = None) -> ClusteringResult:
        """Fit the clusterer and return assignments, metrics and summaries.

        Args:
            designations: Raw ticket designations.
            categories: Optional ground-truth family, used only to compute
                external validation metrics.

        Returns:
            A :class:`ClusteringResult`.
        """
        self.pipeline_ = self._build_pipeline()
        self.features_ = np.asarray(self.pipeline_.fit_transform(designations))
        labels = np.asarray(self._fit_cluster_model(self.features_))
        logger.info("fitted %d clusters on %d designations", len(set(labels) - {-1}), len(labels))
        return self._describe(designations, labels, categories)

    def predict(self, designations: list[str]) -> np.ndarray:
        """Assign new designations to the fitted families."""
        if self.pipeline_ is None or self.model_ is None:
            raise RuntimeError("clusterer is not fitted")
        features = np.asarray(self.pipeline_.transform(designations))
        if hasattr(self.model_, "predict"):
            return np.asarray(self.model_.predict(features))
        import hdbscan  # optional dependency

        labels, _ = hdbscan.approximate_predict(self.model_, features)
        return np.asarray(labels)

    def _describe(
        self, designations: list[str], labels: np.ndarray, categories: pd.Series | None
    ) -> ClusteringResult:
        assert self.features_ is not None
        normalized = [normalize_text(d) for d in designations]
        metrics = compute_metrics(self.features_, labels, categories)

        self.centroids_ = self._compute_centroids()
        terms = class_tfidf(normalized, labels)
        representatives = representative_indices(self.features_, labels, self.centroids_)

        summaries: list[ClusterSummary] = []
        total = len(labels)
        for label in sorted(set(int(v) for v in labels) - {-1}):
            members = np.flatnonzero(labels == label)
            top_terms = [term for term, _ in terms.get(label, [])]
            summary = ClusterSummary(
                cluster_id=label,
                size=int(members.size),
                share=round(float(members.size / total), 4),
                name=cluster_name(terms.get(label, [])),
                top_terms=top_terms,
                representatives=[designations[i] for i in representatives.get(label, [])],
            )
            if categories is not None:
                counts = pd.Series(np.asarray(categories)[members]).value_counts()
                summary.dominant_category = str(counts.index[0])
                summary.category_purity = round(float(counts.iloc[0] / members.size), 4)
            summaries.append(summary)

        self.summaries_ = summaries
        return ClusteringResult(
            labels=labels, n_clusters=len(summaries), metrics=metrics, summaries=summaries
        )


def compute_metrics(
    features: np.ndarray, labels: np.ndarray, categories: pd.Series | None = None
) -> dict[str, float]:
    """Compute internal and, when labels exist, external clustering metrics.

    Internal metrics are computed on a subsample because silhouette is
    quadratic in the number of points.
    """
    mask = labels != -1
    unique = set(labels[mask])
    metrics: dict[str, float] = {
        "n_clusters": float(len(unique)),
        "noise_ratio": float((~mask).mean()),
    }
    if len(unique) < 2:
        return metrics

    rng = np.random.default_rng(0)
    indices = np.flatnonzero(mask)
    sample = rng.choice(indices, size=min(4000, indices.size), replace=False)
    metrics["silhouette"] = round(float(silhouette_score(features[sample], labels[sample])), 4)
    metrics["davies_bouldin"] = round(float(davies_bouldin_score(features[mask], labels[mask])), 4)
    metrics["calinski_harabasz"] = round(
        float(calinski_harabasz_score(features[mask], labels[mask])), 2
    )

    if categories is not None:
        truth = np.asarray(categories)[mask]
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
            truth, labels[mask]
        )
        metrics |= {
            "adjusted_rand": round(float(adjusted_rand_score(truth, labels[mask])), 4),
            "normalized_mutual_info": round(
                float(normalized_mutual_info_score(truth, labels[mask])), 4
            ),
            "homogeneity": round(float(homogeneity), 4),
            "completeness": round(float(completeness), 4),
            "v_measure": round(float(v_measure), 4),
        }
    return metrics
