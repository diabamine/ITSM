"""Naming clusters: class-based TF-IDF and representative tickets.

A cluster id is useless to a service-desk manager. Two complementary devices
turn it into something readable:

* **c-TF-IDF** - documents of a cluster are concatenated into one pseudo
  document, then weighted against the other clusters. Terms that are frequent
  *inside* the cluster and rare *elsewhere* rise to the top, which is exactly
  the discriminative vocabulary of the family.
* **Representative tickets** - the designations closest to the cluster
  centroid, which give the reader a concrete anchor.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from itsm.text.normalize import STOPWORDS_FR_EN


def class_tfidf(
    documents: list[str],
    labels: np.ndarray,
    top_n: int = 8,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
) -> dict[int, list[tuple[str, float]]]:
    """Rank the most discriminative terms of each cluster.

    Implements the c-TF-IDF weighting popularised by BERTopic:
    ``w(t, c) = tf(t, c) * log(1 + A / f(t))`` where ``tf`` is the frequency of
    term ``t`` in the concatenated documents of cluster ``c``, ``A`` the average
    number of terms per cluster and ``f(t)`` the total frequency of ``t``.

    Args:
        documents: Normalised texts, one per ticket.
        labels: Cluster assignment of each ticket. ``-1`` (noise) is skipped.
        top_n: Number of terms to keep per cluster.
        ngram_range: N-gram range of the term vocabulary.
        min_df: Minimum document frequency **in the joined space**, where one
            document is one whole cluster. It must stay at 1: raising it would
            drop exactly the cluster-specific terms this function exists to
            surface. Rare terms are already suppressed by their low ``tf``.

    Returns:
        Mapping ``cluster id -> [(term, weight), ...]`` sorted by weight.
    """
    unique = sorted({int(label) for label in labels if int(label) != -1})
    if not unique:
        return {}

    joined = [" ".join(np.asarray(documents)[labels == label]) for label in unique]
    vectorizer = CountVectorizer(
        ngram_range=ngram_range, min_df=min_df, stop_words=sorted(STOPWORDS_FR_EN)
    )
    counts = vectorizer.fit_transform(joined).toarray().astype(float)
    vocabulary = np.asarray(vectorizer.get_feature_names_out())

    term_frequency = counts.sum(axis=0)
    average_terms = counts.sum() / max(len(unique), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        idf = np.log1p(average_terms / np.where(term_frequency == 0, np.nan, term_frequency))
    tf = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
    weights = np.nan_to_num(tf * idf)

    ranked: dict[int, list[tuple[str, float]]] = {}
    for row, label in enumerate(unique):
        order = np.argsort(weights[row])[::-1][:top_n]
        ranked[label] = [(str(vocabulary[i]), float(weights[row, i])) for i in order]
    return ranked


def cluster_name(terms: list[tuple[str, float]], max_terms: int = 3) -> str:
    """Build a short human label from the top c-TF-IDF terms."""
    return " / ".join(term for term, _ in terms[:max_terms]) or "unlabelled"


def representative_indices(
    features: np.ndarray, labels: np.ndarray, centroids: np.ndarray, n: int = 3
) -> dict[int, list[int]]:
    """Return, per cluster, the row indices closest to the centroid.

    Args:
        features: Dense feature matrix used for clustering.
        labels: Cluster assignment per row.
        centroids: Cluster centroids, indexed by sorted cluster id.
        n: Number of representatives per cluster.

    Returns:
        Mapping ``cluster id -> [row index, ...]``.
    """
    output: dict[int, list[int]] = {}
    for position, label in enumerate(sorted({int(v) for v in labels if int(v) != -1})):
        members = np.flatnonzero(labels == label)
        if members.size == 0:
            continue
        distances = np.linalg.norm(features[members] - centroids[position], axis=1)
        output[label] = [int(members[i]) for i in np.argsort(distances)[:n]]
    return output
