"""Sub-topic discovery and growth attribution: what is driving a ticket increase.

The weak-signal detector (:mod:`itsm.models.weak_signal`) answers *when* and *in
which category* volume moved. That is rarely the question a service-desk owner
actually has. "Network tickets are up 40%" is not actionable; "78% of that
increase is one symptom - the VPN client crashing after the Windows update" is.

Declared categories are too coarse to answer it, because they are a routing
taxonomy: a handful of buckets chosen years ago so tickets reach the right
queue. A root cause lives one level below - a *sub-topic* - and nobody has
labelled those, because they are only known after they appear.

This module therefore does three things:

1. **Discovers sub-topics inside each category** from the text, using
   embeddings so that wordings that share no vocabulary ("VPN drops",
   "connexion coupee") land together.
2. **Names them**, offline by default and with an LLM when one is available.
3. **Attributes the increase** to them - the part that turns a cluster into a
   root cause.

The attribution is the point, and it is deliberately not anomaly detection.
Running a detector on each sub-topic series fails: the counts are small, the
series are noisy, and a detector answers yes/no where the question is "how
much of the rise is this?". So the increase is **decomposed** instead: the
category delta is the sum of its sub-topic deltas, and each sub-topic gets its
share.

Two ranking rules matter, and they disagree on purpose:

* **Absolute contribution** (``delta_per_day``) ranks what to fix. A sub-topic
  going from 20 to 30 tickets a day contributes ten times more than one going
  from 1 to 3, even though the second grew three-fold.
* **Growth ratio** ranks what is *new*. A sub-topic absent from the baseline
  has no ratio at all, and those are reported separately as ``emerging`` -
  they are the genuinely novel failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from itsm.analysis.embedding_map import combined_text
from itsm.analysis.labeling import class_tfidf
from itsm.logging import get_logger
from itsm.text.normalize import TextNormalizer, normalize_text
from itsm.text.vectorize import EmbeddingVectorizer, build_tfidf

logger = get_logger("analysis.root_cause")

#: Cosine similarity above which a sub-topic from a new run is considered "the
#: same" as one from the previous run and inherits its id. Below it, the topic
#: is new. Tags that change meaning between runs make trend tracking useless,
#: which is why this is explicit rather than incidental.
STABLE_ID_THRESHOLD = 0.80

#: Centroid cosine above which two clusters are the same sub-topic. Calibrated
#: on the reference corpus: cross-language variants of one symptom score 0.73
#: and 0.85, while unrelated sub-topics of a category reach 0.61 at the 95th
#: percentile. Re-measure it on a new corpus rather than trusting it blindly.
MERGE_THRESHOLD = 0.70

#: A ticket further than this from every centroid is left unassigned rather
#: than forced into the nearest topic. Refusing to tag is a feature: it is how
#: a genuinely new failure mode becomes visible instead of being absorbed.
ASSIGN_THRESHOLD = 0.45


@dataclass
class SubTopic:
    """One discovered sub-category inside a parent category."""

    topic_id: str
    """Stable identifier, reused across runs when the topic persists."""
    parent_category: str
    label: str
    keywords: list[str]
    size: int
    examples: list[str] = field(default_factory=list)
    centroid: np.ndarray | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view; the centroid is dropped."""
        return {
            "topic_id": self.topic_id,
            "parent_category": self.parent_category,
            "label": self.label,
            "keywords": self.keywords,
            "size": self.size,
            "examples": self.examples,
        }


@dataclass
class SubTopicModel:
    """The discovered sub-topic taxonomy, fitted on a ticket corpus."""

    topics: list[SubTopic]
    assignments: np.ndarray
    """Topic id per input ticket; empty string means unassigned."""
    backend: str
    coverage: float
    """Share of tickets that received a topic - the rest are too diffuse."""
    encoder: Any = None
    """The *fitted* text encoder, kept so new tickets can be projected into the
    same space. Re-fitting one on the incoming batch would place it in a
    different basis and make every similarity meaningless."""

    def by_id(self) -> dict[str, SubTopic]:
        """Index the topics by their stable id."""
        return {topic.topic_id: topic for topic in self.topics}

    def assign(self, texts: list[str]) -> list[str]:
        """Tag new tickets with the existing topics.

        A ticket whose nearest centroid is further than
        :data:`ASSIGN_THRESHOLD` stays unassigned - which is how a genuinely
        new failure mode stays visible instead of being absorbed into the
        closest existing topic.
        """
        centroids = [t.centroid for t in self.topics if t.centroid is not None]
        if not centroids or self.encoder is None:
            return [""] * len(texts)

        matrix = np.vstack(centroids)
        ids = [t.topic_id for t in self.topics if t.centroid is not None]
        features = _normalise(np.asarray(self.encoder.transform(texts)))

        similarity = features @ matrix.T
        best = similarity.argmax(axis=1)
        return [
            ids[index] if similarity[row, index] >= ASSIGN_THRESHOLD else ""
            for row, index in enumerate(best)
        ]


def build_encoder(
    backend: Literal["tfidf", "embeddings"], embedding_model: str | None, n_documents: int
) -> Pipeline:
    """Build the text encoder, kept fitted so new tickets land in one space.

    ``embeddings`` is stateless - a sentence-transformer maps any text into a
    fixed space - but TF-IDF has to *learn* its vocabulary and its SVD basis,
    so the fitted object is what matters, not the function that produced it.
    """
    if backend == "embeddings":
        encoder = (
            EmbeddingVectorizer(model_name=embedding_model)
            if embedding_model
            else EmbeddingVectorizer()
        )
        return Pipeline([("embed", encoder)])

    return Pipeline(
        [
            ("normalize", TextNormalizer()),
            ("vectorize", build_tfidf()),
            (
                "reduce",
                TruncatedSVD(n_components=min(100, max(2, n_documents - 1)), random_state=42),
            ),
            ("scale", Normalizer(copy=False)),
        ]
    )


def _normalise(features: np.ndarray) -> np.ndarray:
    """L2-normalise rows so a dot product is a cosine similarity."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-9, None)


def _cluster(features: np.ndarray, min_cluster_size: int, random_state: int) -> np.ndarray:
    """Split one category into sub-topics.

    HDBSCAN when available - it finds variable-density groups and refuses to
    assign outliers, which is exactly right here. KMeans otherwise, with k
    derived from the corpus size so the granularity stays comparable.
    """
    if len(features) < 2 * min_cluster_size:
        return np.zeros(len(features), dtype=int)

    try:
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(
            features
        )
        if (labels >= 0).sum() >= min_cluster_size:
            return np.asarray(labels)
        logger.info("hdbscan found no dense sub-topic; falling back to kmeans")
    except ImportError:  # pragma: no cover - optional dependency
        logger.info("hdbscan not installed; using kmeans")

    n_clusters = max(2, min(12, len(features) // (2 * min_cluster_size)))
    return KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state).fit_predict(features)


def _merge_similar(features: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Merge clusters whose centroids are near-identical, and remap the labels.

    Clustering splits the *same symptom written in two languages* into two
    groups - measured on the reference corpus, the French and English variants
    of one Outlook failure sit at cosine 0.85, and a mailbox-quota pair at
    0.73, while unrelated sub-topics of the same category reach only 0.61 at
    the 95th percentile. A single agglomerative pass at :data:`MERGE_THRESHOLD`
    closes that gap.

    It matters for attribution, not for tidiness: an unmerged root cause is
    reported as two rows of 33% and 24% instead of one of 56%, which is how a
    real driver gets mistaken for two minor ones.
    """
    unique = sorted(set(int(v) for v in labels) - {-1})
    if len(unique) < 2:
        return labels

    centroids = _normalise(np.vstack([features[labels == label].mean(axis=0) for label in unique]))
    parent = {label: label for label in unique}

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    similarity = centroids @ centroids.T
    for i, left in enumerate(unique):
        for j, right in enumerate(unique[i + 1 :], start=i + 1):
            if similarity[i, j] >= threshold:
                parent[find(right)] = find(left)

    remapped = labels.copy()
    for label in unique:
        remapped[labels == label] = find(label)
    if len(set(remapped) - {-1}) < len(unique):
        logger.info("merged %d sub-topics into %d", len(unique), len(set(remapped) - {-1}))
    return remapped


def _offline_label(keywords: list[str], examples: list[str]) -> str:
    """Name a sub-topic without any model.

    The shortest representative designation beats a bag of keywords: a real
    title reads like a symptom ("Le VPN se deconnecte toutes les cinq
    minutes"), while ``poste / ecran / bleu`` reads like debug output. The
    keywords are kept alongside rather than used as the name.
    """
    if examples:
        shortest = min(examples, key=len)
        return shortest[:90].strip()
    return " / ".join(keywords[:3]) or "unlabelled"


def discover_subtopics(
    tickets: pd.DataFrame,
    backend: Literal["tfidf", "embeddings"] = "embeddings",
    within: str | None = "category",
    min_cluster_size: int = 20,
    min_parent_size: int = 60,
    merge_threshold: float = MERGE_THRESHOLD,
    embedding_model: str | None = None,
    previous: SubTopicModel | None = None,
    labeller: Any | None = None,
    random_state: int = 42,
) -> SubTopicModel:
    """Discover sub-topics, optionally inside each parent category.

    Args:
        tickets: Validated ticket frame.
        backend: Text representation. ``embeddings`` is the point of the
            exercise - TF-IDF groups tickets that share words, embeddings group
            tickets that share a meaning - but ``tfidf`` runs anywhere.
        within: Column defining the parent bucket, or ``None`` for one global
            pass. Per-category is the default because a root cause is reported
            inside the queue that owns it.
        min_cluster_size: Smallest group that counts as a sub-topic.
        min_parent_size: Categories smaller than this are left as a single
            topic rather than split into noise.
        merge_threshold: Centroid similarity above which two clusters are
            folded into one - chiefly how the same symptom written in two
            languages is reunited.
        embedding_model: Sentence-transformer override.
        previous: A model from an earlier run; matching topics keep their ids.
        labeller: Object with ``label(keywords, examples, category) -> str``.
            Defaults to the offline labeller.
        random_state: Seed.

    Returns:
        The fitted :class:`SubTopicModel`.
    """
    texts = combined_text(tickets)
    # Encoded once, for the whole corpus, so every centroid lives in the same
    # space. Encoding per category would fit a separate TF-IDF/SVD basis per
    # slice, and centroids from two categories would then be incomparable -
    # which breaks `assign` outright and makes any cross-category similarity
    # meaningless.
    encoder = build_encoder(backend, embedding_model, len(texts))
    features_all = _normalise(np.asarray(encoder.fit_transform(texts)))
    parents = (
        tickets[within].fillna("unknown").astype(str)
        if within and within in tickets.columns
        else pd.Series(["all"] * len(tickets), index=tickets.index)
    )

    assignments = np.array([""] * len(tickets), dtype=object)
    topics: list[SubTopic] = []
    previous_topics = previous.topics if previous else []

    for parent in sorted(parents.unique()):
        mask = (parents == parent).to_numpy()
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue

        parent_texts = [texts[i] for i in indices]
        features = features_all[indices]
        labels = (
            np.zeros(indices.size, dtype=int)
            if indices.size < min_parent_size
            else _cluster(features, min_cluster_size, random_state)
        )
        labels = _merge_similar(features, labels, merge_threshold)

        normalised = [normalize_text(text) for text in parent_texts]
        terms = class_tfidf(normalised, labels, top_n=8)

        for local_id in sorted(set(int(v) for v in labels) - {-1}):
            members = np.flatnonzero(labels == local_id)
            if members.size < min(min_cluster_size, indices.size):
                continue

            centroid = _normalise(features[members].mean(axis=0, keepdims=True))[0]
            distances = features[members] @ centroid
            closest = members[np.argsort(distances)[::-1][:5]]
            examples = [str(tickets.iloc[indices[i]]["designation"]) for i in closest]
            keywords = [term for term, _ in terms.get(local_id, [])]

            label = (
                labeller.label(keywords, examples, parent)
                if labeller is not None
                else _offline_label(keywords, examples)
            )
            topic_id = _stable_id(centroid, parent, previous_topics, len(topics))
            topics.append(
                SubTopic(
                    topic_id=topic_id,
                    parent_category=parent,
                    label=label,
                    keywords=keywords,
                    size=int(members.size),
                    examples=examples[:3],
                    centroid=centroid,
                )
            )
            assignments[indices[members]] = topic_id

    coverage = float((assignments != "").mean()) if len(assignments) else 0.0
    logger.info("discovered %d sub-topics, coverage %.1f%%", len(topics), 100 * coverage)
    return SubTopicModel(
        topics=topics,
        assignments=assignments,
        backend=backend,
        coverage=coverage,
        encoder=encoder,
    )


def _stable_id(centroid: np.ndarray, parent: str, previous: list[SubTopic], counter: int) -> str:
    """Reuse the id of a matching topic from the previous run, or mint a new one.

    Without this, re-running the discovery renames every topic and the trend
    you were tracking last month points at something else this month - which is
    the failure mode that makes automatic tagging useless in practice.
    """
    candidates = [
        topic
        for topic in previous
        if topic.parent_category == parent and topic.centroid is not None
    ]
    if candidates:
        similarities = np.array([float(centroid @ topic.centroid) for topic in candidates])
        best = int(similarities.argmax())
        if similarities[best] >= STABLE_ID_THRESHOLD:
            return candidates[best].topic_id

    prefix = "".join(word[0] for word in parent.split()[:2]).upper() or "T"
    return f"{prefix}-{counter:03d}"


def attribute_growth(
    tickets: pd.DataFrame,
    assignments: np.ndarray | list[str],
    model: SubTopicModel,
    recent_days: int = 14,
    baseline_days: int = 56,
    as_of: pd.Timestamp | None = None,
    min_recent: int = 5,
) -> pd.DataFrame:
    """Decompose the change in ticket volume across sub-topics.

    Args:
        tickets: Validated ticket frame.
        assignments: Topic id per ticket, aligned with ``tickets``.
        model: The fitted model, used for labels.
        recent_days: Length of the window being explained.
        baseline_days: Length of the comparison window immediately before it.
        as_of: End of the recent window; defaults to the last ``opened_at``.
        min_recent: Topics with fewer recent tickets are not ranked - a jump
            from 0 to 2 is not a root cause, it is noise.

    Returns:
        One row per sub-topic, sorted by absolute contribution to the increase.

    Note:
        ``share_of_increase`` is a share of the **total** rise, including the
        sub-topics too small to be ranked, so the displayed column sums to less
        than 1. Rescaling it to the rows shown would inflate every headline
        figure, which is the opposite of what an attribution is for.
    """
    frame = tickets.assign(topic_id=list(assignments))
    opened = pd.to_datetime(frame["opened_at"])
    as_of = as_of or opened.max()

    recent_start = as_of - pd.Timedelta(days=recent_days)
    baseline_start = recent_start - pd.Timedelta(days=baseline_days)

    recent = frame[(opened > recent_start) & (opened <= as_of)]
    baseline = frame[(opened > baseline_start) & (opened <= recent_start)]

    topics = model.by_id()
    recent_counts = recent["topic_id"].value_counts()
    baseline_counts = baseline["topic_id"].value_counts()

    rows: list[dict[str, Any]] = []
    for topic_id in set(recent_counts.index) | set(baseline_counts.index):
        if not topic_id:
            continue
        n_recent = int(recent_counts.get(topic_id, 0))
        n_baseline = int(baseline_counts.get(topic_id, 0))
        recent_rate = n_recent / recent_days
        baseline_rate = n_baseline / baseline_days
        expected = baseline_rate * recent_days

        topic = topics.get(topic_id)
        rows.append(
            {
                "topic_id": topic_id,
                "parent_category": topic.parent_category if topic else "",
                "label": topic.label if topic else "",
                "keywords": ", ".join(topic.keywords[:5]) if topic else "",
                "n_recent": n_recent,
                "n_baseline": n_baseline,
                "recent_per_day": round(recent_rate, 3),
                "baseline_per_day": round(baseline_rate, 3),
                "delta_per_day": round(recent_rate - baseline_rate, 3),
                "growth_ratio": (
                    round(recent_rate / baseline_rate, 2) if baseline_rate > 0 else None
                ),
                "is_emerging": baseline_rate == 0 and n_recent >= min_recent,
                # Probability of seeing at least this many tickets if the rate
                # had not changed. Small means the rise is not sampling noise.
                "poisson_p": (
                    round(float(stats.poisson.sf(n_recent - 1, max(expected, 1e-9))), 6)
                    if n_recent > 0
                    else 1.0
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    increases = result.loc[result["delta_per_day"] > 0, "delta_per_day"].sum()
    result["share_of_increase"] = (
        (result["delta_per_day"].clip(lower=0) / increases).round(4) if increases > 0 else 0.0
    )
    result = result[(result["n_recent"] >= min_recent) | (result["delta_per_day"] < 0)]
    return result.sort_values("delta_per_day", ascending=False).reset_index(drop=True)
