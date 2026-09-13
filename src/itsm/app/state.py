"""Cached resources shared by the Streamlit pages.

Streamlit reruns the whole script on every interaction, so anything expensive
must be cached explicitly:

* ``st.cache_resource`` for unhashable, process-wide singletons (models);
* ``st.cache_data`` for serialisable results (dataframes, computed reports).

Nothing here implements analysis: every function delegates to the same library
the CLI and the API use. The app is a client, not a second implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from itsm.analysis.clustering import DesignationClusterer
from itsm.analysis.embedding_map import EmbeddingMap, combined_text, embed, project
from itsm.config import Settings, get_settings
from itsm.data.loading import load_anomaly_manifest, load_events, load_tickets
from itsm.models.artifacts import ArtifactStore
from itsm.models.classifier import TicketClassifier
from itsm.models.sentiment import build_sentiment
from itsm.models.weak_signal import DetectorConfig, WeakSignalDetector, build_count_series
from itsm.pipelines.run_analysis import CLUSTERER_ARTIFACT
from itsm.pipelines.train_classifier import ARTIFACT_NAME as CLASSIFIER_ARTIFACT


@st.cache_resource(show_spinner=False)
def settings() -> Settings:
    """Process settings."""
    current = get_settings()
    current.ensure_dirs()
    return current


@st.cache_resource(show_spinner=False)
def store() -> ArtifactStore:
    """Artifact store."""
    return ArtifactStore(settings().artifacts_dir)


def artifacts_ready() -> dict[str, bool]:
    """Return which artifacts the app needs and whether they exist."""
    current = store()
    return {
        CLASSIFIER_ARTIFACT: current.exists(CLASSIFIER_ARTIFACT),
        CLUSTERER_ARTIFACT: current.exists(CLUSTERER_ARTIFACT),
    }


@st.cache_data(show_spinner="Loading tickets...")
def tickets() -> pd.DataFrame:
    """Curated ticket extract."""
    return load_tickets(settings())


@st.cache_data(show_spinner="Loading status updates...")
def events() -> pd.DataFrame:
    """Status-update log."""
    return load_events(settings())


@st.cache_data(show_spinner=False)
def anomaly_manifest() -> list[dict[str, Any]]:
    """Injected anomalies of the curated dataset, when the source has them."""
    return load_anomaly_manifest(settings())


@st.cache_resource(show_spinner="Loading the classifier...")
def classifier() -> TicketClassifier:
    """Trained category classifier."""
    return store().load(CLASSIFIER_ARTIFACT)


@st.cache_resource(show_spinner="Loading the clusterer...")
def clusterer() -> DesignationClusterer:
    """Fitted designation clusterer."""
    return store().load(CLUSTERER_ARTIFACT)


@st.cache_resource(show_spinner=False)
def sentiment() -> Any:
    """Configured sentiment analyser."""
    current = settings()
    return build_sentiment(current.sentiment_backend, current.sentiment_model)


@st.cache_data(show_spinner=False)
def count_series(by: str, freq: str) -> pd.DataFrame:
    """Arrival count series, one column per group."""
    return build_count_series(tickets(), by=by, freq=freq)


@st.cache_data(show_spinner="Scanning for weak signals...")
def scan(
    by: str, freq: str, z_spike: float, ewma_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the detector and return ``(alerts, diagnostics)``.

    Cached on the four parameters, so moving a threshold slider re-runs the
    detection but nothing else.
    """
    detector = WeakSignalDetector(DetectorConfig(z_spike=z_spike, ewma_threshold=ewma_threshold))
    report = detector.detect(count_series(by, freq))
    return report.to_frame(), report.diagnostics


@st.cache_data(show_spinner=False)
def routing_table() -> dict[str, str]:
    """Most frequent assignment group per category, for routing suggestions."""
    frame = tickets()
    return (
        frame.groupby("category")["assignment_group"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )


@st.cache_data(show_spinner=False)
def resolution_reference() -> pd.DataFrame:
    """Median and p90 resolution time per category, used as an ETA baseline."""
    frame = tickets().assign(_hours=pd.to_numeric(tickets()["resolution_hours"], errors="coerce"))
    return (
        frame.groupby("category")["_hours"]
        .agg(median_hours="median", p90_hours=lambda s: float(np.nanpercentile(s.dropna(), 90)))
        .round(1)
    )


@st.cache_resource(show_spinner="Indexing past tickets...")
def corpus_matrix() -> np.ndarray:
    """Vector representation of every past designation.

    Reuses the clusterer pipeline, so a "similar ticket" is similar in exactly
    the space the families were built in. Vectors are already L2-normalised,
    hence the dot product below is a cosine similarity.
    """
    return np.asarray(fitted_pipeline().transform(tickets()["designation"].astype(str).tolist()))


def fitted_pipeline() -> Pipeline:
    """Return the clusterer text pipeline, or fail with an actionable message."""
    pipeline = clusterer().pipeline_
    if pipeline is None:
        raise RuntimeError("the stored clusterer is not fitted; re-run `itsm analyze designations`")
    return pipeline


def similar_tickets(text: str, k: int = 5) -> pd.DataFrame:
    """Return the k most similar past tickets and how they were handled."""
    query = np.asarray(fitted_pipeline().transform([text]))
    similarity = corpus_matrix() @ query.T
    top = np.argsort(similarity.ravel())[::-1][:k]
    columns = [
        "ticket_id",
        "designation",
        "category",
        "assignment_group",
        "priority",
        "resolution_hours",
    ]
    frame = tickets().iloc[top][columns].copy()
    frame["similarity"] = similarity.ravel()[top].round(3)
    return frame.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def embedding_sample(sample_n: int, seed: int) -> pd.DataFrame:
    """Sample tickets and attach their combined designation + description text.

    Split out from :func:`embedding_features` so the (expensive) embedding step
    below is never recomputed just because a downstream UMAP/HDBSCAN parameter
    changed.
    """
    frame = tickets()
    if 0 < sample_n < len(frame):
        frame = frame.sample(n=sample_n, random_state=seed)
    frame = frame.reset_index(drop=True)
    return frame.assign(text=combined_text(frame))


@st.cache_data(show_spinner="Embedding ticket text...")
def embedding_features(backend: str, sample_n: int, seed: int) -> np.ndarray:
    """Dense per-ticket vectors for the 2D map, cached independently of it."""
    frame = embedding_sample(sample_n, seed)
    model_name = settings().embedding_model if backend == "embeddings" else None
    return embed(frame["text"].tolist(), backend=backend, embedding_model=model_name)  # type: ignore[arg-type]


@st.cache_data(show_spinner="Projecting to 2D with UMAP and clustering with HDBSCAN...")
def embedding_map(
    backend: str,
    sample_n: int,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    min_cluster_size: int,
) -> EmbeddingMap:
    """Fit the 2D map (UMAP + HDBSCAN) for the given sample and parameters."""
    frame = embedding_sample(sample_n, seed)
    features = embedding_features(backend, sample_n, seed)
    return project(
        features=features,
        texts=frame["text"].tolist(),
        categories=frame["category"],
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        min_cluster_size=min_cluster_size,
        random_state=seed,
    )


def bootstrap() -> None:
    """Build the dataset and train the artifacts the app needs.

    Exposed as an explicit button rather than run on start-up: training is a
    side effect, and a page load should not silently overwrite a model.
    """
    from itsm.data.loading import build_dataset
    from itsm.pipelines.run_analysis import run_designation_analysis
    from itsm.pipelines.train_classifier import run as train_classifier

    current = settings()
    with st.status("Preparing the demo...", expanded=True) as status:
        st.write("Building the dataset")
        build_dataset(current)
        st.write("Training the classifier")
        report = train_classifier(current, compare_baselines=False)
        st.write(f"macro-F1 = {report['metrics']['macro_f1']:.3f}")
        st.write("Discovering ticket families")
        run_designation_analysis(current)
        status.update(label="Ready", state="complete", expanded=False)
    st.cache_data.clear()
    st.cache_resource.clear()
