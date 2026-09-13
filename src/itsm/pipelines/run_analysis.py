"""Descriptive analysis pipeline: ticket families and lifecycle statistics.

This is step 1 of the project - the pass that runs *before* any model, to
answer: what families exist in the designations, does the declared taxonomy
match them, and what do the status updates say about how work flows.

The fitted clusterer is persisted so the API can assign an incoming ticket to a
family without refitting.
"""

from __future__ import annotations

from typing import Any, Literal

from itsm.analysis.clustering import DesignationClusterer
from itsm.analysis.lifecycle import (
    aging_backlog,
    backlog_timeline,
    dwell_times,
    kpi_overview,
    kpis_by,
    rework_profile,
    state_transitions,
    volume_timeline,
)
from itsm.config import Settings, get_settings
from itsm.data.loading import load_events, load_tickets
from itsm.logging import get_logger
from itsm.models.artifacts import ArtifactMetadata, ArtifactStore
from itsm.pipelines.reporting import write_report

logger = get_logger("pipelines.run_analysis")

CLUSTERER_ARTIFACT = "clusterer"


def run_designation_analysis(
    settings: Settings | None = None,
    n_clusters: int | None = None,
    backend: Literal["tfidf", "embeddings"] | None = None,
    algorithm: Literal["kmeans", "hdbscan"] = "kmeans",
    save: bool = True,
) -> dict[str, Any]:
    """Cluster the designations and describe the discovered families.

    Args:
        settings: Configuration.
        n_clusters: Number of families; ``None`` selects k by silhouette.
        backend: Text representation; defaults to the configured backend.
        algorithm: ``kmeans`` or ``hdbscan``.
        save: Persist the fitted clusterer.

    Returns:
        The report, also written to ``reports/designation_analysis.json``.
    """
    settings = settings or get_settings()
    tickets = load_tickets(settings)
    designations = tickets["designation"].astype(str).tolist()

    clusterer = DesignationClusterer(
        n_clusters=n_clusters,
        backend=backend or settings.embedding_backend,
        algorithm=algorithm,
        embedding_model=settings.embedding_model,
    )
    result = clusterer.fit(designations, tickets["category"])

    report = {
        "dataset": settings.dataset,
        "n_tickets": len(tickets),
        "backend": clusterer.backend,
        "algorithm": algorithm,
        "metrics": result.metrics,
        "clusters": [summary.__dict__ for summary in result.summaries],
        "taxonomy_alignment": {
            "declared_categories": int(tickets["category"].nunique()),
            "discovered_clusters": result.n_clusters,
            "mean_cluster_purity": round(
                float(
                    sum(s.category_purity or 0 for s in result.summaries)
                    / max(result.n_clusters, 1)
                ),
                4,
            ),
        },
    }

    if save:
        ArtifactStore(settings.artifacts_dir).save(
            CLUSTERER_ARTIFACT,
            clusterer,
            ArtifactMetadata(
                name=CLUSTERER_ARTIFACT,
                kind="clustering",
                dataset=settings.dataset,
                n_train=len(designations),
                params={
                    "backend": clusterer.backend,
                    "algorithm": algorithm,
                    "k": result.n_clusters,
                },
                metrics=result.metrics,
                notes="Assigns a designation to a discovered ticket family.",
            ),
        )

    write_report("designation_analysis", report, settings)
    return report


def run_lifecycle_analysis(
    settings: Settings | None = None,
    dimensions: tuple[str, ...] = ("category", "priority", "assignment_group"),
) -> dict[str, Any]:
    """Compute ticket and status-update statistics.

    Args:
        settings: Configuration.
        dimensions: Columns to break the indicators down by.

    Returns:
        The report, also written to ``reports/lifecycle_analysis.json``.
    """
    settings = settings or get_settings()
    tickets = load_tickets(settings)
    events = load_events(settings)

    report: dict[str, Any] = {
        "dataset": settings.dataset,
        "overview": kpi_overview(tickets),
        "by": {
            dimension: kpis_by(tickets, dimension).to_dict(orient="records")
            for dimension in dimensions
            if dimension in tickets.columns
        },
        "weekly_volume": volume_timeline(tickets, freq="W").to_dict(orient="records"),
        "backlog": backlog_timeline(tickets, freq="W").to_dict(orient="records"),
        "aging": aging_backlog(tickets).to_dict(orient="records"),
    }

    if len(events):
        report["status_updates"] = {
            "transitions": state_transitions(events).to_dict(orient="records"),
            "dwell_times": dwell_times(events).to_dict(orient="records"),
            "rework": rework_profile(events),
        }
    else:
        report["status_updates"] = None
        report["note"] = "This dataset has no audit trail; flow statistics are unavailable."

    write_report("lifecycle_analysis", report, settings)
    logger.info("lifecycle analysis done: %s", report["overview"])
    return report


def run(settings: Settings | None = None, **kwargs: Any) -> dict[str, Any]:
    """Run both descriptive analyses and return the merged report."""
    return {
        "designations": run_designation_analysis(settings, **kwargs),
        "lifecycle": run_lifecycle_analysis(settings),
    }
