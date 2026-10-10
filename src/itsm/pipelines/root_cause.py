"""Pipeline: tag tickets with sub-topics and rank what is driving the increase.

One command answers "why is volume up?" with a ranked, named list instead of a
category chart. The fitted taxonomy is persisted, so the next run reuses the
topic ids and a trend stays comparable from month to month.

On a corpus with no recent increase the ranking is still produced and simply
shows small, insignificant deltas - which is the honest output, not a failure.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from itsm.analysis.root_cause import SubTopicModel, attribute_growth, discover_subtopics
from itsm.analysis.topic_labeller import build_labeller
from itsm.config import Settings, get_settings
from itsm.data.loading import load_tickets
from itsm.logging import get_logger
from itsm.models.artifacts import ArtifactMetadata, ArtifactStore
from itsm.pipelines.reporting import write_report

logger = get_logger("pipelines.root_cause")

ARTIFACT_NAME = "subtopics"

#: A driver has to be both large and unlikely to be noise. The p-value alone
#: would promote a jump from 1 to 5 tickets; the share alone would promote a
#: big family drifting upwards with the business.
SIGNIFICANCE = 0.01
MIN_SHARE = 0.05


def run(
    settings: Settings | None = None,
    backend: Literal["tfidf", "embeddings"] | None = None,
    within: str | None = "category",
    recent_days: int = 14,
    baseline_days: int = 56,
    min_cluster_size: int = 20,
    labeller_kind: str = "template",
    labeller_options: dict[str, Any] | None = None,
    as_of: pd.Timestamp | None = None,
    reuse_previous: bool = True,
    save: bool = True,
    top_n: int = 15,
) -> dict[str, Any]:
    """Discover sub-topics and attribute the recent volume change to them.

    Args:
        settings: Configuration; defaults to the process settings.
        backend: Text representation; defaults to the configured backend.
        within: Parent column to sub-divide, or ``None`` for one global pass.
        recent_days: Length of the window being explained.
        baseline_days: Comparison window immediately before it.
        min_cluster_size: Smallest group that counts as a sub-topic.
        labeller_kind: ``template`` (offline), ``local`` or ``api``.
        labeller_options: Extra arguments for the LLM labellers.
        as_of: End of the recent window; defaults to the last ticket.
        reuse_previous: Inherit topic ids from the stored model.
        save: Persist the fitted taxonomy.
        top_n: Rows kept in the report.

    Returns:
        The report, also written to ``reports/root_cause.json``.
    """
    settings = settings or get_settings()
    backend = backend or settings.embedding_backend
    store = ArtifactStore(settings.artifacts_dir)

    tickets = load_tickets(settings)
    previous: SubTopicModel | None = None
    if reuse_previous and store.exists(ARTIFACT_NAME):
        previous = store.load(ARTIFACT_NAME)
        logger.info("reusing %d topic ids from the stored taxonomy", len(previous.topics))

    labeller = build_labeller(labeller_kind, **(labeller_options or {}))
    model = discover_subtopics(
        tickets,
        backend=backend,
        within=within,
        min_cluster_size=min_cluster_size,
        embedding_model=settings.embedding_model,
        previous=previous,
        labeller=labeller,
    )

    ranked = attribute_growth(
        tickets,
        model.assignments,
        model,
        recent_days=recent_days,
        baseline_days=baseline_days,
        as_of=as_of,
    )
    drivers = (
        ranked[(ranked["poisson_p"] <= SIGNIFICANCE) & (ranked["share_of_increase"] >= MIN_SHARE)]
        if not ranked.empty
        else ranked
    )

    report: dict[str, Any] = {
        "dataset": settings.dataset,
        "backend": backend,
        "labeller": getattr(labeller, "name", labeller_kind),
        "within": within,
        "recent_days": recent_days,
        "baseline_days": baseline_days,
        "n_tickets": len(tickets),
        "n_subtopics": len(model.topics),
        "coverage": round(model.coverage, 4),
        "n_drivers": len(drivers),
        "drivers": drivers.head(top_n).to_dict(orient="records"),
        "emerging": (
            ranked[ranked["is_emerging"]].head(top_n).to_dict(orient="records")
            if not ranked.empty
            else []
        ),
        "ranking": ranked.head(top_n).to_dict(orient="records"),
        "subtopics": [topic.as_dict() for topic in model.topics],
    }

    if save:
        store.save(
            ARTIFACT_NAME,
            model,
            ArtifactMetadata(
                name=ARTIFACT_NAME,
                kind="subtopics",
                dataset=settings.dataset,
                n_train=len(tickets),
                params={
                    "backend": backend,
                    "within": within,
                    "min_cluster_size": min_cluster_size,
                    "labeller": getattr(labeller, "name", labeller_kind),
                },
                metrics={"n_subtopics": len(model.topics), "coverage": round(model.coverage, 4)},
                notes="Sub-topic taxonomy; topic ids are stable across runs.",
            ),
        )

    write_report("root_cause", report, settings)
    if not drivers.empty:
        top = drivers.iloc[0]
        logger.info(
            "top driver: %s (%.0f%% of the increase, x%s)",
            top["label"],
            100 * top["share_of_increase"],
            top["growth_ratio"],
        )
    return report
