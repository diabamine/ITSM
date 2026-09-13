"""Liveness, readiness and model introspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from itsm import __version__
from itsm.api.deps import SettingsDep, StoreDep
from itsm.api.schemas import HealthResponse, ModelInfo
from itsm.pipelines.run_analysis import CLUSTERER_ARTIFACT
from itsm.pipelines.train_classifier import ARTIFACT_NAME as CLASSIFIER_ARTIFACT

router = APIRouter(tags=["service"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
def healthz(settings: SettingsDep, store: StoreDep) -> HealthResponse:
    """Return ``ok`` as soon as the process serves traffic."""
    return HealthResponse(
        status="ok",
        version=__version__,
        dataset=settings.dataset,
        artifacts={
            CLASSIFIER_ARTIFACT: store.exists(CLASSIFIER_ARTIFACT),
            CLUSTERER_ARTIFACT: store.exists(CLUSTERER_ARTIFACT),
        },
    )


@router.get("/readyz", response_model=HealthResponse, summary="Readiness probe")
def readyz(settings: SettingsDep, store: StoreDep) -> HealthResponse:
    """Return ``degraded`` when a model endpoint would answer 503.

    The service stays *live* without artifacts - the statistics endpoints work
    - but it is not *ready* to serve predictions.
    """
    artifacts = {
        CLASSIFIER_ARTIFACT: store.exists(CLASSIFIER_ARTIFACT),
        CLUSTERER_ARTIFACT: store.exists(CLUSTERER_ARTIFACT),
    }
    return HealthResponse(
        status="ok" if all(artifacts.values()) else "degraded",
        version=__version__,
        dataset=settings.dataset,
        artifacts=artifacts,
    )


@router.get("/v1/models", response_model=list[ModelInfo], summary="List trained models")
def list_models(store: StoreDep) -> list[ModelInfo]:
    """Return every stored artifact with its training metadata and metrics.

    Exposing the evaluation scores next to the model is deliberate: a consumer
    of the API can check what the prediction it just received is worth.
    """
    return [
        ModelInfo(**{k: v for k, v in item.items() if k in ModelInfo.model_fields})
        for item in store.list_artifacts()
    ]
