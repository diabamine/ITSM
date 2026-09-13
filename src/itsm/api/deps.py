"""Dependency providers for the API.

Models and dataframes are loaded lazily and cached for the process lifetime:
the first request that needs the classifier pays the deserialisation, the rest
are served from memory. Nothing is loaded at import time, so the app starts
even when no model has been trained yet - ``/readyz`` reports what is missing
instead of crashing at boot.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

import pandas as pd
from fastapi import Depends, HTTPException, status

from itsm.analysis.clustering import DesignationClusterer
from itsm.config import Settings, get_settings
from itsm.data.loading import load_events, load_tickets
from itsm.models.artifacts import ArtifactStore
from itsm.models.classifier import TicketClassifier
from itsm.models.sentiment import LexiconSentiment, TransformerSentiment, build_sentiment
from itsm.pipelines.run_analysis import CLUSTERER_ARTIFACT
from itsm.pipelines.train_classifier import ARTIFACT_NAME as CLASSIFIER_ARTIFACT


@lru_cache(maxsize=1)
def get_store() -> ArtifactStore:
    """Return the process-wide artifact store."""
    return ArtifactStore(get_settings().artifacts_dir)


@lru_cache(maxsize=1)
def get_classifier() -> TicketClassifier:
    """Return the trained classifier, or 503 if it has not been trained."""
    try:
        return get_store().load(CLASSIFIER_ARTIFACT)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


@lru_cache(maxsize=1)
def get_clusterer() -> DesignationClusterer:
    """Return the fitted clusterer, or 503 if the analysis has not been run."""
    try:
        return get_store().load(CLUSTERER_ARTIFACT)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


@lru_cache(maxsize=1)
def get_sentiment() -> LexiconSentiment | TransformerSentiment:
    """Return the configured sentiment analyser."""
    settings = get_settings()
    return build_sentiment(settings.sentiment_backend, settings.sentiment_model)


@lru_cache(maxsize=1)
def get_tickets() -> pd.DataFrame:
    """Return the ticket table, cached for the process lifetime."""
    return load_tickets(get_settings())


@lru_cache(maxsize=1)
def get_ticket_events() -> pd.DataFrame:
    """Return the event table, cached for the process lifetime."""
    return load_events(get_settings())


def reset_caches() -> None:
    """Clear every cached dependency; used by tests and after a retrain."""
    for provider in (
        get_store,
        get_classifier,
        get_clusterer,
        get_sentiment,
        get_tickets,
        get_ticket_events,
    ):
        provider.cache_clear()


SettingsDep = Annotated[Settings, Depends(get_settings)]
StoreDep = Annotated[ArtifactStore, Depends(get_store)]
ClassifierDep = Annotated[TicketClassifier, Depends(get_classifier)]
ClustererDep = Annotated[DesignationClusterer, Depends(get_clusterer)]
SentimentDep = Annotated["LexiconSentiment | TransformerSentiment", Depends(get_sentiment)]
TicketsDep = Annotated[pd.DataFrame, Depends(get_tickets)]
EventsDep = Annotated[pd.DataFrame, Depends(get_ticket_events)]
