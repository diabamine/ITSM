"""Request and response models of the API.

Explicit Pydantic models rather than free-form dicts: they are the contract,
they generate the OpenAPI schema, and they keep the serving layer independent
from the internal dataframes.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class TextBatch(BaseModel):
    """A batch of ticket texts to score."""

    texts: list[str] = Field(
        min_length=1, max_length=512, examples=[["VPN disconnects every hour"]]
    )


class ClassificationRequest(TextBatch):
    """Classification request with an optional abstention threshold."""

    min_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Below this confidence the prediction is flagged for human triage.",
    )
    top_k: int = Field(default=3, ge=1, le=10)


class ClassificationItem(BaseModel):
    """Predicted category for one text."""

    text: str
    label: str
    confidence: float
    abstained: bool = Field(description="True when the ticket should go to human triage.")
    alternatives: list[dict[str, float | str]]


class ClassificationResponse(BaseModel):
    """Batch classification result."""

    model_name: str
    predictions: list[ClassificationItem]


class SentimentItem(BaseModel):
    """Sentiment of one text."""

    text: str
    label: Literal["negative", "neutral", "positive"]
    score: float = Field(description="Signed intensity in [-1, 1]; negative means frustrated.")
    evidence: list[str] = Field(default_factory=list)


class SentimentResponse(BaseModel):
    """Batch sentiment result."""

    backend: str
    results: list[SentimentItem]


class ClusterAssignmentItem(BaseModel):
    """Family assigned to one designation."""

    text: str
    cluster_id: int
    cluster_name: str


class ClusterAssignmentResponse(BaseModel):
    """Batch family assignment."""

    results: list[ClusterAssignmentItem]


class ClusterSummaryOut(BaseModel):
    """One discovered family."""

    cluster_id: int
    size: int
    share: float
    name: str
    top_terms: list[str]
    representatives: list[str]
    dominant_category: str | None = None
    category_purity: float | None = None


class SignalScanRequest(BaseModel):
    """Parameters of a weak-signal scan."""

    by: str = Field(default="category", description="Column defining the series.")
    freq: Literal["D", "W"] = "D"
    z_spike: float = Field(default=3.5, gt=0)
    ewma_threshold: float = Field(default=1.1, gt=0)
    since: date | None = Field(default=None, description="Only return alerts from this date on.")


class AlertOut(BaseModel):
    """One raised alert."""

    series: str
    period: date
    kind: Literal["spike", "emerging"]
    observed: int
    expected: float
    robust_z: float
    poisson_p: float
    ewma: float
    severity: float


class SignalScanResponse(BaseModel):
    """Result of a weak-signal scan."""

    n_series: int
    n_periods: int
    alerts: list[AlertOut]


class ModelInfo(BaseModel):
    """Metadata of a stored artifact."""

    name: str
    kind: str | None = None
    created_at: str | None = None
    dataset: str | None = None
    n_train: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class HealthResponse(BaseModel):
    """Liveness / readiness payload."""

    status: Literal["ok", "degraded"]
    version: str
    dataset: str
    artifacts: dict[str, bool]


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    detail: str
    request_id: str | None = None


class AnonymisationRequest(TextBatch):
    """Texts to strip of personal data."""


class SpanOut(BaseModel):
    """Location and type of one detected identifier.

    The matched value is deliberately absent: echoing it back into a response
    that will be logged would defeat the purpose of the call.
    """

    start: int
    end: int
    entity: str
    score: float


class AnonymisedItem(BaseModel):
    """One anonymised text."""

    text: str
    n_replaced: int
    spans: list[SpanOut]


class AnonymisationResponse(BaseModel):
    """Batch anonymisation result."""

    backend: str
    results: list[AnonymisedItem]
