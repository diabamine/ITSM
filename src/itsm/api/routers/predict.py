"""Model-serving endpoints: classification, sentiment and family assignment."""

from __future__ import annotations

from fastapi import APIRouter

from itsm.api.deps import ClassifierDep, ClustererDep, SentimentDep
from itsm.api.schemas import (
    ClassificationItem,
    ClassificationRequest,
    ClassificationResponse,
    ClusterAssignmentItem,
    ClusterAssignmentResponse,
    SentimentItem,
    SentimentResponse,
    TextBatch,
)

router = APIRouter(prefix="/v1", tags=["inference"])


@router.post("/classify", response_model=ClassificationResponse, summary="Classify ticket texts")
def classify(request: ClassificationRequest, classifier: ClassifierDep) -> ClassificationResponse:
    """Predict the category of each text, with an abstention flag.

    ``abstained=true`` means the model is not confident enough for automatic
    routing; the caller should send the ticket to human triage rather than
    trusting ``label``.
    """
    predictions = classifier.predict_with_confidence(
        request.texts, min_confidence=request.min_confidence, top_k=request.top_k
    )
    return ClassificationResponse(
        model_name=classifier.model,
        predictions=[
            ClassificationItem(
                text=text,
                label=prediction.label,
                confidence=prediction.confidence,
                abstained=prediction.abstained,
                alternatives=[
                    {"label": label, "score": score} for label, score in prediction.alternatives
                ],
            )
            for text, prediction in zip(request.texts, predictions, strict=True)
        ],
    )


@router.post("/sentiment", response_model=SentimentResponse, summary="Score requester sentiment")
def sentiment(request: TextBatch, analyzer: SentimentDep) -> SentimentResponse:
    """Return the frustration polarity of each text.

    ``score`` is a signed intensity in ``[-1, 1]``; ``evidence`` lists the
    markers that produced it when the lexicon backend is used.
    """
    results = analyzer.analyze(request.texts)
    return SentimentResponse(
        backend=analyzer.backend,
        results=[
            SentimentItem(
                text=text, label=result.label, score=result.score, evidence=result.evidence
            )
            for text, result in zip(request.texts, results, strict=True)
        ],
    )


@router.post(
    "/clusters/assign",
    response_model=ClusterAssignmentResponse,
    summary="Assign designations to a discovered family",
)
def assign_cluster(request: TextBatch, clusterer: ClustererDep) -> ClusterAssignmentResponse:
    """Map each designation to the nearest family found by the analysis."""
    labels = clusterer.predict(request.texts)
    names = {summary.cluster_id: summary.name for summary in getattr(clusterer, "summaries_", [])}
    return ClusterAssignmentResponse(
        results=[
            ClusterAssignmentItem(
                text=text,
                cluster_id=int(label),
                cluster_name=names.get(int(label), f"cluster {int(label)}"),
            )
            for text, label in zip(request.texts, labels, strict=True)
        ]
    )
