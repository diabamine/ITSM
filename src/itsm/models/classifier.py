"""Ticket classification into operational categories.

The task: from the designation (plus, optionally, the body), predict the
category that decides routing. Three design choices drive the implementation.

**Linear models on n-grams are the right default.** On short, templated,
domain-specific texts with tens of thousands of examples, a calibrated linear
classifier over word+char TF-IDF is within a couple of F1 points of a
fine-tuned transformer, trains in seconds on CPU, and stays inspectable -
``top_features_for`` shows exactly which tokens drive a class. The transformer
path exists (``backend="embeddings"``), and the evaluation pipeline compares
them on the same split, so the trade-off is measured rather than assumed.

**Probabilities matter more than labels.** Routing is only automated when the
model is confident; otherwise the ticket goes to a human. Every backend is
therefore calibrated and the classifier exposes an abstention threshold via
:meth:`TicketClassifier.predict_with_confidence`.

**Class imbalance is structural.** Password resets outnumber database
incidents by two orders of magnitude. Classes are weighted, and the headline
metric is macro-F1, not accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from itsm.logging import get_logger
from itsm.text.normalize import TextNormalizer
from itsm.text.vectorize import build_vectorizer

logger = get_logger("models.classifier")

ModelName = Literal["linear_svc", "logreg", "complement_nb", "majority"]


@dataclass
class Prediction:
    """One classification result, with its runner-up alternatives."""

    label: str
    confidence: float
    alternatives: list[tuple[str, float]]
    abstained: bool = False


def _build_estimator(model: ModelName, random_state: int) -> Any:
    """Instantiate the requested estimator, calibrated when needed."""
    if model == "linear_svc":
        # LinearSVC has no predict_proba: wrap it so routing can use a threshold.
        return CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", random_state=random_state),
            method="sigmoid",
            cv=3,
        )
    if model == "logreg":
        return LogisticRegression(
            C=4.0, max_iter=2000, class_weight="balanced", random_state=random_state
        )
    if model == "complement_nb":
        return ComplementNB(alpha=0.3)
    if model == "majority":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"unknown model {model!r}")


class TicketClassifier:
    """Category classifier over ticket text.

    Args:
        model: Estimator to use. ``majority`` is the trivial baseline every
            reported score must beat.
        backend: Text representation (``tfidf`` or ``embeddings``).
        min_confidence: Default abstention threshold used by
            :meth:`predict_with_confidence`.
        random_state: Seed.
        embedding_model: Override of the sentence-transformer model name.
    """

    def __init__(
        self,
        model: ModelName = "linear_svc",
        backend: Literal["tfidf", "embeddings"] = "tfidf",
        min_confidence: float = 0.55,
        random_state: int = 42,
        embedding_model: str | None = None,
    ) -> None:
        self.model = model
        self.backend = backend
        self.min_confidence = min_confidence
        self.random_state = random_state
        self.embedding_model = embedding_model
        self.pipeline_: Pipeline | None = None
        self.classes_: np.ndarray | None = None

    def build(self) -> Pipeline:
        """Assemble the (unfitted) end-to-end pipeline."""
        return Pipeline(
            [
                ("normalize", TextNormalizer()),
                ("vectorize", build_vectorizer(self.backend, self.embedding_model)),
                ("estimator", _build_estimator(self.model, self.random_state)),
            ]
        )

    def fit(self, texts: list[str], labels: list[str] | np.ndarray) -> TicketClassifier:
        """Fit the pipeline on raw designations and their categories."""
        self.pipeline_ = self.build()
        self.pipeline_.fit(texts, labels)
        self.classes_ = np.asarray(self.pipeline_.named_steps["estimator"].classes_)
        logger.info(
            "trained %s/%s on %d tickets, %d classes",
            self.model,
            self.backend,
            len(texts),
            len(self.classes_),
        )
        return self

    def _require_fitted(self) -> Pipeline:
        if self.pipeline_ is None:
            raise RuntimeError("classifier is not fitted")
        return self.pipeline_

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return the most likely category of each text."""
        return np.asarray(self._require_fitted().predict(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return the class probability matrix, shape ``(n_texts, n_classes)``."""
        pipeline = self._require_fitted()
        estimator = pipeline.named_steps["estimator"]
        if hasattr(estimator, "predict_proba"):
            return np.asarray(pipeline.predict_proba(texts))
        # Fall back to a softmax over decision values for uncalibrated models.
        scores = np.asarray(pipeline.decision_function(texts))
        scores = scores - scores.max(axis=1, keepdims=True)
        exponentials = np.exp(scores)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict_with_confidence(
        self, texts: list[str], min_confidence: float | None = None, top_k: int = 3
    ) -> list[Prediction]:
        """Predict with an abstention rule.

        Args:
            texts: Raw designations.
            min_confidence: Threshold below which the prediction is flagged
                ``abstained`` - the ticket should go to human triage.
            top_k: Number of alternatives to return.

        Returns:
            One :class:`Prediction` per input.
        """
        threshold = self.min_confidence if min_confidence is None else min_confidence
        probabilities = self.predict_proba(texts)
        classes = np.asarray(self.classes_)

        results: list[Prediction] = []
        for row in probabilities:
            order = np.argsort(row)[::-1][:top_k]
            alternatives = [(str(classes[i]), round(float(row[i]), 4)) for i in order]
            best_label, best_score = alternatives[0]
            results.append(
                Prediction(
                    label=best_label,
                    confidence=best_score,
                    alternatives=alternatives,
                    abstained=best_score < threshold,
                )
            )
        return results

    def top_features_for(self, label: str, n: int = 15) -> list[tuple[str, float]]:
        """Return the n-grams that push a text towards ``label``.

        Only available for linear models over TF-IDF; returns an empty list
        otherwise. This is the cheapest form of model explanation and the one
        support teams actually read.
        """
        pipeline = self._require_fitted()
        estimator = pipeline.named_steps["estimator"]
        if isinstance(estimator, CalibratedClassifierCV):
            calibrated = estimator.calibrated_classifiers_[0]
            base = getattr(calibrated, "estimator", None)
        else:
            base = estimator
        coefficients = getattr(base, "coef_", None)
        if coefficients is None or self.backend != "tfidf":
            return []

        vectorizer = pipeline.named_steps["vectorize"]
        names = np.asarray(vectorizer.get_feature_names_out())
        classes = np.asarray(getattr(base, "classes_", self.classes_))
        index = int(np.flatnonzero(classes == label)[0])
        weights = coefficients[index] if coefficients.ndim > 1 else coefficients[0]
        order = np.argsort(weights)[::-1][:n]
        return [(str(names[i]), round(float(weights[i]), 4)) for i in order]
