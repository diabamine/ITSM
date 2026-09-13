"""Evaluation protocol for the category classifier.

Two splits are supported and they answer different questions:

``stratified``
    Random split with preserved class proportions. Measures raw separability -
    the number to quote when comparing representations or estimators.

``temporal`` (default)
    Train on the past, test on the most recent slice. Lower and more honest:
    ticket vocabulary drifts (new applications, new error codes, a migration
    that renames everything), so a random split systematically flatters a model
    that will be deployed on tomorrow's tickets.

Reported metrics:

* **macro-F1** as the headline - the taxonomy is heavily imbalanced and
  accuracy would be dominated by password resets.
* **balanced accuracy** and **top-3 accuracy** - the latter matters because a
  triage UI can propose three categories to the agent.
* **per-class F1 and support** - to see *which* families fail, usually the
  small ones or the two that genuinely overlap.
* **coverage/accuracy trade-off** - accuracy restricted to the tickets the
  model is confident about, which is the only curve that decides how much
  routing can actually be automated.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from itsm.logging import get_logger
from itsm.models.classifier import ModelName, TicketClassifier

logger = get_logger("pipelines.eval_classifier")

SplitStrategy = Literal["temporal", "stratified"]


def split_dataset(
    tickets: pd.DataFrame,
    text_column: str = "designation",
    label_column: str = "category",
    strategy: SplitStrategy = "temporal",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Split tickets into train and test sets.

    Returns:
        ``(x_train, x_test, y_train, y_test)``.
    """
    frame = tickets.dropna(subset=[text_column, label_column])
    if strategy == "temporal":
        frame = frame.sort_values("opened_at")
        cut = int(len(frame) * (1 - test_size))
        train, test = frame.iloc[:cut], frame.iloc[cut:]
        # Classes absent from training cannot be predicted; drop them from the
        # test set rather than silently counting them as errors.
        test = test[test[label_column].isin(set(train[label_column]))]
    else:
        train, test = train_test_split(
            frame, test_size=test_size, random_state=random_state, stratify=frame[label_column]
        )
    return (
        train[text_column].tolist(),
        test[text_column].tolist(),
        train[label_column].to_numpy(),
        test[label_column].to_numpy(),
    )


def coverage_accuracy_curve(
    probabilities: np.ndarray, classes: np.ndarray, y_true: np.ndarray, steps: int = 10
) -> list[dict[str, float]]:
    """Accuracy as a function of the confidence threshold.

    Each point answers: *if we auto-route only the tickets above this
    confidence, what share of the volume is covered and how often are we
    wrong?*
    """
    confidence = probabilities.max(axis=1)
    predicted = classes[probabilities.argmax(axis=1)]
    correct = predicted == y_true

    curve: list[dict[str, float]] = []
    for threshold in np.round(np.linspace(0.0, 0.9, steps), 2):
        mask = confidence >= threshold
        if not mask.any():
            continue
        curve.append(
            {
                "threshold": round(float(threshold), 3),
                "coverage": round(float(mask.mean()), 4),
                "accuracy": round(float(correct[mask].mean()), 4),
                "error_rate": round(float(1 - correct[mask].mean()), 4),
            }
        )
    return curve


def top_k_accuracy(
    probabilities: np.ndarray, classes: np.ndarray, y_true: np.ndarray, k: int = 3
) -> float:
    """Share of tickets whose true category is in the k most likely."""
    top_k = np.argsort(probabilities, axis=1)[:, -k:]
    hits = [y_true[i] in classes[top_k[i]] for i in range(len(y_true))]
    return float(np.mean(hits))


def evaluate(classifier: TicketClassifier, x_test: list[str], y_test: np.ndarray) -> dict[str, Any]:
    """Evaluate a fitted classifier on a held-out set."""
    probabilities = classifier.predict_proba(x_test)
    classes = np.asarray(classifier.classes_)
    predicted = classes[probabilities.argmax(axis=1)]

    report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    per_class = {
        label: {
            "f1": round(float(scores["f1-score"]), 4),
            "precision": round(float(scores["precision"]), 4),
            "recall": round(float(scores["recall"]), 4),
            "support": int(scores["support"]),
        }
        for label, scores in report.items()
        if isinstance(scores, dict) and label not in {"macro avg", "weighted avg", "accuracy"}
    }

    return {
        "n_test": len(y_test),
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predicted)), 4),
        "macro_f1": round(float(f1_score(y_test, predicted, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(y_test, predicted, average="weighted", zero_division=0)), 4
        ),
        "top3_accuracy": round(top_k_accuracy(probabilities, classes, y_test, k=3), 4),
        "per_class": per_class,
        "worst_classes": sorted(per_class.items(), key=lambda kv: kv[1]["f1"])[:3],
        "coverage_curve": coverage_accuracy_curve(probabilities, classes, y_test),
        "confusion_matrix": {
            "labels": [str(c) for c in classes],
            "matrix": confusion_matrix(y_test, predicted, labels=classes).tolist(),
        },
    }


def cross_validate(
    texts: list[str], labels: np.ndarray, model: ModelName = "linear_svc", n_splits: int = 5
) -> dict[str, float | list[float]]:
    """Stratified k-fold macro-F1, to size the variance of the estimate."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: list[float] = []
    texts_array = np.asarray(texts)
    for train_index, test_index in folds.split(texts_array, labels):
        classifier = TicketClassifier(model=model).fit(
            texts_array[train_index].tolist(), labels[train_index]
        )
        predicted = classifier.predict(texts_array[test_index].tolist())
        scores.append(
            float(f1_score(labels[test_index], predicted, average="macro", zero_division=0))
        )
    return {
        "macro_f1_mean": round(float(np.mean(scores)), 4),
        "macro_f1_std": round(float(np.std(scores)), 4),
        "folds": [round(s, 4) for s in scores],
    }


def compare_models(
    x_train: list[str],
    x_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    models: tuple[ModelName, ...] = ("majority", "complement_nb", "logreg", "linear_svc"),
) -> pd.DataFrame:
    """Train and score several estimators on the same split.

    The ``majority`` baseline is included on purpose: a macro-F1 that does not
    clearly beat it means the pipeline is broken, not that the task is hard.
    """
    rows: list[dict[str, Any]] = []
    for name in models:
        classifier = TicketClassifier(model=name).fit(x_train, y_train)
        metrics = evaluate(classifier, x_test, y_test)
        logger.info("%s -> macro_f1=%.4f", name, metrics["macro_f1"])
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "top3_accuracy": metrics["top3_accuracy"],
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
