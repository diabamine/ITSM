"""Evaluation pipeline for sentiment / frustration scoring.

**The honest caveat first.** There is no gold sentiment annotation on ITSM
tickets, and buying one is expensive. This pipeline therefore scores models
against a *weak label*: on the synthetic dataset, the requester mood that
generated the wording; on real data, a proxy built from operational evidence
(reopened, SLA breached, several reassignments), which is what a business
stakeholder means by "an unhappy requester" anyway.

A weak label is not a gold label: it is noisy and it correlates with variables
the model never sees. The numbers below are therefore valid for *comparing*
backends on equal footing and for detecting gross failures - not for claiming
an absolute accuracy. The path to a defensible figure is a few hundred
human-annotated tickets; :func:`sample_for_annotation` produces the batch to
send for annotation, prioritising the cases where backends disagree.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from itsm.config import Settings, get_settings
from itsm.data.loading import load_tickets
from itsm.logging import get_logger
from itsm.models.sentiment import build_sentiment
from itsm.pipelines.reporting import write_report

logger = get_logger("pipelines.eval_sentiment")

LABELS = ("negative", "neutral", "positive")


def operational_weak_labels(tickets: pd.DataFrame) -> pd.Series:
    """Derive a frustration proxy from operational evidence.

    A ticket is labelled ``negative`` when it was reopened, or breached its SLA
    while being reassigned at least twice - the situations that generate
    complaints. Everything else is ``neutral``; ``positive`` is never inferred
    from operations, only from text.
    """
    reopened = tickets["reopen_count"].fillna(0) > 0
    churned = tickets["sla_breached"].fillna(False).astype(bool) & (
        tickets["reassignment_count"].fillna(0) >= 2
    )
    return pd.Series(np.where(reopened | churned, "negative", "neutral"), index=tickets.index)


def sample_for_annotation(
    tickets: pd.DataFrame, predictions: dict[str, list[str]], n: int = 200, seed: int = 42
) -> pd.DataFrame:
    """Select the tickets worth sending to human annotation.

    Disagreement between backends is the cheapest proxy for difficulty, so the
    batch is built from disagreements first, then filled with a random sample
    to keep the set representative.
    """
    frame = tickets[["ticket_id", "designation", "description", "language"]].copy()
    for name, labels in predictions.items():
        frame[f"pred_{name}"] = labels

    prediction_columns = [c for c in frame.columns if c.startswith("pred_")]
    frame["disagreement"] = frame[prediction_columns].nunique(axis=1) > 1

    rng = np.random.default_rng(seed)
    disputed = frame[frame["disagreement"]]
    remainder = frame[~frame["disagreement"]]
    take_random = max(n - len(disputed), 0)
    sampled = remainder.iloc[
        rng.choice(len(remainder), size=min(take_random, len(remainder)), replace=False)
    ]
    return pd.concat([disputed.head(n), sampled]).head(n).reset_index(drop=True)


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0, labels=list(LABELS)
    )
    return {
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "negative_recall": round(float(report["negative"]["recall"]), 4),
        "negative_precision": round(float(report["negative"]["precision"]), 4),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
        },
    }


def run(
    settings: Settings | None = None,
    backends: tuple[Literal["lexicon", "transformer"], ...] = ("lexicon",),
    text_column: str = "description",
    label_source: Literal["generator", "operational"] = "generator",
    sample_size: int | None = 4000,
) -> dict[str, Any]:
    """Score the available sentiment backends against the weak labels.

    Args:
        settings: Configuration.
        backends: Backends to evaluate; add ``"transformer"`` when the ``[hf]``
            extra is installed.
        text_column: Column fed to the analysers.
        label_source: ``generator`` uses the synthetic mood label,
            ``operational`` derives the proxy from reopen / SLA evidence.
        sample_size: Cap on the number of tickets scored, for runtime.

    Returns:
        The evaluation report, also written to ``reports/sentiment.json``.
    """
    settings = settings or get_settings()
    tickets = load_tickets(settings)
    if sample_size and len(tickets) > sample_size:
        tickets = tickets.sample(sample_size, random_state=42).reset_index(drop=True)

    if label_source == "generator" and tickets["frustration_label"].notna().any():
        truth = tickets["frustration_label"].astype(str)
        label_note = "weak label from the synthetic generator (mood that produced the wording)"
    else:
        truth = operational_weak_labels(tickets)
        label_note = "operational proxy: reopened, or SLA breach with >= 2 reassignments"

    texts = tickets[text_column].fillna(tickets["designation"]).astype(str).tolist()
    predictions: dict[str, list[str]] = {}
    results: dict[str, Any] = {}

    for backend in backends:
        analyzer = build_sentiment(
            backend, settings.sentiment_model if backend == "transformer" else None
        )
        labels: list[str] = [str(result.label) for result in analyzer.analyze(texts)]
        predictions[backend] = labels
        results[backend] = _score(truth.to_numpy(), np.asarray(labels))
        logger.info("%s -> macro_f1=%.4f", backend, results[backend]["macro_f1"])

    report = {
        "dataset": settings.dataset,
        "n_scored": len(tickets),
        "label_source": label_source,
        "label_caveat": label_note,
        "class_balance": truth.value_counts(normalize=True).round(4).to_dict(),
        "backends": results,
        "annotation_batch_size": len(
            sample_for_annotation(tickets, predictions, n=min(200, len(tickets)))
        ),
    }
    write_report("sentiment", report, settings)
    return report
