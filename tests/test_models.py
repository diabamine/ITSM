"""Classifier, sentiment and weak-signal behaviour."""

from __future__ import annotations

import pandas as pd
import pytest

from itsm.data.synthetic import generate
from itsm.models.classifier import TicketClassifier
from itsm.models.sentiment import LexiconSentiment
from itsm.models.weak_signal import (
    DetectorConfig,
    WeakSignalDetector,
    build_count_series,
    evaluate_detections,
)
from itsm.pipelines.eval_classifier import coverage_accuracy_curve, evaluate, split_dataset


def test_classifier_beats_the_majority_baseline(tickets: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = split_dataset(tickets, strategy="stratified")

    model = TicketClassifier(model="linear_svc").fit(x_train, y_train)
    baseline = TicketClassifier(model="majority").fit(x_train, y_train)

    scores = evaluate(model, x_test, y_test)
    baseline_scores = evaluate(baseline, x_test, y_test)

    assert scores["macro_f1"] > baseline_scores["macro_f1"] + 0.3
    assert scores["top3_accuracy"] >= scores["accuracy"]
    assert set(scores["per_class"]) <= set(map(str, tickets["category"].unique()))


def test_abstention_flags_low_confidence(tickets: pd.DataFrame) -> None:
    x_train, _, y_train, _ = split_dataset(tickets, strategy="stratified")
    model = TicketClassifier(model="logreg").fit(x_train, y_train)

    confident = model.predict_with_confidence(["Reinitialisation du mot de passe Windows"])
    gibberish = model.predict_with_confidence(["zzz qqq"], min_confidence=0.99)

    assert confident[0].confidence > 0
    assert gibberish[0].abstained is True
    assert len(gibberish[0].alternatives) == 3


def test_coverage_curve_is_monotone_in_coverage(tickets: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = split_dataset(tickets, strategy="stratified")
    model = TicketClassifier(model="logreg").fit(x_train, y_train)
    curve = coverage_accuracy_curve(model.predict_proba(x_test), model.classes_, y_test)
    coverages = [point["coverage"] for point in curve]
    assert coverages == sorted(coverages, reverse=True)


def test_top_features_are_available_for_linear_models(tickets: pd.DataFrame) -> None:
    x_train, _, y_train, _ = split_dataset(tickets, strategy="stratified")
    model = TicketClassifier(model="logreg").fit(x_train, y_train)
    features = model.top_features_for(str(model.classes_[0]), n=5)
    assert len(features) == 5


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("C'est la troisieme fois que j'ouvre ce ticket, delai inacceptable", "negative"),
        ("Merci beaucoup, tres bon support, pas urgent", "positive"),
        ("Installation de MATLAB sur mon poste", "neutral"),
        ("This is unacceptable, I am completely blocked and nobody called back", "negative"),
    ],
)
def test_lexicon_sentiment_labels(text: str, expected: str) -> None:
    assert LexiconSentiment().analyze([text])[0].label == expected


def test_sentiment_returns_evidence() -> None:
    result = LexiconSentiment().analyze(["Totalement bloque, relance sans reponse"])[0]
    assert result.evidence
    assert -1.0 <= result.score <= 1.0


def test_detector_finds_injected_anomalies() -> None:
    dataset = generate(n_tickets=9000, days=300, seed=5)
    counts = build_count_series(dataset.tickets, by="category")
    report = WeakSignalDetector().detect(counts)

    metrics = evaluate_detections(
        report,
        [a.as_dict() for a in dataset.anomalies],
        n_series_days=int(counts.shape[0] * counts.shape[1]),
    )
    assert metrics["recall"] >= 0.75
    assert metrics["mean_detection_delay_days"] is not None
    assert metrics["false_alarms_per_100_series_days"] < 5


def test_emerging_signal_fires_before_the_ramp_peaks() -> None:
    dataset = generate(n_tickets=9000, days=300, seed=5)
    ramp = next(a for a in dataset.anomalies if a.kind == "ramp")
    counts = build_count_series(dataset.tickets, by="category")
    report = WeakSignalDetector().detect(counts)

    alerts = [
        alert
        for alert in report.alerts
        if alert.series == ramp.category and ramp.start <= alert.period <= ramp.end
    ]
    assert alerts, "the slow ramp must be detected"
    first = min(alerts, key=lambda a: a.period)
    # Caught in the first half of the ramp, i.e. before it becomes obvious.
    assert (first.period - ramp.start).days <= (ramp.end - ramp.start).days / 2


def test_quiet_series_do_not_alert() -> None:
    dates = pd.date_range("2025-01-01", periods=200, freq="D")
    steady = pd.DataFrame({"stable": [10] * 200}, index=dates)
    report = WeakSignalDetector(DetectorConfig(min_history=28)).detect(steady)
    assert report.alerts == []
