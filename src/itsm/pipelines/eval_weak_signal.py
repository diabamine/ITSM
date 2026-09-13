"""Evaluation pipeline for weak-signal detection.

Anomaly detection is usually shipped unevaluated ("look, it found the outage").
Here the synthetic generator injects a known set of spikes, ramps and level
shifts, so the detector can be scored the way a classifier is: detection rate,
false-alarm rate, and detection delay.

On a real extract without ground truth, the same pipeline runs in
``label_source="none"`` mode: it still produces the alert list and the
diagnostic series, and the thresholds are then set from the false-alarm budget
the team accepts (the sweep reports alarms per 100 series-days).
"""

from __future__ import annotations

from typing import Any

from itsm.config import Settings, get_settings
from itsm.data.loading import load_anomaly_manifest, load_tickets
from itsm.logging import get_logger
from itsm.models.weak_signal import (
    DetectorConfig,
    WeakSignalDetector,
    build_count_series,
    evaluate_detections,
    threshold_sweep,
)
from itsm.pipelines.reporting import write_report

logger = get_logger("pipelines.eval_weak_signal")


def run(
    settings: Settings | None = None,
    by: str = "category",
    freq: str = "D",
    config: DetectorConfig | None = None,
    with_sweep: bool = True,
    top_alerts: int = 25,
) -> dict[str, Any]:
    """Run the detector over the dataset and score it against ground truth.

    Args:
        settings: Configuration.
        by: Column defining the series (``category``, ``assignment_group``...).
        freq: Aggregation frequency of the count series.
        config: Detector tuning; defaults to :class:`DetectorConfig`.
        with_sweep: Also grid-search the two thresholds.
        top_alerts: Number of alerts included in the report.

    Returns:
        The evaluation report, also written to ``reports/weak_signal.json``.
    """
    settings = settings or get_settings()
    tickets = load_tickets(settings)
    counts = build_count_series(tickets, by=by, freq=freq)
    detector = WeakSignalDetector(config)
    detection = detector.detect(counts)

    n_series_days = int(counts.shape[0] * counts.shape[1])
    truth = load_anomaly_manifest(settings)

    report: dict[str, Any] = {
        "dataset": settings.dataset,
        "series_column": by,
        "frequency": freq,
        "n_series": int(counts.shape[1]),
        "n_periods": int(counts.shape[0]),
        "config": detector.config.__dict__,
        "n_alerts": len(detection.alerts),
        "alerts_by_kind": detection.to_frame()["kind"].value_counts().to_dict()
        if detection.alerts
        else {},
        "top_alerts": detection.to_frame().head(top_alerts).to_dict(orient="records"),
    }

    if truth:
        report["evaluation"] = evaluate_detections(detection, truth, n_series_days=n_series_days)
        logger.info(
            "recall=%s precision=%s mean delay=%s days",
            report["evaluation"]["recall"],
            report["evaluation"]["precision"],
            report["evaluation"]["mean_detection_delay_days"],
        )
        if with_sweep:
            report["threshold_sweep"] = threshold_sweep(counts, truth).to_dict(orient="records")
    else:
        report["evaluation"] = None
        report["note"] = (
            "No ground-truth manifest for this dataset: alerts are produced but not scored. "
            "Set thresholds from the accepted false-alarm budget."
        )

    write_report("weak_signal", report, settings)
    return report
