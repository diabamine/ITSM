"""Weak-signal detection on ticket arrival series.

Goal: flag a family whose arrival rate departs from its own trend *before* it
becomes an obvious outage - the "weak signal" that lets a team act while the
volume is still small.

Why not a plain z-score on a rolling mean? Three properties of ITSM arrival
series break it:

1. **Strong weekly seasonality.** Monday is not an anomaly. The baseline is
   therefore computed over the *same weekday* inside the trailing window.
2. **Counts, not Gaussians.** Daily volumes are small integers; the natural
   noise model is Poisson, whose variance equals its mean. The detector
   reports a Poisson tail probability alongside the robust z-score, so a jump
   from 2 to 8 tickets is judged on the right scale.
3. **Contamination.** The mean and standard deviation of a window that already
   contains the beginning of an incident are dragged towards it, and the
   detector blinds itself. Median and MAD are used instead - up to 50% of the
   window can be corrupted before the baseline moves.

Two alarm families are emitted:

``spike``
    Abrupt departure on a single day - an outage that already happened.
``emerging``
    An EWMA of standardised residuals crossing a threshold: several
    consecutive mild positive deviations, none of them individually
    significant. This is the actual weak signal, and it fires days before the
    spike does.

Everything here is evaluated rather than asserted: :func:`evaluate_detections`
scores the alerts against injected ground truth on detection rate, false-alarm
rate and, most importantly, **detection delay**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from itsm.logging import get_logger

logger = get_logger("models.weak_signal")


def build_count_series(
    tickets: pd.DataFrame,
    by: str | None = "category",
    freq: str = "D",
    timestamp_column: str = "opened_at",
) -> pd.DataFrame:
    """Turn the ticket table into regular count series, one column per group.

    Gaps are filled with zeros: a day without tickets is information, not a
    missing value.

    Args:
        tickets: Validated ticket frame.
        by: Column defining the series (``None`` yields a single ``all`` series).
        freq: Resampling frequency.
        timestamp_column: Timestamp used as the arrival date.

    Returns:
        Frame indexed by period, one integer column per group.
    """
    frame = tickets.copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column])
    frame["_period"] = frame[timestamp_column].dt.to_period(freq).dt.to_timestamp()
    key = by or "_all"
    if by is None:
        frame[key] = "all"

    pivot = frame.groupby(["_period", key]).size().unstack(fill_value=0).sort_index()
    full_index = pd.date_range(pivot.index.min(), pivot.index.max(), freq=freq)
    return pivot.reindex(full_index, fill_value=0).rename_axis("period")


@dataclass
class DetectorConfig:
    """Tuning of :class:`WeakSignalDetector`.

    Attributes:
        window_days: Length of the trailing baseline window.
        guard_days: Days skipped between the baseline window and the scored
            day. Without this gap a slow ramp walks into its own baseline and
            the detector learns the incident instead of flagging it; measured
            on the reference dataset, a 7-day guard cuts the detection delay of
            the ramp roughly in half.
        seasonal: Compare a day only with the same weekday of the window.
        min_history: Days of history required before scoring starts.
        z_spike: Robust z-score above which a single day is a ``spike``.
        poisson_alpha: Poisson tail probability required to confirm a spike.
        ewma_alpha: Smoothing factor of the residual EWMA (0.3 ~ 6-day memory).
        ewma_threshold: EWMA level above which an ``emerging`` signal fires.
        min_observed: Absolute count floor; avoids alerting on 1 vs 0 tickets.
    """

    window_days: int = 56
    guard_days: int = 7
    seasonal: bool = True
    min_history: int = 28
    z_spike: float = 3.5
    poisson_alpha: float = 1e-3
    ewma_alpha: float = 0.3
    ewma_threshold: float = 1.1
    min_observed: int = 4


@dataclass
class Alert:
    """One detection on one series and one period."""

    series: str
    period: date
    kind: str
    observed: int
    expected: float
    robust_z: float
    poisson_p: float
    ewma: float
    severity: float
    """Ranking score: ``max(robust_z, 3 * ewma)``, comparable across kinds."""


@dataclass
class DetectionReport:
    """Alerts plus the per-day diagnostic frame that produced them."""

    alerts: list[Alert] = field(default_factory=list)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_frame(self) -> pd.DataFrame:
        """Return the alerts as a dataframe, most severe first."""
        if not self.alerts:
            return pd.DataFrame(
                columns=[
                    "series",
                    "period",
                    "kind",
                    "observed",
                    "expected",
                    "robust_z",
                    "poisson_p",
                    "ewma",
                    "severity",
                ]
            )
        return pd.DataFrame([a.__dict__ for a in self.alerts]).sort_values(
            "severity", ascending=False
        )


class WeakSignalDetector:
    """Robust, seasonality-aware detector of volume anomalies.

    Args:
        config: Detector tuning; the defaults are documented in
            :class:`DetectorConfig`.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    def _baseline(self, history: pd.Series, weekday: int) -> tuple[float, float]:
        """Return (expected, scale) from the trailing window.

        The rate is the **maximum of the median and a 20% trimmed mean**. Both
        halves matter:

        * the median resists contamination - an incident already under way
          cannot raise its own baseline;
        * but on a low-volume series the median of small integers collapses to
          0 or 1, and a rate of 0 makes the Poisson tail test fire on any
          non-zero day. The trimmed mean keeps a usable rate there while still
          discarding the top of the window.

        ``scale`` is the MAD-derived standard deviation, floored by the Poisson
        standard deviation ``sqrt(expected)`` so a constant window cannot
        produce an infinite z-score.
        """
        subset = history
        if self.config.seasonal:
            same_weekday = history[history.index.dayofweek == weekday]
            if len(same_weekday) >= 4:
                subset = same_weekday

        values = subset.to_numpy(dtype=float)
        median = float(np.median(values))
        trimmed = float(stats.trim_mean(values, 0.2)) if len(values) >= 5 else float(values.mean())
        expected = max(median, trimmed)
        mad = float(np.median(np.abs(values - median)))
        scale = max(1.4826 * mad, np.sqrt(max(expected, 1.0)), 0.8)
        return expected, scale

    def score(self, series: pd.Series) -> pd.DataFrame:
        """Score one count series day by day.

        Returns:
            Frame indexed by period with ``observed``, ``expected``,
            ``robust_z``, ``poisson_p`` and ``ewma`` columns.
        """
        config = self.config
        records: list[dict[str, object]] = []
        ewma = 0.0

        for position in range(len(series)):
            period = series.index[position]
            observed = int(series.iloc[position])
            if position < config.min_history + config.guard_days:
                records.append(
                    {
                        "period": period,
                        "observed": observed,
                        "expected": np.nan,
                        "robust_z": 0.0,
                        "poisson_p": 1.0,
                        "ewma": 0.0,
                    }
                )
                continue

            stop = max(position - config.guard_days, 1)
            history = series.iloc[max(0, stop - config.window_days) : stop]
            expected, scale = self._baseline(history, int(period.dayofweek))
            robust_z = (observed - expected) / scale
            poisson_p = float(stats.poisson.sf(observed - 1, max(expected, 0.1)))

            # The EWMA only accumulates positive deviations: we are looking for
            # emerging pressure, and a quiet week must not build up credit that
            # masks the next rise.
            ewma = config.ewma_alpha * max(robust_z, 0.0) + (1 - config.ewma_alpha) * ewma
            records.append(
                {
                    "period": period,
                    "observed": observed,
                    "expected": round(expected, 2),
                    "robust_z": round(float(robust_z), 3),
                    "poisson_p": poisson_p,
                    "ewma": round(float(ewma), 3),
                }
            )

        return pd.DataFrame(records).set_index("period")

    def detect(self, counts: pd.DataFrame) -> DetectionReport:
        """Score every series of ``counts`` and collect the alerts.

        Args:
            counts: Output of :func:`build_count_series`.

        Returns:
            A :class:`DetectionReport`.
        """
        config = self.config
        alerts: list[Alert] = []
        diagnostics: list[pd.DataFrame] = []

        for name in counts.columns:
            scored = self.score(counts[name]).assign(series=str(name))
            diagnostics.append(scored.reset_index())

            for period, row in scored.iterrows():
                if row["observed"] < config.min_observed or pd.isna(row["expected"]):
                    continue
                is_spike = (
                    row["robust_z"] >= config.z_spike and row["poisson_p"] <= config.poisson_alpha
                )
                is_emerging = row["ewma"] >= config.ewma_threshold
                if not (is_spike or is_emerging):
                    continue
                alerts.append(
                    Alert(
                        series=str(name),
                        period=period.date(),
                        kind="spike" if is_spike else "emerging",
                        observed=int(row["observed"]),
                        expected=float(row["expected"]),
                        robust_z=float(row["robust_z"]),
                        poisson_p=float(row["poisson_p"]),
                        ewma=float(row["ewma"]),
                        severity=round(max(float(row["robust_z"]), 3 * float(row["ewma"])), 3),
                    )
                )

        logger.info("raised %d alerts over %d series", len(alerts), counts.shape[1])
        return DetectionReport(
            alerts=alerts,
            diagnostics=pd.concat(diagnostics, ignore_index=True)
            if diagnostics
            else pd.DataFrame(),
        )


def evaluate_detections(
    report: DetectionReport,
    truth: list[dict[str, object]],
    tolerance_days: int = 2,
    n_series_days: int | None = None,
) -> dict[str, object]:
    """Score a detection report against ground-truth anomaly windows.

    Metrics reported, and why:

    * ``recall`` - share of injected anomalies caught at all. A detector that
      misses the outage is worthless regardless of its precision.
    * ``precision`` / ``false_alarms_per_100_series_days`` - alerts outside any
      window. Precision alone is misleading here because the number of alerts
      depends on how long an anomaly lasts; the normalised false-alarm rate is
      what an on-call team experiences.
    * ``mean_detection_delay_days`` - days between the start of the anomaly and
      the first alert. Negative values are impossible; zero means same-day.
      This is the metric that separates a weak-signal detector from a
      post-mortem tool.

    Args:
        report: Output of :meth:`WeakSignalDetector.detect`.
        truth: Anomaly manifest, entries with ``category``, ``start``, ``end``
            and ``kind`` (as produced by the synthetic generator).
        tolerance_days: Grace period after the window end during which an alert
            is still counted as a true positive.
        n_series_days: Total number of scored series-days, used to normalise the
            false-alarm rate.

    Returns:
        A dictionary of metrics plus a per-anomaly detail list.
    """
    windows = [
        {
            "category": str(item["category"]),
            "start": pd.Timestamp(str(item["start"])).date(),
            "end": (pd.Timestamp(str(item["end"])) + pd.Timedelta(days=tolerance_days)).date(),
            "kind": str(item.get("kind", "unknown")),
        }
        for item in truth
    ]

    alerts = report.alerts
    matched_alerts: set[int] = set()
    details: list[dict[str, object]] = []

    for window in windows:
        hits = [
            (index, alert)
            for index, alert in enumerate(alerts)
            if alert.series == window["category"]
            and window["start"] <= alert.period <= window["end"]
        ]
        matched_alerts.update(index for index, _ in hits)
        first = min((alert for _, alert in hits), key=lambda a: a.period, default=None)
        details.append(
            {
                "category": window["category"],
                "kind": window["kind"],
                "start": window["start"].isoformat(),
                "detected": first is not None,
                "first_alert": first.period.isoformat() if first else None,
                "first_alert_kind": first.kind if first else None,
                "detection_delay_days": (first.period - window["start"]).days if first else None,
                "n_alerts": len(hits),
            }
        )

    detected = [d for d in details if d["detected"]]
    false_positives = len(alerts) - len(matched_alerts)
    delays = [value for item in detected if isinstance(value := item["detection_delay_days"], int)]

    precision = len(matched_alerts) / len(alerts) if alerts else 0.0
    recall = len(detected) / len(windows) if windows else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics: dict[str, object] = {
        "n_anomalies": len(windows),
        "n_alerts": len(alerts),
        "detected": len(detected),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "false_positive_alerts": false_positives,
        "mean_detection_delay_days": round(float(np.mean(delays)), 2) if delays else None,
        "max_detection_delay_days": max(delays) if delays else None,
        "per_anomaly": details,
    }
    if n_series_days:
        metrics["false_alarms_per_100_series_days"] = round(
            100 * false_positives / n_series_days, 3
        )
    return metrics


def threshold_sweep(
    counts: pd.DataFrame,
    truth: list[dict[str, object]],
    z_values: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0, 5.0),
    ewma_values: tuple[float, ...] = (0.7, 0.9, 1.1, 1.4, 1.8),
) -> pd.DataFrame:
    """Grid-search the two thresholds and report the operating points.

    There is no single best setting: the choice is a business trade-off between
    catching an incident a day earlier and waking someone up for nothing. The
    sweep makes that trade-off explicit instead of hiding it in a constant.
    """
    n_series_days = int(counts.shape[0] * counts.shape[1])
    rows: list[dict[str, object]] = []
    for z_spike in z_values:
        for ewma_threshold in ewma_values:
            detector = WeakSignalDetector(
                DetectorConfig(z_spike=z_spike, ewma_threshold=ewma_threshold)
            )
            metrics = evaluate_detections(
                detector.detect(counts), truth, n_series_days=n_series_days
            )
            rows.append(
                {
                    "z_spike": z_spike,
                    "ewma_threshold": ewma_threshold,
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "f1": metrics["f1"],
                    "mean_detection_delay_days": metrics["mean_detection_delay_days"],
                    "false_alarms_per_100_series_days": metrics.get(
                        "false_alarms_per_100_series_days"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["f1", "recall"], ascending=False).reset_index(drop=True)
