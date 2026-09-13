"""Signals page: trends, spikes and emerging weak signals."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from itsm.app import charts, state


def render() -> None:
    """Draw the weak-signal page."""
    st.title("Trends and weak signals")
    st.caption(
        "Robust, seasonality-aware detection on ticket arrival series "
        "(`itsm.models.weak_signal`). `spike` = an outage that already happened; "
        "`emerging` = the slow build-up that fires days earlier."
    )

    controls = st.columns([2, 1, 2, 2])
    dimension = controls[0].selectbox(
        "Series",
        ["category", "assignment_group", "priority", "site"],
        help="Column defining one series.",
    )
    frequency = controls[1].radio("Step", ["D", "W"], horizontal=True)
    z_spike = controls[2].slider(
        "Spike threshold (robust z)",
        2.0,
        6.0,
        3.5,
        0.1,
        help="Higher = fewer, surer spike alerts.",
    )
    ewma_threshold = controls[3].slider(
        "Emerging threshold (EWMA)",
        0.4,
        2.5,
        1.1,
        0.1,
        help="Lower = earlier weak signals, more false alarms.",
    )

    alerts, diagnostics = state.scan(dimension, frequency, z_spike, ewma_threshold)
    counts = state.count_series(dimension, frequency)

    summary = st.columns(4)
    summary[0].metric("Series", counts.shape[1])
    summary[1].metric("Alerts", len(alerts))
    summary[2].metric("Spikes", int((alerts["kind"] == "spike").sum()) if len(alerts) else 0)
    summary[3].metric("Emerging", int((alerts["kind"] == "emerging").sum()) if len(alerts) else 0)

    st.subheader("Alerts")
    if not len(alerts):
        st.success("No series departs from its own baseline at these thresholds.")
    else:
        st.dataframe(
            alerts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "poisson_p": st.column_config.NumberColumn("Poisson p", format="%.2e"),
                "robust_z": st.column_config.NumberColumn("robust z", format="%.2f"),
                "severity": st.column_config.ProgressColumn(
                    "severity",
                    min_value=0.0,
                    max_value=float(max(alerts["severity"].max(), 1.0)),
                    format="%.1f",
                ),
            },
        )

    st.divider()
    st.subheader("Evidence behind an alert")
    default_index = 0
    if len(alerts):
        options = list(counts.columns)
        default_index = (
            options.index(alerts.iloc[0]["series"]) if alerts.iloc[0]["series"] in options else 0
        )
    series_name = st.selectbox("Series", list(counts.columns), index=default_index)

    series_diagnostics = diagnostics[diagnostics["series"] == series_name].copy()
    series_alerts = (
        alerts[alerts["series"] == series_name].copy() if len(alerts) else pd.DataFrame()
    )
    if len(series_alerts):
        series_alerts["period"] = pd.to_datetime(series_alerts["period"])

    st.altair_chart(
        charts.anomaly_chart(series_diagnostics, series_alerts), use_container_width=True
    )
    st.caption(
        "Dashed line: expected volume, the median of the same weekday over the trailing "
        "window. Robust to contamination, so the start of an incident does not raise its "
        "own baseline."
    )
    st.altair_chart(
        charts.residual_chart(series_diagnostics, ewma_threshold), use_container_width=True
    )

    manifest = state.anomaly_manifest()
    if not manifest:
        st.info(
            "This dataset carries no ground truth, so alerts cannot be scored. Set the "
            "thresholds from the false-alarm budget the team accepts."
        )
        return

    st.divider()
    st.subheader("Scored against ground truth")
    st.caption(
        "The curated dataset injects known anomalies, which turns detection into a "
        "measurable task - detection delay included."
    )

    from itsm.models.weak_signal import Alert as SignalAlert
    from itsm.models.weak_signal import DetectionReport, evaluate_detections

    report = DetectionReport(
        alerts=[
            SignalAlert(**{**row, "period": pd.Timestamp(row["period"]).date()})
            for row in alerts.to_dict(orient="records")
        ]
        if len(alerts)
        else [],
        diagnostics=diagnostics,
    )
    metrics = evaluate_detections(
        report, manifest, n_series_days=int(counts.shape[0] * counts.shape[1])
    )

    scores = st.columns(4)
    scores[0].metric(
        "Recall", f"{metrics['recall']:.0%}", help="Share of injected anomalies caught."
    )
    scores[1].metric("Precision", f"{metrics['precision']:.0%}")
    scores[2].metric(
        "Mean delay",
        f"{metrics['mean_detection_delay_days']} d"
        if metrics["mean_detection_delay_days"] is not None
        else "-",
        help=(
            "Days between the start of an anomaly and the first alert - the metric that "
            "separates early warning from post-mortem."
        ),
    )
    scores[3].metric(
        "False alarms / 100 series-days",
        str(metrics.get("false_alarms_per_100_series_days", "-")),
    )

    st.dataframe(
        pd.DataFrame(metrics["per_anomaly"]),
        use_container_width=True,
        hide_index=True,
    )
