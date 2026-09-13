"""Analytics endpoints: lifecycle statistics, families and weak signals.

These endpoints read the ticket extract that the service was started with. In
production the dataframe dependencies would be replaced by a query against the
ITSM database or a warehouse view; nothing else in this module changes.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from itsm.analysis.lifecycle import (
    aging_backlog,
    backlog_timeline,
    dwell_times,
    followed_ticket,
    kpi_overview,
    kpis_by,
    rework_profile,
    state_transitions,
    volume_timeline,
)
from itsm.api.deps import ClustererDep, EventsDep, TicketsDep
from itsm.api.schemas import AlertOut, ClusterSummaryOut, SignalScanRequest, SignalScanResponse
from itsm.models.weak_signal import DetectorConfig, WeakSignalDetector, build_count_series

router = APIRouter(prefix="/v1", tags=["analytics"])


@router.get("/stats/overview", summary="Headline ticket indicators")
def stats_overview(tickets: TicketsDep) -> dict[str, int | float]:
    """Volume, backlog, MTTR, SLA, reopen and first-time-right rates."""
    return kpi_overview(tickets)


@router.get("/stats/by/{dimension}", summary="Indicators broken down by a dimension")
def stats_by(dimension: str, tickets: TicketsDep) -> list[dict[str, Any]]:
    """Break the indicators down by ``category``, ``priority``, ``site``..."""
    try:
        return kpis_by(tickets, dimension).to_dict(orient="records")
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.get("/stats/flow", summary="Arrivals, departures and backlog over time")
def stats_flow(
    tickets: TicketsDep,
    events: EventsDep,
    freq: str = Query(default="W", pattern="^(D|W|ME)$"),
) -> dict[str, Any]:
    """Return the flow view: volume, backlog, ageing and status-update stats."""
    payload: dict[str, Any] = {
        "volume": volume_timeline(tickets, freq=freq).to_dict(orient="records"),
        "backlog": backlog_timeline(tickets, freq=freq).to_dict(orient="records"),
        "aging": aging_backlog(tickets).to_dict(orient="records"),
    }
    if len(events):
        payload["status_updates"] = {
            "transitions": state_transitions(events).to_dict(orient="records"),
            "dwell_times": dwell_times(events).to_dict(orient="records"),
            "rework": rework_profile(events),
        }
    return payload


@router.get("/tickets/{ticket_id}/trace", summary="Follow one ticket and its status updates")
def ticket_trace(ticket_id: str, tickets: TicketsDep, events: EventsDep) -> dict[str, Any]:
    """Return the attributes and the full chronology of a single ticket."""
    try:
        return followed_ticket(tickets, events, ticket_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.get(
    "/clusters", response_model=list[ClusterSummaryOut], summary="Discovered ticket families"
)
def clusters(clusterer: ClustererDep) -> list[ClusterSummaryOut]:
    """Return the families found by the designation analysis."""
    return [ClusterSummaryOut(**summary.__dict__) for summary in clusterer.summaries_]


@router.post("/signals/scan", response_model=SignalScanResponse, summary="Scan for weak signals")
def scan_signals(request: SignalScanRequest, tickets: TicketsDep) -> SignalScanResponse:
    """Detect volume anomalies on the ticket arrival series.

    Returns both confirmed ``spike`` alerts and ``emerging`` weak signals - the
    slow build-ups that fire before a spike becomes visible.
    """
    if request.by not in tickets.columns:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown column {request.by!r}"
        )
    counts = build_count_series(tickets, by=request.by, freq=request.freq)
    detector = WeakSignalDetector(
        DetectorConfig(z_spike=request.z_spike, ewma_threshold=request.ewma_threshold)
    )
    report = detector.detect(counts)
    alerts = [
        AlertOut(**alert.__dict__)
        for alert in report.alerts
        if request.since is None or alert.period >= request.since
    ]
    return SignalScanResponse(
        n_series=int(counts.shape[1]),
        n_periods=int(counts.shape[0]),
        alerts=sorted(alerts, key=lambda a: a.severity, reverse=True),
    )


@router.get("/signals/series", summary="Diagnostic series behind the alerts")
def signal_series(
    tickets: TicketsDep,
    series: str = Query(description="Value of the grouping column, e.g. 'Network & VPN'."),
    by: str = "category",
    freq: str = Query(default="D", pattern="^(D|W)$"),
) -> list[dict[str, Any]]:
    """Return observed vs expected counts and scores, day by day.

    This is the evidence behind an alert: without it, an anomaly score is not
    actionable.
    """
    counts = build_count_series(tickets, by=by, freq=freq)
    if series not in counts.columns:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown series {series!r}; available: {list(counts.columns)[:20]}",
        )
    scored = WeakSignalDetector().score(counts[series]).reset_index()
    scored["period"] = scored["period"].dt.date.astype(str)
    return scored.to_dict(orient="records")
