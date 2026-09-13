"""Descriptive statistics on tickets and on their status updates.

Two families of indicators, which answer different questions:

*Ticket-level* (from the ticket table): how much arrives, how long it takes,
how often the promise is missed. Durations are summarised by **median and
p90**, never by the mean - handling-time distributions are heavy-tailed and a
mean MTTR is dominated by a handful of month-long tickets.

*Flow-level* (from the event table): what the audit trail says about how work
actually moves - dwell time per state, transition probabilities, rework loops
(reassignment, reopen) and the arrival/departure balance that drives backlog.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from itsm.data.schema import TERMINAL_STATES, TicketState
from itsm.timeutils import period_freq

#: Age buckets (in days) used to age the open backlog.
AGING_BUCKETS = (1, 3, 7, 14, 30, 90)


def _percentile(series: pd.Series, q: float) -> float:
    values = series.dropna()
    return float(np.percentile(values, q)) if len(values) else float("nan")


def kpi_overview(
    tickets: pd.DataFrame, as_of: pd.Timestamp | None = None
) -> dict[str, int | float]:
    """Return the headline indicators of the whole extract.

    Args:
        tickets: Validated ticket frame.
        as_of: Reference instant for backlog and ageing; defaults to the last
            observed ``opened_at``.

    Returns:
        A flat dictionary of indicators, ready to serialise.
    """
    as_of = as_of or pd.to_datetime(tickets["opened_at"]).max()
    open_mask = ~tickets["state"].isin({s.value for s in TERMINAL_STATES})
    resolution = pd.to_numeric(tickets["resolution_hours"], errors="coerce")
    span_days = max((as_of - pd.to_datetime(tickets["opened_at"]).min()).days, 1)

    return {
        "tickets": len(tickets),
        "period_days": int(span_days),
        "tickets_per_day": round(len(tickets) / span_days, 2),
        "open_tickets": int(open_mask.sum()),
        "open_ratio": round(float(open_mask.mean()), 4),
        "mttr_median_hours": round(float(resolution.median()), 2),
        "mttr_p90_hours": round(_percentile(resolution, 90), 2),
        "sla_breach_rate": round(
            float(tickets["sla_breached"].fillna(False).astype(bool).mean()), 4
        ),
        "reopen_rate": round(float((tickets["reopen_count"].fillna(0) > 0).mean()), 4),
        "reassignment_mean": round(float(tickets["reassignment_count"].fillna(0).mean()), 3),
        "first_time_right_rate": round(
            float(
                (
                    (tickets["reassignment_count"].fillna(0) == 0)
                    & (tickets["reopen_count"].fillna(0) == 0)
                ).mean()
            ),
            4,
        ),
        "updates_per_ticket_mean": round(float(tickets["sys_mod_count"].fillna(0).mean()), 2),
    }


def kpis_by(tickets: pd.DataFrame, dimension: str = "category") -> pd.DataFrame:
    """Break the headline indicators down by any categorical column.

    Args:
        tickets: Validated ticket frame.
        dimension: Column to group by (``category``, ``assignment_group``,
            ``priority``, ``site``, ``channel``...).

    Returns:
        One row per group, sorted by decreasing volume.
    """
    if dimension not in tickets.columns:
        raise KeyError(f"unknown dimension {dimension!r}")

    frame = tickets.assign(
        _resolution=pd.to_numeric(tickets["resolution_hours"], errors="coerce"),
        _breached=tickets["sla_breached"].fillna(False).astype(bool),
        _reopened=tickets["reopen_count"].fillna(0) > 0,
        _open=~tickets["state"].isin({s.value for s in TERMINAL_STATES}),
    )
    grouped = frame.groupby(dimension, dropna=False)
    summary = pd.DataFrame(
        {
            "tickets": grouped.size(),
            "share": grouped.size() / len(frame),
            "mttr_median_hours": grouped["_resolution"].median(),
            "mttr_p90_hours": grouped["_resolution"].quantile(0.9),
            "sla_breach_rate": grouped["_breached"].mean(),
            "reopen_rate": grouped["_reopened"].mean(),
            "reassignment_mean": grouped["reassignment_count"].mean(),
            "open_tickets": grouped["_open"].sum(),
        }
    )
    return summary.sort_values("tickets", ascending=False).round(4).reset_index()


def volume_timeline(tickets: pd.DataFrame, freq: str = "D", by: str | None = None) -> pd.DataFrame:
    """Return the arrival volume per period, optionally split by a dimension.

    Args:
        tickets: Validated ticket frame.
        freq: Pandas offset alias (``D``, ``W``, ``ME``...).
        by: Optional column to split on.

    Returns:
        Long-format frame with columns ``period``, [``by``], ``tickets``.
    """
    frame = tickets.assign(
        period=pd.to_datetime(tickets["opened_at"]).dt.to_period(period_freq(freq))
    )
    keys = ["period"] + ([by] if by else [])
    out = frame.groupby(keys, dropna=False).size().rename("tickets").reset_index()
    out["period"] = out["period"].dt.to_timestamp()
    return out


def backlog_timeline(tickets: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Return created / resolved / net flow and the running open backlog.

    The backlog is the cumulative difference between arrivals and departures,
    which is the quantity a capacity plan actually tracks.
    """
    alias = period_freq(freq)
    opened = (
        pd.to_datetime(tickets["opened_at"]).dt.to_period(alias).value_counts().rename("created")
    )
    resolved = (
        pd.to_datetime(tickets["resolved_at"])
        .dropna()
        .dt.to_period(alias)
        .value_counts()
        .rename("resolved")
    )
    frame = pd.concat([opened, resolved], axis=1).fillna(0).sort_index()
    frame["net"] = frame["created"] - frame["resolved"]
    frame["open_backlog"] = frame["net"].cumsum()
    frame.index = frame.index.to_timestamp()
    return frame.reset_index(names="period").astype({"created": int, "resolved": int, "net": int})


def aging_backlog(tickets: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Bucket the currently open tickets by age.

    Ageing is the early-warning view of the backlog: a stable open count can
    hide a tail of tickets quietly growing older.
    """
    as_of = as_of or pd.to_datetime(tickets["opened_at"]).max()
    open_tickets = tickets[~tickets["state"].isin({s.value for s in TERMINAL_STATES})].copy()
    if open_tickets.empty:
        return pd.DataFrame(columns=["age_bucket", "tickets", "share"])

    age_days = (as_of - pd.to_datetime(open_tickets["opened_at"])).dt.total_seconds() / 86400.0
    edges = [0, *AGING_BUCKETS, float("inf")]
    labels = (
        [f"<{AGING_BUCKETS[0]}d"]
        + [f"{low}-{high}d" for low, high in pairwise(AGING_BUCKETS)]
        + [f">{AGING_BUCKETS[-1]}d"]
    )
    buckets = pd.cut(age_days, bins=edges, labels=labels, right=False)
    counts = buckets.value_counts().reindex(labels).fillna(0).astype(int)
    return pd.DataFrame(
        {
            "age_bucket": labels,
            "tickets": counts.to_numpy(),
            "share": (counts / counts.sum()).round(4).to_numpy(),
        }
    )


def state_transitions(events: pd.DataFrame) -> pd.DataFrame:
    """Return the empirical transition matrix of the status updates.

    Args:
        events: Validated event frame.

    Returns:
        Frame with ``from_state``, ``to_state``, ``transitions``,
        ``probability`` (row-normalised) and ``median_hours`` spent in
        ``from_state`` before the transition.
    """
    if events.empty:
        return pd.DataFrame(
            columns=["from_state", "to_state", "transitions", "probability", "median_hours"]
        )

    frame = events.sort_values(["ticket_id", "timestamp"]).copy()
    grouped = frame.groupby("ticket_id", sort=False)
    frame["to_state"] = grouped["state"].shift(-1)
    frame["dwell_hours"] = (
        grouped["timestamp"].shift(-1) - frame["timestamp"]
    ).dt.total_seconds() / 3600.0
    frame = frame.dropna(subset=["to_state"])

    summary = (
        frame.groupby(["state", "to_state"], dropna=False)
        .agg(transitions=("state", "size"), median_hours=("dwell_hours", "median"))
        .reset_index()
        .rename(columns={"state": "from_state"})
    )
    summary["probability"] = summary["transitions"] / summary.groupby("from_state")[
        "transitions"
    ].transform("sum")
    return summary.sort_values(["from_state", "transitions"], ascending=[True, False]).round(4)


def dwell_times(events: pd.DataFrame) -> pd.DataFrame:
    """Return median / p90 time spent in each state before the next update."""
    if events.empty:
        return pd.DataFrame(columns=["state", "visits", "median_hours", "p90_hours"])

    frame = events.sort_values(["ticket_id", "timestamp"]).copy()
    frame["dwell_hours"] = (
        frame.groupby("ticket_id", sort=False)["timestamp"].shift(-1) - frame["timestamp"]
    ).dt.total_seconds() / 3600.0
    summary = (
        frame.dropna(subset=["dwell_hours"])
        .groupby("state")
        .agg(
            visits=("dwell_hours", "size"),
            median_hours=("dwell_hours", "median"),
            p90_hours=("dwell_hours", lambda s: float(np.percentile(s, 90))),
        )
        .reset_index()
    )
    return summary.sort_values("visits", ascending=False).round(3)


def rework_profile(events: pd.DataFrame) -> dict[str, float]:
    """Quantify rework loops observed in the audit trail.

    Reassignments and reopens are the two loops that inflate handling time
    without adding value; they are the prime target of routing automation.
    """
    if events.empty:
        return {}
    per_ticket = events.groupby("ticket_id").agg(
        updates=("event_index", "size"),
        reassignments=("is_reassignment", "sum"),
        reopens=("is_reopen", "sum"),
    )
    return {
        "tickets": len(per_ticket),
        "updates_per_ticket_median": float(per_ticket["updates"].median()),
        "reassignment_rate": round(float((per_ticket["reassignments"] > 0).mean()), 4),
        "reassignments_per_ticket_mean": round(float(per_ticket["reassignments"].mean()), 3),
        "reopen_rate": round(float((per_ticket["reopens"] > 0).mean()), 4),
        "touched_more_than_5_times": round(float((per_ticket["updates"] > 5).mean()), 4),
    }


def followed_ticket(
    tickets: pd.DataFrame, events: pd.DataFrame, ticket_id: str
) -> dict[str, object]:
    """Return the full trace of a single ticket: attributes and status updates.

    This is the "followed ticket" view: the chronology an analyst reads when
    auditing why a specific ticket took as long as it did.
    """
    matches = tickets[tickets["ticket_id"] == ticket_id]
    if matches.empty:
        raise KeyError(f"unknown ticket {ticket_id!r}")
    ticket = matches.iloc[0]

    trace = events[events["ticket_id"] == ticket_id].sort_values("timestamp").copy()
    if not trace.empty:
        trace["dwell_hours"] = (
            trace["timestamp"].shift(-1) - trace["timestamp"]
        ).dt.total_seconds() / 3600.0

    return {
        "ticket": ticket.to_dict(),
        "n_updates": len(trace),
        "timeline": trace.to_dict(orient="records"),
        "states_visited": list(dict.fromkeys(trace["state"].tolist())) if not trace.empty else [],
        "is_open": bool(ticket["state"] not in {s.value for s in TERMINAL_STATES}),
        "current_state": str(ticket["state"] or TicketState.NEW.value),
    }
