"""Trends, distributions, projections and batch-level alerts.

Pure functions over DataFrames, no Streamlit. The dashboard and the report call
exactly the same code, so the number on screen and the number in the PDF cannot
disagree.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from backend.alerts import LEVEL_ORDER, Alert
from backend.schema import NEGATIVE_TONES, Col

FREQ = {"Day": "D", "Week": "W", "Month": "M"}
#: First day of the next period, per granularity, used to extend the time axis.
FUTURE_FREQ = {"Day": "D", "Week": "W-MON", "Month": "MS"}
OPEN_STAGES = ("Nouveau", "Affecté", "En cours", "En attente utilisateur")


# --------------------------------------------------------------------------------------
# Volumes
# --------------------------------------------------------------------------------------
def volume_series(tickets: pd.DataFrame, granularity: str = "Week", by: str | None = None) -> pd.DataFrame:
    """Count tickets per period, with empty periods filled with zero.

    A period with no ticket is information, not a missing value: leaving the gap
    would distort every chart and every projection built on the series.

    Args:
        tickets: Tickets to count.
        granularity: ``"Day"``, ``"Week"`` or ``"Month"``.
        by: Optional column to split the count by.

    Returns:
        Long frame: ``period`` (start of the period), ``tickets``, ``complete``
        (False for a first or last period the data only partly covers), and the
        ``by`` column when one is given.
    """
    freq = FREQ[granularity]
    periods = tickets[Col.OPENED].dt.to_period(freq)
    if periods.empty:
        return pd.DataFrame(columns=["period", "tickets", "complete"])
    full = pd.period_range(periods.min(), periods.max(), freq=freq)

    if by is None:
        counts = periods.value_counts().reindex(full, fill_value=0).sort_index()
        frame = pd.DataFrame({"period": full.start_time, "tickets": counts.to_numpy()})
    else:
        table = pd.crosstab(periods, tickets[by].astype(str)).reindex(full, fill_value=0)
        table.index = full.start_time
        frame = table.rename_axis("period").reset_index().melt(id_vars="period", var_name=by, value_name="tickets")
    end_of = dict(zip(full.start_time, full.end_time, strict=True))
    frame["complete"] = frame["period"].map(end_of) <= tickets[Col.OPENED].max()
    # A batch that starts mid-period has a partial first period too.
    slack = {"D": pd.Timedelta(days=1), "W": pd.Timedelta(days=1), "M": pd.Timedelta(days=3)}[freq]
    if tickets[Col.OPENED].min() - full[0].start_time > slack:
        frame.loc[frame["period"] == full[0].start_time, "complete"] = False
    return frame.sort_values("period").reset_index(drop=True)


def distribution(tickets: pd.DataFrame, column: str, top_n: int = 8) -> pd.DataFrame:
    """Share of tickets per value of ``column``, with the tail grouped as ``Autres``."""
    counts = tickets[column].fillna("Non renseigné").astype(str).value_counts()
    head = counts.head(top_n)
    if len(counts) > top_n:
        head = pd.concat([head, pd.Series({"Autres": counts.iloc[top_n:].sum()})])
    return pd.DataFrame({column: head.index, "tickets": head.to_numpy(), "share": head.to_numpy() / counts.sum()})


def kpis(tickets: pd.DataFrame) -> dict[str, float | int | None]:
    """Headline indicators of a set of tickets. Undefined ones are ``None``."""
    resolved = tickets[tickets[Col.SLA].isin(["Respecté", "Dépassé"])]
    tones = tickets[Col.TONE].dropna()
    resolution = pd.to_numeric(tickets[Col.RESOLUTION_HOURS], errors="coerce").dropna()
    reopen = pd.to_numeric(tickets[Col.REOPEN_COUNT], errors="coerce").dropna()
    return {
        "tickets": len(tickets),
        "open": int(tickets[Col.STAGE].isin(OPEN_STAGES).sum()),
        "median_resolution_h": float(resolution.median()) if len(resolution) else None,
        "sla_breach_pct": float((resolved[Col.SLA] == "Dépassé").mean() * 100) if len(resolved) else None,
        "reopen_pct": float((reopen > 0).mean() * 100) if len(reopen) else None,
        "negative_tone_pct": float(tones.isin(NEGATIVE_TONES).mean() * 100) if len(tones) else None,
    }


def change_vs_previous(all_tickets: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    """Percent change in volume against the equally long period just before.

    Returns ``None`` when the previous period is empty (no baseline to compare to).
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    length = (end - start) + pd.Timedelta(days=1)
    opened = all_tickets[Col.OPENED]
    current = int(((opened >= start) & (opened < end + pd.Timedelta(days=1))).sum())
    previous = int(((opened >= start - length) & (opened < start)).sum())
    return (current - previous) / previous * 100 if previous else None


# --------------------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------------------
def project(series: pd.Series, horizon: int, granularity: str = "Week", phi: float = 0.9) -> pd.DataFrame:
    """Project a count series forward with damped Holt exponential smoothing.

    The model keeps a *level* and a *trend* and damps the trend (``phi < 1``) so a
    recent rise does not extrapolate to infinity. Smoothing parameters are
    chosen by minimising the one-step-ahead error. The band is the usual
    approximate prediction interval: it widens with the horizon.

    It ignores seasonality on purpose: with weekly or monthly counts over a year or
    two there is too little data to estimate it reliably, and a model that claims
    more than it knows is worse than a plain one. Read the band, not just the line.

    Args:
        series: Counts indexed by period start. Must contain only complete periods.
        horizon: Number of periods to project.
        granularity: ``"Day"``, ``"Week"`` or ``"Month"``; sets the future dates.
        phi: Damping factor in ``(0, 1]``.

    Returns:
        ``period``, ``forecast``, ``lower``, ``upper`` for each future period.
    """
    y = series.to_numpy(dtype=float)
    future = pd.date_range(series.index[-1], periods=horizon + 1, freq=FUTURE_FREQ[granularity])[1:]

    if len(y) < 8:  # too short to fit a trend: a flat projection with a wide band
        mean = np.full(horizon, float(np.mean(y[-4:])))
        sigma = np.full(horizon, float(np.std(y)) if len(y) > 1 else 1.0)
    else:
        alpha, beta = _fit_holt(y, phi)
        level, trend, residuals = _run_holt(y, alpha, beta, phi)
        mean = level + trend * np.cumsum(phi ** np.arange(1, horizon + 1))
        # ETS(A,Ad,N) prediction variance: sigma^2 * (1 + sum_{j<h} c_j^2),
        # with c_j = alpha * (1 + beta * sum_{i<=j} phi^i).
        c = alpha * (1 + beta * np.cumsum(phi ** np.arange(1, horizon)))
        variance = 1 + np.concatenate([[0.0], np.cumsum(c**2)])
        sigma = float(np.std(residuals, ddof=1)) * np.sqrt(variance)

    mean = np.maximum(mean, 0)
    return pd.DataFrame(
        {
            "period": future,
            "forecast": mean,
            "lower": np.maximum(mean - 1.96 * sigma, 0),
            "upper": mean + 1.96 * sigma,
        }
    )


def _run_holt(y: np.ndarray, alpha: float, beta: float, phi: float) -> tuple[float, float, np.ndarray]:
    """One pass of damped Holt in error-correction form; returns level, trend, residuals."""
    level, trend = y[0], float(np.mean(np.diff(y[:4])))
    residuals = []
    for value in y[1:]:
        error = value - (level + phi * trend)
        residuals.append(error)
        level = level + phi * trend + alpha * error
        trend = phi * trend + alpha * beta * error
    return float(level), float(trend), np.array(residuals)


def _fit_holt(y: np.ndarray, phi: float) -> tuple[float, float]:
    """Pick the (alpha, beta) pair with the smallest one-step squared error."""
    best, best_sse = (0.3, 0.1), np.inf
    for alpha in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        for beta in (0.05, 0.1, 0.2, 0.4):
            sse = float(np.sum(_run_holt(y, alpha, beta, phi)[2] ** 2))
            if sse < best_sse:
                best, best_sse = (alpha, beta), sse
    return best


# --------------------------------------------------------------------------------------
# What is growing
# --------------------------------------------------------------------------------------
def growth_by(
    tickets: pd.DataFrame, column: str, recent_days: int = 28, baseline_days: int = 84, min_recent: int = 10
) -> pd.DataFrame:
    """Compare the recent rate of each value of ``column`` with its baseline rate.

    The "now" is the latest ticket of the batch. For each value:

    * ``ratio``: recent rate / baseline rate (``None`` if it did not exist before);
    * ``share_of_increase``: its part of the total *rise* across all values;
    * ``p_value``: probability of seeing at least this many recent tickets if the
      rate had not changed (Poisson), i.e. how unlikely it is to be noise.

    Returns an empty frame when the batch is too short to define a baseline.
    """
    columns = [column, "recent", "baseline_per_day", "recent_per_day", "ratio", "share_of_increase", "p_value"]
    opened = tickets[Col.OPENED]
    now = opened.max()
    recent_start = now - pd.Timedelta(days=recent_days)
    baseline_start = max(opened.min(), recent_start - pd.Timedelta(days=baseline_days))
    baseline_span = (recent_start - baseline_start).days
    if baseline_span < 14:
        return pd.DataFrame(columns=columns)

    values = tickets[column].fillna("Non renseigné").astype(str)
    recent = values[opened > recent_start].value_counts()
    baseline = values[(opened > baseline_start) & (opened <= recent_start)].value_counts()

    rows = []
    for value in recent.index:
        n_recent, n_base = int(recent[value]), int(baseline.get(value, 0))
        recent_rate, base_rate = n_recent / recent_days, n_base / baseline_span
        rows.append(
            {
                column: value,
                "recent": n_recent,
                "baseline_per_day": base_rate,
                "recent_per_day": recent_rate,
                "ratio": recent_rate / base_rate if base_rate > 0 else None,
                "delta": recent_rate - base_rate,
                "p_value": float(stats.poisson.sf(n_recent - 1, max(base_rate * recent_days, 0.1))),
            }
        )
    result = pd.DataFrame(rows)
    rise = result["delta"].clip(lower=0).sum()
    result["share_of_increase"] = result["delta"].clip(lower=0) / rise if rise > 0 else 0.0
    result = result[result["recent"] >= min_recent].drop(columns="delta")
    return result.sort_values("share_of_increase", ascending=False).reset_index(drop=True)[columns]


# --------------------------------------------------------------------------------------
# Batch alerts
# --------------------------------------------------------------------------------------
TOPIC_COLUMNS = {Col.INCIDENT_TYPE: "incident type", Col.AFFECTED_SYSTEM: "system", Col.ROOT_CAUSE: "root cause"}


def batch_alerts(tickets: pd.DataFrame) -> list[Alert]:
    """Raise the alerts that apply to a batch of tickets.

    * a volume spike against the recent weekly norm (robust to past spikes);
    * a sustained rise over the last four weeks;
    * topics (type, system, root cause) growing fast and unlikely to be noise;
    * SLA breaches and negative tone clearly above the batch's own average.

    Returns:
        Alerts sorted from most to least severe. Empty for a batch too small to judge.
    """
    alerts: list[Alert] = []
    if len(tickets) < 30:
        return alerts

    weekly = volume_series(tickets, "Week")
    full = weekly[weekly["complete"]]["tickets"].to_numpy(dtype=float)
    if len(full) >= 6:
        last, history = full[-1], full[-9:-1]
        median = float(np.median(history))
        scale = max(1.4826 * float(np.median(np.abs(history - median))), np.sqrt(max(median, 1.0)))
        z = (last - median) / scale
        if z >= 2 and last >= 1.3 * median:
            alerts.append(
                Alert(
                    "critical" if z >= 3 else "warning",
                    "VOLUME_SPIKE",
                    f"Volume spike: {last:.0f} tickets last week vs {median:.0f} typical",
                    f"{z:.1f} robust standard deviations above the previous weeks.",
                )
            )
        recent4, before = float(np.mean(full[-4:])), full[-12:-4]
        if len(before) >= 4 and recent4 >= 1.2 * float(np.mean(before)):
            alerts.append(
                Alert(
                    "warning",
                    "SUSTAINED_RISE",
                    "Sustained rise in volume",
                    f"{recent4:.0f} tickets/week over the last 4 weeks vs {np.mean(before):.0f} before.",
                )
            )

    for column, label in TOPIC_COLUMNS.items():
        growth = growth_by(tickets, column)
        hot = growth[
            (growth["ratio"].fillna(np.inf) >= 1.5) & (growth["p_value"] < 0.01) & (growth["share_of_increase"] >= 0.15)
        ]
        for _, row in hot.head(1).iterrows():
            ratio = "new" if pd.isna(row["ratio"]) else f"x{row['ratio']:.1f}"
            alerts.append(
                Alert(
                    "critical" if row["share_of_increase"] >= 0.4 else "warning",
                    "GROWING_TOPIC",
                    f"Growing {label}: {row[column]} ({ratio})",
                    f"{int(row['recent'])} tickets in the last 28 days, "
                    f"{row['share_of_increase']:.0%} of the total increase.",
                )
            )

    alerts += _drift_alert(tickets, "SLA_DRIFT", "SLA breaches", _breach_share)
    alerts += _drift_alert(tickets, "TONE_DRIFT", "Negative tone", _negative_share)
    return sorted(alerts, key=lambda a: LEVEL_ORDER[a.level])


def _breach_share(frame: pd.DataFrame) -> tuple[float, int]:
    done = frame[frame[Col.SLA].isin(["Respecté", "Dépassé"])]
    return (float((done[Col.SLA] == "Dépassé").mean()) if len(done) else 0.0), len(done)


def _negative_share(frame: pd.DataFrame) -> tuple[float, int]:
    tones = frame[Col.TONE].dropna()
    return (float(tones.isin(NEGATIVE_TONES).mean()) if len(tones) else 0.0), len(tones)


def _drift_alert(
    tickets: pd.DataFrame, code: str, label: str, share: Callable[[pd.DataFrame], tuple[float, int]]
) -> list[Alert]:
    """Alert when the last 28 days of a rate are well above the rest of the batch."""
    cutoff = tickets[Col.OPENED].max() - pd.Timedelta(days=28)
    recent, earlier = tickets[tickets[Col.OPENED] > cutoff], tickets[tickets[Col.OPENED] <= cutoff]
    (recent_rate, n_recent), (earlier_rate, n_earlier) = share(recent), share(earlier)
    if n_recent >= 30 and n_earlier >= 30 and recent_rate >= earlier_rate + 0.08 and recent_rate >= 1.3 * earlier_rate:
        return [
            Alert(
                "warning",
                code,
                f"{label} rising: {recent_rate:.0%} vs {earlier_rate:.0%} before",
                "Last 28 days compared with the rest of the selected period.",
            )
        ]
    return []
