"""All the computations: KPIs, volumes, projection, growth, alerts and similar tickets.

Plain functions over DataFrames. No Streamlit and no plotting, so everything here can
be tested, or reused from a notebook, on its own.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

from backend import data as d


class Alert(NamedTuple):
    """A flag raised on a ticket or a batch. ``level`` is also the name of the matching ``st.*`` box."""

    level: str  # "error" | "warning" | "info"
    title: str
    detail: str = ""


_SEVERITY = {"error": 0, "warning": 1, "info": 2}
FREQ = {"W": "W-MON", "M": "MS"}  # to extend a series of weeks / months into the future


# --- basics -----------------------------------------------------------------------------
def texts(df: pd.DataFrame) -> list[str]:
    """Text of each ticket as sent to ``ai.embed``: ``"<short description>. <description>"``."""
    return (df[d.TITLE].fillna("") + ". " + df[d.DESCRIPTION].fillna("")).tolist()


def _pct(flags: pd.Series) -> float | None:
    return float(flags.mean() * 100) if len(flags) else None


def kpis(df: pd.DataFrame) -> dict[str, float | int | None]:
    """Headline figures. A figure is ``None`` when its column is empty in the extract."""
    hours = df[d.RESOLUTION_H].dropna()
    return {
        "tickets": len(df),
        "open": int(df[d.RESOLVED].isna().sum()),
        "median_resolution_h": float(hours.median()) if len(hours) else None,
        "sla_missed_pct": _pct(df[d.SLA].dropna() == d.SLA_BREACHED),
        "reopened_pct": _pct(df[d.REOPENED].dropna() > 0),
        "negative_tone_pct": _pct(df[d.TONE].dropna().isin(d.NEGATIVE_TONES)),
    }


def volume(df: pd.DataFrame, freq: str = "W", by: str | None = None) -> pd.DataFrame:
    """Tickets per week (``"W"``) or month (``"M"``); empty periods count as 0.

    The result is indexed by period start, with one column ``Tickets`` or, with ``by``, one
    column per value. A first or last period that the data covers only partly is dropped,
    otherwise it would look like a collapse. Two days of slack are allowed because a week
    that ends on a quiet Friday is still a full week.
    """
    opened = df[d.OPENED]
    period = opened.dt.to_period(freq)
    index = pd.period_range(period.min(), period.max(), freq=freq)
    if by is None:
        counts = period.value_counts().reindex(index, fill_value=0).to_frame("Tickets")
    else:
        counts = pd.crosstab(period, df[by].fillna("n/a").astype(str)).reindex(index, fill_value=0)
        counts.columns.name = None
    slack = pd.Timedelta(days=2)
    complete = (index.start_time >= opened.min() - slack) & (index.end_time <= opened.max() + slack)
    counts, starts = counts[complete], index.start_time[complete]
    counts.index = starts
    return counts


# --- projection -------------------------------------------------------------------------
def _holt(y: np.ndarray, alpha: float, beta: float, phi: float) -> tuple[float, float, np.ndarray]:
    """One pass of damped Holt smoothing: final level, final trend and one-step errors."""
    level, trend, errors = y[0], float(np.diff(y[:4]).mean()), []
    for value in y[1:]:
        error = value - (level + phi * trend)
        errors.append(error)
        level, trend = level + phi * trend + alpha * error, phi * trend + alpha * beta * error
    return float(level), trend, np.array(errors)


def project(series: pd.Series, horizon: int, freq: str = "W", phi: float = 0.9) -> pd.DataFrame:
    """Project a series of counts ``horizon`` periods ahead, with an approximate 95% band.

    Damped Holt exponential smoothing: a level plus a trend that fades (``phi < 1``), so a recent
    rise does not extrapolate forever. The smoothing parameters minimise the one-step error and
    the band widens with the horizon. Seasonality is ignored on purpose: one or two years of
    weekly data are too few to estimate it reliably. Fewer than 8 points give a flat projection.

    Returns:
        ``forecast``, ``lower`` and ``upper``, indexed by the start of each future period.
    """
    y = series.to_numpy(float)
    future = pd.date_range(series.index[-1], periods=horizon + 1, freq=FREQ[freq])[1:]
    if len(y) < 8:
        mean = np.full(horizon, y[-4:].mean())
        sigma = np.full(horizon, y.std() if len(y) > 1 else 1.0)
    else:
        grid = [(a, b) for a in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9) for b in (0.05, 0.1, 0.2, 0.4)]
        alpha, beta = min(grid, key=lambda ab: np.sum(_holt(y, *ab, phi)[2] ** 2))
        level, trend, errors = _holt(y, alpha, beta, phi)
        steps = np.arange(1, horizon + 1)
        mean = level + trend * np.cumsum(phi**steps)
        c = alpha * (1 + beta * np.cumsum(phi ** steps[:-1]))  # standard ETS(A,Ad,N) variance
        sigma = errors.std(ddof=1) * np.sqrt(1 + np.concatenate([[0.0], np.cumsum(c**2)]))
    mean = np.maximum(mean, 0)
    return pd.DataFrame(
        {"forecast": mean, "lower": np.maximum(mean - 1.96 * sigma, 0), "upper": mean + 1.96 * sigma},
        index=future,
    )


# --- what is growing --------------------------------------------------------------------
def growth(df: pd.DataFrame, column: str, recent: int = 28, baseline: int = 84, min_recent: int = 10) -> pd.DataFrame:
    """Compare the last ``recent`` days with the ``baseline`` days before, for each value of ``column``.

    Columns: ``recent`` (tickets), ``was_per_day``, ``now_per_day``, ``ratio`` (NaN when the value did not
    exist before), ``share_of_increase`` (its part of the total rise) and ``p_value`` (chance of seeing at
    least this many recent tickets if the rate had not changed: small means it is not noise).
    Empty when the data is too short to define a baseline.
    """
    opened = df[d.OPENED]
    cut = opened.max() - pd.Timedelta(days=recent)
    start = max(opened.min(), cut - pd.Timedelta(days=baseline))
    span = (cut - start).days
    if span < 14:
        return pd.DataFrame()
    values = df[column].fillna("n/a").astype(str)
    now = values[opened > cut].value_counts()
    before = values[(opened > start) & (opened <= cut)].value_counts().reindex(now.index, fill_value=0)

    out = pd.DataFrame({"recent": now, "was_per_day": before / span, "now_per_day": now / recent})
    out["ratio"] = out["now_per_day"] / out["was_per_day"].replace(0, np.nan)
    rise = (out["now_per_day"] - out["was_per_day"]).clip(lower=0)
    out["share_of_increase"] = rise / rise.sum() if rise.sum() > 0 else 0.0
    out["p_value"] = poisson.sf(now - 1, np.maximum(out["was_per_day"] * recent, 0.1))
    return out[out["recent"] >= min_recent].sort_values("share_of_increase", ascending=False)


def batch_alerts(df: pd.DataFrame) -> list[Alert]:
    """Alerts on a batch: volume spike, sustained rise, and topics that are growing fast.

    A topic (incident type, system or root cause) is flagged when it grew at least 1.5x, holds 15% or
    more of the total increase, and is unlikely to be noise (p < 0.01). Batches under 30 tickets are
    too small to judge and return nothing. Most severe first.
    """
    alerts: list[Alert] = []
    if len(df) < 30:
        return alerts

    weekly = volume(df, "W")["Tickets"].to_numpy(float)
    if len(weekly) >= 6:
        last, past = weekly[-1], weekly[-9:-1]
        median = np.median(past)
        scale = max(1.4826 * np.median(np.abs(past - median)), np.sqrt(max(median, 1.0)))
        z = (last - median) / scale
        if z >= 2 and last >= 1.3 * median:
            alerts.append(
                Alert(
                    "error" if z >= 3 else "warning",
                    f"Volume spike: {last:.0f} tickets last week vs {median:.0f} typical",
                    f"{z:.1f} robust standard deviations above the previous weeks.",
                )
            )
        before = weekly[-12:-4]
        if len(before) >= 4 and weekly[-4:].mean() >= 1.2 * before.mean():
            alerts.append(
                Alert(
                    "warning",
                    "Sustained rise in volume",
                    f"{weekly[-4:].mean():.0f}/week over the last 4 weeks vs {before.mean():.0f} before.",
                )
            )

    for column, label in {d.INCIDENT_TYPE: "incident type", d.AFFECTED_SYSTEM: "system", d.ROOT_CAUSE: "root cause"}.items():
        g = growth(df, column)
        if g.empty:
            continue
        hot = g[(g["ratio"].fillna(np.inf) >= 1.5) & (g["p_value"] < 0.01) & (g["share_of_increase"] >= 0.15)]
        for name, row in hot.head(1).iterrows():
            ratio = "new" if pd.isna(row["ratio"]) else f"x{row['ratio']:.1f}"
            alerts.append(
                Alert(
                    "error" if row["share_of_increase"] >= 0.4 else "warning",
                    f"Growing {label}: {name} ({ratio})",
                    f"{row['recent']:.0f} tickets in the last 28 days, {row['share_of_increase']:.0%} of the total increase.",
                )
            )
    return sorted(alerts, key=lambda a: _SEVERITY[a.level])


# --- one new ticket ---------------------------------------------------------------------
CONTEXT = [d.SERVICE, d.GROUP, d.INCIDENT_TYPE, d.AFFECTED_SYSTEM, d.SITE]  # categories blended into the similarity


def nearest(
    df: pd.DataFrame, vectors: np.ndarray, query: np.ndarray, context: dict[str, str], k: int = 30, text_weight: float = 0.7
) -> pd.DataFrame:
    """The ``k`` past tickets closest to a new one (a plain k-NN, no index needed).

    ``similarity = text_weight x cosine(embeddings) + (1 - text_weight) x share of matching categories``.
    Both terms are in [0, 1], so a score reads directly: 0.8 means close wording and mostly the same
    context. Ties go to the most recent ticket.

    Args:
        df: The history; ``vectors[i]`` must be the embedding of ``df.iloc[i]``.
        vectors: ``(len(df), dimension)`` embeddings of the history.
        query: Embedding of the new ticket.
        context: Category values of the new ticket (form plus LLM fields), keyed by column name.

    Returns:
        ``df`` rows, best first, with a ``Similarity`` column.
    """

    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)

    text = np.clip(unit(np.asarray(vectors, float)) @ unit(np.asarray(query, float)), 0, 1)
    matches = [(df[c].astype(str) == str(context[c])).to_numpy() for c in CONTEXT if context.get(c)]
    category = np.sum(matches, axis=0) / len(CONTEXT) if matches else np.zeros(len(df))
    score = text_weight * text + (1 - text_weight) * category
    order = np.lexsort((-df[d.OPENED].astype("int64").to_numpy(), -score))[:k]
    return df.iloc[order].assign(Similarity=score[order])


def ticket_alerts(ticket: dict[str, str], fields: dict[str, str], similar: pd.DataFrame, now: pd.Timestamp) -> list[Alert]:
    """Alerts on a new ticket, from the form, the LLM fields and the similar tickets. Most severe first.

    They are plain rules, so each one can be read and tuned here. ``similar`` is the output of
    :func:`nearest` (empty is fine) and ``now`` is the date of the latest ticket in the history.
    """
    alerts: list[Alert] = []
    top = d.LEVELS[-1]
    if ticket.get(d.URGENCY) == top and ticket.get(d.IMPACT) == top:
        alerts.append(Alert("error", "Possible major incident", "Urgency and impact are both at the highest level."))
    if fields.get(d.INCIDENT_TYPE) == d.SECURITY:
        alerts.append(Alert("error", "Security-related ticket", "Route it to the security team."))
    if fields.get(d.SCOPE) in d.WIDE_SCOPES:
        alerts.append(Alert("warning", f"Wide impact: {fields[d.SCOPE]}", "The text suggests many users are affected."))
    if fields.get(d.TONE) in d.NEGATIVE_TONES:
        alerts.append(
            Alert(
                "warning",
                f"Escalation risk: tone is {str(fields[d.TONE]).lower()}",
                "Consider a proactive call-back before the SLA expires.",
            )
        )

    if len(similar):
        close = similar[(similar["Similarity"] >= 0.6) & (similar[d.OPENED] >= now - pd.Timedelta(days=14))]
        if len(close) >= 5:
            alerts.append(
                Alert(
                    "error" if len(close) >= 10 else "warning",
                    f"Emerging pattern: {len(close)} similar tickets in 14 days",
                    "Several recent tickets describe the same problem: it may be one incident, not many.",
                )
            )
        head = similar.head(10)
        reopened, missed = head[d.REOPENED].mean(), (head[d.SLA] == d.SLA_BREACHED).mean()
        if reopened >= 0.5 or missed >= 0.4:
            alerts.append(
                Alert(
                    "warning",
                    "Similar tickets were hard to resolve",
                    f"Among the 10 closest: {missed:.0%} missed their SLA, {reopened:.1f} reopenings on average.",
                )
            )
    return sorted(alerts, key=lambda a: _SEVERITY[a.level])
