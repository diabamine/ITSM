"""Page 3 - monitor how tickets are distributed and how they evolve, with a projection.

All numbers come from :mod:`backend.analytics`; this page chooses what to ask
for and draws the answer.
"""

from __future__ import annotations

from datetime import timedelta

import streamlit as st

from backend.analytics import change_vs_previous, distribution, kpis, project, volume_series
from backend.schema import Col
from frontend import charts, state

DIMENSIONS = {
    "Incident type": Col.INCIDENT_TYPE,
    "Service": Col.SERVICE,
    "Assignment group": Col.GROUP,
    "Affected system": Col.AFFECTED_SYSTEM,
    "Root cause": Col.ROOT_CAUSE,
    "Site": Col.SITE,
    "Entity": Col.ENTITY,
    "Tone": Col.TONE,
    "Priority": Col.PRIORITY,
}
HISTORY = {"Last 6 months": 183, "Last 12 months": 365, "Full history": None}
HORIZONS = {"Week": (8, 26), "Month": (3, 12)}  # (default, maximum) number of periods


def _fmt(value: float | None, suffix: str = "", digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def render() -> None:
    """Draw the monitoring dashboard."""
    st.title("Dashboard")
    st.caption("Volume and distribution over time, with a projection of where each series is heading.")
    store = state.get_store()

    with st.container(border=True):
        a, b, c = st.columns(3)
        granularity = a.radio("Granularity", ["Week", "Month"], horizontal=True)
        history = b.selectbox("History", list(HISTORY))
        dimension_label = c.selectbox("Break down by", list(DIMENSIONS))
        d, e, f = st.columns(3)
        top_n = d.slider("Series shown", 2, 8, 5)
        default_h, max_h = HORIZONS[granularity]
        horizon = e.slider("Projection horizon (periods)", 1, max_h, default_h, key=f"horizon-{granularity}")
        show_projection = f.toggle("Show projection", value=True)
    dimension = DIMENSIONS[dimension_label]

    end = store.reference_date
    days = HISTORY[history]
    start = end - timedelta(days=days) if days else store.tickets[Col.OPENED].min()
    tickets = store.filter(start, end)
    if tickets.empty:
        st.warning("No ticket in this window.")
        return

    indicators = kpis(tickets)
    change = change_vs_previous(store.tickets, start, end) if days else None
    cards = st.columns(5)
    cards[0].metric(
        "Tickets",
        f"{indicators['tickets']:,}".replace(",", " "),
        delta=None if change is None else f"{change:+.0f}% vs previous period",
        delta_color="inverse",
    )
    cards[1].metric("Open now", indicators["open"])
    cards[2].metric("Median resolution", _fmt(indicators["median_resolution_h"], " h", 1))
    cards[3].metric("SLA missed", _fmt(indicators["sla_breach_pct"], "%", 1))
    cards[4].metric("Negative tone", _fmt(indicators["negative_tone_pct"], "%", 1))

    st.subheader("Volume and projection")
    weekly = volume_series(tickets, granularity)
    done = weekly[weekly["complete"]]
    if len(done) < 4:
        st.info("Not enough complete periods to chart a trend. Choose a longer history or a finer granularity.")
        return
    projection = project(done.set_index("period")["tickets"], horizon, granularity) if show_projection else None
    st.plotly_chart(charts.volume_chart(done, projection), width="stretch")
    note = "Partial first and last periods are left out, so they do not distort the trend."
    if show_projection and len(done) < 8:
        note += " Fewer than 8 periods: the projection is flat. Widen the history for a trend."
    st.caption(note + " The shaded band is an approximate 95% range; it widens the further out you look.")

    st.subheader(f"By {dimension_label.lower()}")
    by_long = volume_series(tickets, granularity, by=dimension)
    by_long = by_long[by_long["complete"]]
    top = by_long.groupby(dimension)["tickets"].sum().nlargest(top_n).index
    by_long = by_long[by_long[dimension].isin(top)]
    projections = {}
    if show_projection:
        for value, group in by_long.groupby(dimension):
            projections[value] = project(group.set_index("period")["tickets"], horizon, granularity)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Evolution** (dotted = projection)")
        st.plotly_chart(charts.series_chart(by_long, dimension, projections), width="stretch")
    with right:
        st.markdown("**Distribution**")
        st.plotly_chart(charts.distribution_chart(distribution(tickets, dimension, top_n)), width="stretch")

    if projection is not None:
        with st.expander("Projected numbers"):
            table = projection.assign(period=projection["period"].dt.date).round(1)
            st.dataframe(
                table.rename(
                    columns={"period": "Period start", "forecast": "Projection", "lower": "Low", "upper": "High"}
                ),
                hide_index=True,
            )
