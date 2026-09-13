"""Overview page: volume, backlog and status-update statistics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from itsm.analysis.lifecycle import (
    aging_backlog,
    backlog_timeline,
    dwell_times,
    kpi_overview,
    kpis_by,
    rework_profile,
    state_transitions,
    volume_timeline,
)
from itsm.app import charts, state


def render() -> None:
    """Draw the overview page."""
    st.title("Service desk overview")
    st.caption(
        "Descriptive statistics computed by `itsm.analysis.lifecycle` - the same "
        "functions the CLI and the API call."
    )

    tickets = state.tickets()
    events = state.events()
    overview = kpi_overview(tickets)

    first, second, third, fourth = st.columns(4)
    first.metric("Tickets", f"{overview['tickets']:,}".replace(",", " "))
    first.metric("Per day", overview["tickets_per_day"])
    second.metric("Open backlog", f"{overview['open_tickets']:,}".replace(",", " "))
    second.metric("Open ratio", f"{overview['open_ratio']:.1%}")
    third.metric("MTTR median", f"{overview['mttr_median_hours']:.1f} h")
    third.metric("MTTR p90", f"{overview['mttr_p90_hours']:.1f} h")
    fourth.metric("SLA breach", f"{overview['sla_breach_rate']:.1%}")
    fourth.metric(
        "First time right",
        f"{overview['first_time_right_rate']:.1%}",
        help="Neither reassigned nor reopened - the best single summary of triage quality.",
    )

    st.divider()
    left, right = st.columns([3, 1])
    frequency = right.radio("Granularity", ["D", "W", "ME"], index=1, horizontal=True)
    split = right.toggle("Split by category", value=False)

    volume = volume_timeline(tickets, freq=frequency, by="category" if split else None)
    left.altair_chart(
        charts.volume_line(volume, color="category" if split else None),
        use_container_width=True,
    )

    st.subheader("Flow and backlog")
    st.caption(
        "Capacity is driven by the difference between arrivals and departures, not by volume."
    )
    st.altair_chart(
        charts.backlog_area(backlog_timeline(tickets, freq=frequency)), use_container_width=True
    )

    ageing_column, dimension_column = st.columns([1, 2])
    with ageing_column:
        st.subheader("Backlog ageing")
        ageing = aging_backlog(tickets)
        st.altair_chart(
            charts.category_bars(ageing, value="tickets", label="age_bucket"),
            use_container_width=True,
        )
    with dimension_column:
        st.subheader("Indicators by dimension")
        dimension = st.selectbox(
            "Break down by", ["category", "priority", "assignment_group", "site", "channel"]
        )
        st.dataframe(
            kpis_by(tickets, dimension),
            use_container_width=True,
            hide_index=True,
            column_config={
                "share": st.column_config.ProgressColumn(
                    "share", min_value=0.0, max_value=1.0, format="%.1f%%"
                ),
                "sla_breach_rate": st.column_config.NumberColumn("SLA breach", format="%.1f%%"),
                "mttr_median_hours": st.column_config.NumberColumn(
                    "MTTR median (h)", format="%.1f"
                ),
            },
        )

    if not len(events):
        st.info("This dataset has no audit trail: status-update statistics are unavailable.")
        return

    st.divider()
    st.subheader("Status updates")
    st.caption(
        "The audit trail is the only place where *how* work moves is visible: dwell "
        "times, rework loops and the reopen probability."
    )
    rework = rework_profile(events)
    columns = st.columns(4)
    columns[0].metric("Updates per ticket", rework["updates_per_ticket_median"])
    columns[1].metric("Reassigned", f"{rework['reassignment_rate']:.1%}")
    columns[2].metric("Reopened", f"{rework['reopen_rate']:.1%}")
    columns[3].metric("Touched > 5 times", f"{rework['touched_more_than_5_times']:.1%}")

    transitions_column, dwell_column = st.columns(2)
    with transitions_column:
        st.markdown("**Transition matrix**")
        st.dataframe(
            state_transitions(events), use_container_width=True, hide_index=True, height=320
        )
    with dwell_column:
        st.markdown("**Time spent per state**")
        st.dataframe(dwell_times(events), use_container_width=True, hide_index=True, height=320)

    with st.expander("Follow one ticket"):
        ticket_id = st.selectbox("Ticket", tickets["ticket_id"].tail(200).tolist()[::-1])
        from itsm.analysis.lifecycle import followed_ticket

        trace = followed_ticket(tickets, events, str(ticket_id))
        st.write(
            f"**{trace['current_state']}** - {trace['n_updates']} updates - "
            f"{'open' if trace['is_open'] else 'closed'}"
        )
        st.dataframe(pd.DataFrame(trace["timeline"]), use_container_width=True, hide_index=True)
