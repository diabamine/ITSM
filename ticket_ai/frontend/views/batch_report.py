"""Page 2 - select a batch of tickets and get a report on its trend and alerts.

The page only collects the selection and shows the result. The analysis and the
rendering (HTML, PDF) live in :mod:`backend.report`, so the downloaded file and
the on-screen preview are produced by the very same code.
"""

from __future__ import annotations

from datetime import timedelta

import streamlit as st

from backend.report import build_report, pdf_available
from backend.schema import Col
from frontend import state
from frontend.components import show_html

FILTERS = {
    "Incident type": Col.INCIDENT_TYPE,
    "Service": Col.SERVICE,
    "Assignment group": Col.GROUP,
    "Site": Col.SITE,
    "Priority": Col.PRIORITY,
}


def render() -> None:
    """Draw the batch report page."""
    st.title("Batch report")
    st.caption(
        "Pick a period and, if you want, some types or services. The report shows the global trend "
        "and the alerts raised on that batch, and can be downloaded."
    )
    store = state.get_store()
    client = state.get_client()

    first, last = store.tickets[Col.OPENED].min().date(), store.tickets[Col.OPENED].max().date()
    with st.container(border=True):
        period = st.date_input(
            "Opening date range", value=(max(first, last - timedelta(days=120)), last), min_value=first, max_value=last
        )
        if len(period) != 2:
            st.info("Select an end date.")
            return
        selections: dict[str, list[str]] = {}
        columns = st.columns(len(FILTERS))
        for (label, column), container in zip(FILTERS.items(), columns, strict=True):
            selections[column] = container.multiselect(label, store.options(column), placeholder="All")

    batch = store.filter(period[0], period[1], selections)
    description = {"Period": f"{period[0]} to {period[1]}"} | {
        label: ", ".join(selections[column]) for label, column in FILTERS.items() if selections[column]
    }

    left, right = st.columns([1, 3])
    left.metric("Tickets in the selection", len(batch))
    title = right.text_input("Report title", value="Ticket batch report")
    use_llm = right.toggle(
        "Write the summary with the LLM (only aggregated figures are sent, never ticket text)",
        value=client is not None,
        disabled=client is None,
        help="Needs LLM_BASE_URL and LLM_MODEL. Without it, a template summary is used.",
    )

    if st.button("Generate report", type="primary", disabled=batch.empty):
        with st.spinner("Analysing the batch..."):
            report = build_report(batch, title, description, client if use_llm else None)
            st.session_state["report"] = report
            st.session_state["report_pdf"] = report.to_pdf() if pdf_available() else None
            st.session_state["report_selection"] = description
    if batch.empty:
        st.warning("No ticket matches this selection.")

    report = st.session_state.get("report")
    if report is None:
        return
    if st.session_state.get("report_selection") != description:
        st.warning("The selection changed since this report was generated.", icon=":material/refresh:")

    st.divider()
    html_col, pdf_col, _ = st.columns([1, 1, 3])
    name = f"ticket_report_{report.period.replace(' ', '').replace('-', '_')}"
    html_col.download_button("Download HTML", report.to_html(), f"{name}.html", "text/html", icon=":material/code:")
    pdf = st.session_state.get("report_pdf")
    if pdf:
        pdf_col.download_button("Download PDF", pdf, f"{name}.pdf", "application/pdf", icon=":material/picture_as_pdf:")
    else:
        pdf_col.button(
            "PDF unavailable",
            disabled=True,
            icon=":material/picture_as_pdf:",
            help="Install WeasyPrint (pip install weasyprint) to enable PDF export.",
        )
    show_html(report.to_html(), height=1100)
