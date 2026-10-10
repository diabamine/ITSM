"""Streamlit app. Run from the ``ticket_ai`` folder:  ``streamlit run frontend/app.py``.

Reads the ticket extract (CSV or XLSX) stored in ``data/``. Three pages use it: a ticket form, a
batch report and a dashboard. This file only collects inputs and shows results: the
computations are in ``backend/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so that `backend` and `frontend` import

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from backend import ai
from backend import analytics as an
from backend import data as d
from frontend import render

REPORT_FILTERS = [d.INCIDENT_TYPE, d.SERVICE, d.GROUP, d.SITE]
DIMENSIONS = [d.INCIDENT_TYPE, d.SERVICE, d.GROUP, d.AFFECTED_SYSTEM, d.ROOT_CAUSE, d.SITE, d.ENTITY, d.TONE]
SIMILAR_COLUMNS = [d.ID, d.TITLE, d.OPENED, d.ROOT_CAUSE, d.RESOLUTION_SUMMARY, d.RESOLUTION_H, "Similarity"]


@st.cache_data(show_spinner="Reading the extract...")
def read(path: str, modified: float) -> pd.DataFrame:
    """Parse the extract once. ``modified`` is the file's date, so replacing the file reloads it."""
    return d.load(Path(path))


@st.cache_data(show_spinner="Embedding the history (once per extract)...")
def history_vectors(texts: tuple[str, ...]) -> np.ndarray:
    """Embeddings of every past ticket. Restart the app after changing ``ai.embed``."""
    return np.asarray(ai.embed(list(texts)))


def choose(box, label: str, df: pd.DataFrame, column: str, order: list[str] = ()) -> str:
    """A select box filled with the values found in the extract (first choice: empty)."""
    return box.selectbox(label, ["", *d.options(df, column, order)], format_func=lambda v: v or "-")


# --- page 1: a new ticket ---------------------------------------------------------------
def new_ticket(df: pd.DataFrame) -> None:
    """Fill a ticket like a requester; your LLM, the k-NN and the rules analyse it."""
    st.title("New ticket")
    st.caption("On submit: your LLM extracts fields, the closest past tickets are retrieved and alerts are raised.")
    with st.form("ticket"):
        title = st.text_input("Short description *")
        description = st.text_area("Description *", height=140)
        top, bottom = st.columns(4), st.columns(3)
        ticket = {
            d.SERVICE: choose(top[0], "Service", df, d.SERVICE),
            d.GROUP: choose(top[1], "Assignment group", df, d.GROUP),
            d.CONTACT: choose(top[2], "Contact type", df, d.CONTACT),
            d.SITE: choose(top[3], "Site", df, d.SITE),
            d.ENTITY: choose(bottom[0], "Entity", df, d.ENTITY),
            d.URGENCY: choose(bottom[1], "Urgency", df, d.URGENCY, d.LEVELS),
            d.IMPACT: choose(bottom[2], "Impact", df, d.IMPACT, d.LEVELS),
        }
        submitted = st.form_submit_button("Analyse ticket", type="primary", width="stretch")
    if not submitted:
        return
    if not (title.strip() and description.strip()):
        st.error("Short description and description are required.")
        return
    ticket |= {d.TITLE: title.strip(), d.DESCRIPTION: description.strip()}

    fields, similar = {}, df.iloc[0:0]
    try:
        fields = ai.extract_fields(ticket)
    except NotImplementedError as missing:
        st.info(str(missing), icon=":material/smart_toy:")
    try:
        query = ai.embed(an.texts(pd.DataFrame([ticket])))[0]
        similar = an.nearest(df, history_vectors(tuple(an.texts(df))), query, {**ticket, **fields})
    except NotImplementedError as missing:
        st.info(str(missing), icon=":material/smart_toy:")

    if fields:
        st.subheader("AI analysis")
        for box, (name, value) in zip(st.columns(len(fields)), fields.items(), strict=True):
            box.metric(name, str(value))
    st.subheader("Alerts")
    alerts = an.ticket_alerts(ticket, fields, similar, df[d.OPENED].max())
    for alert in alerts:
        getattr(st, alert.level)(f"**{alert.title}**  \n{alert.detail}")
    if not alerts:
        st.success("No alert raised for this ticket.")
    if not similar.empty:
        st.subheader("Similar past tickets")
        st.dataframe(
            similar.head(5)[SIMILAR_COLUMNS],
            hide_index=True,
            column_config={"Similarity": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")},
        )


# --- page 2: batch report ---------------------------------------------------------------
def batch_report(df: pd.DataFrame) -> None:
    """Select a batch of tickets and get a report (HTML, printable to PDF)."""
    st.title("Batch report")
    first, last = df[d.OPENED].min().date(), df[d.OPENED].max().date()
    period = st.date_input(
        "Opening date range", (max(first, last - pd.Timedelta(days=120)), last), min_value=first, max_value=last
    )
    if len(period) != 2:
        st.info("Select an end date.")
        return
    picks = {
        c: box.multiselect(c, d.options(df, c)) for c, box in zip(REPORT_FILTERS, st.columns(len(REPORT_FILTERS)), strict=True)
    }
    batch = d.select(df, *period, picks)

    st.metric("Tickets in the selection", len(batch))
    title = st.text_input("Report title", "Ticket batch report")
    if st.button("Generate report", type="primary", disabled=batch.empty):
        st.session_state["report"] = render.report_html(batch, title)
    if page := st.session_state.get("report"):
        st.download_button("Download HTML", page, "ticket_report.html", "text/html", icon=":material/download:")
        components.html(page, height=1000, scrolling=True)


# --- page 3: dashboard ------------------------------------------------------------------
def dashboard(df: pd.DataFrame) -> None:
    """Volume and distribution over time, with a projection."""
    st.title("Dashboard")
    kpi = an.kpis(df)
    for box, (label, value) in zip(
        st.columns(5),
        [
            ("Tickets", f"{kpi['tickets']:,}"),
            ("Open now", kpi["open"]),
            ("Median resolution", render.fmt(kpi["median_resolution_h"], " h", 1)),
            ("SLA missed", render.fmt(kpi["sla_missed_pct"], "%", 1)),
            ("Negative tone", render.fmt(kpi["negative_tone_pct"], "%", 1)),
        ],
        strict=True,
    ):
        box.metric(label, value)

    left, middle, right = st.columns(3)
    freq = {"Week": "W", "Month": "M"}[left.radio("Granularity", ["Week", "Month"], horizontal=True)]
    horizon = middle.slider("Projection horizon (periods)", 1, 26, 8)
    dimensions = [c for c in DIMENSIONS if df[c].notna().any()]
    by = right.selectbox("Break down by", dimensions) if dimensions else None

    total = an.volume(df, freq)["Tickets"]
    if len(total) < 4:
        st.info("Not enough complete periods to chart a trend.")
        return
    st.subheader("Volume and projection")
    st.plotly_chart(render.volume_fig(total, an.project(total, horizon, freq)), width="stretch")
    st.caption("Partial first and last periods are left out. The shaded band is an approximate 95% range.")

    if by:
        st.subheader(f"By {by}")
        wide = an.volume(df, freq, by)
        wide = wide[wide.sum().nlargest(5).index]
        projections = {c: an.project(wide[c], horizon, freq) for c in wide}
        evolution, distribution = st.columns([3, 2])
        evolution.plotly_chart(render.lines_fig(wide, projections), width="stretch")
        share = df[by].dropna().astype(str).value_counts(normalize=True).head(8)
        distribution.plotly_chart(render.bar_fig(share), width="stretch")


def main() -> None:
    """Load the extract stored in ``data/``, then show the three pages."""
    st.set_page_config(page_title="Ticket AI", page_icon=":material/support_agent:", layout="wide")
    path = d.find_extract()
    if path is None:
        st.title("Ticket AI")
        st.error(f"No extract found. Put your CSV or XLSX file in `{d.DATA_DIR}` and reload.", icon=":material/folder_off:")
        return
    try:
        df = read(str(path), path.stat().st_mtime)
    except ValueError as error:
        st.error(str(error))
        return

    st.sidebar.caption(f"**{path.name}**  \n{len(df):,} tickets, {df[d.OPENED].min():%Y-%m-%d} to {df[d.OPENED].max():%Y-%m-%d}")
    if df[d.INCIDENT_TYPE].isna().all():
        st.sidebar.warning("The LLM columns (IncidentType, RootCause...) are empty: some pages will be limited.")
    st.navigation(
        [
            st.Page(lambda: new_ticket(df), title="New ticket", icon=":material/edit_note:", url_path="new-ticket", default=True),
            st.Page(lambda: batch_report(df), title="Batch report", icon=":material/description:", url_path="report"),
            st.Page(lambda: dashboard(df), title="Dashboard", icon=":material/monitoring:", url_path="dashboard"),
        ]
    ).run()


if __name__ == "__main__":
    main()
