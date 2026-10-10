"""Entry point. Run from the ``ticket_ai`` folder:  ``streamlit run frontend/app.py``.

Builds the navigation and the sidebar. Each page lives in :mod:`frontend.views`
and gets its data from :mod:`backend` through the cached resources in
:mod:`frontend.state`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts this script's folder on the path, not its parent: add the app
# root so that ``backend`` and ``frontend`` import as top-level packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from backend.schema import Col
from frontend import state
from frontend.components import llm_status
from frontend.views import batch_report, dashboard, ticket_form

st.set_page_config(page_title="Ticket AI", page_icon=":material/support_agent:", layout="wide")


def _sidebar() -> None:
    """Show what the app is running on: data, embeddings and LLM."""
    store = state.get_store()
    opened = store.tickets[Col.OPENED]
    with st.sidebar:
        st.subheader("Data")
        st.caption(
            f"**{store.source}**  \n{len(store.tickets):,} tickets  \n"
            f"{opened.min():%Y-%m-%d} to {opened.max():%Y-%m-%d}".replace(",", " ")
        )
        if not store.has_llm_columns:
            st.warning(
                "The LLM-inferred columns (IncidentType, RootCause...) are empty in this extract: "
                "pages that group by them will be limited.",
                icon=":material/warning:",
            )
        st.subheader("AI")
        llm_status(state.get_enricher().mode)


def main() -> None:
    """Declare the three pages and run the selected one."""
    _sidebar()
    navigation = st.navigation(
        [
            st.Page(
                ticket_form.render, title="New ticket", icon=":material/edit_note:", url_path="new-ticket", default=True
            ),
            st.Page(batch_report.render, title="Batch report", icon=":material/description:", url_path="report"),
            st.Page(dashboard.render, title="Dashboard", icon=":material/monitoring:", url_path="dashboard"),
        ]
    )
    navigation.run()


main()
