"""Streamlit entrypoint.

Run with ``itsm app`` (or ``streamlit run src/itsm/app/main.py``).

The app is a **client of the library**, exactly like the CLI and the API: no
analysis code lives here, only presentation and caching. Pages are declared
with ``st.navigation`` rather than a magic ``pages/`` directory so the file
layout stays a normal Python package that can be imported and linted.
"""

from __future__ import annotations

import streamlit as st

from itsm.app import state
from itsm.app.views import embedding_map, families, overview, signals, triage
from itsm.logging import setup_logging

st.set_page_config(
    page_title="ITSM Insights",
    page_icon=":material/support_agent:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _sidebar() -> None:
    """Render the dataset / model status panel."""
    settings = state.settings()
    with st.sidebar:
        st.subheader("Dataset")
        st.code(settings.dataset, language=None)
        tickets = state.tickets()
        st.caption(
            f"{len(tickets):,} tickets - {tickets['opened_at'].min():%Y-%m-%d} to "
            f"{tickets['opened_at'].max():%Y-%m-%d}".replace(",", " ")
        )

        st.subheader("Models")
        for name, ready in state.artifacts_ready().items():
            st.write(f"{':white_check_mark:' if ready else ':x:'} `{name}`")
        metadata = state.store().load_metadata("classifier")
        if metadata.get("metrics"):
            st.caption(f"classifier macro-F1: {metadata['metrics'].get('macro_f1', '-')}")

        st.divider()
        st.caption(
            "Backends: "
            f"text `{settings.embedding_backend}`, sentiment `{settings.sentiment_backend}`."
        )


def main() -> None:
    """Build the navigation and run the selected page."""
    setup_logging(state.settings().log_level)

    if not all(state.artifacts_ready().values()):
        st.title("ITSM Insights")
        st.warning(
            "The models this app serves have not been trained yet. Either run "
            "`itsm train classifier` and `itsm analyze designations` from a terminal, "
            "or prepare everything here.",
            icon=":material/build:",
        )
        if st.button("Prepare the demo", type="primary"):
            state.bootstrap()
            st.rerun()
        st.stop()

    _sidebar()
    navigation = st.navigation(
        {
            "Analyse": [
                st.Page(
                    overview.render,
                    title="Overview",
                    icon=":material/dashboard:",
                    url_path="overview",
                    default=True,
                ),
                st.Page(
                    families.render,
                    title="Ticket families",
                    icon=":material/scatter_plot:",
                    url_path="families",
                ),
                st.Page(
                    embedding_map.render,
                    title="Ticket map (2D)",
                    icon=":material/bubble_chart:",
                    url_path="map",
                ),
                st.Page(
                    signals.render,
                    title="Trends & weak signals",
                    icon=":material/trending_up:",
                    url_path="signals",
                ),
            ],
            "Operate": [
                st.Page(
                    triage.render,
                    title="Intake & triage",
                    icon=":material/support_agent:",
                    url_path="triage",
                ),
            ],
        }
    )
    navigation.run()


main()
