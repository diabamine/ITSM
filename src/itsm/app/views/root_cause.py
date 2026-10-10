"""Root-cause page: what is below the category, and what is driving it up.

A category chart answers "is Network up?". This page answers "which specific
symptom, and how much of the rise is it?" - the question a problem manager
actually has, using the sub-topic discovery and growth attribution in
:mod:`itsm.analysis.root_cause`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from itsm.app import charts, state
from itsm.pipelines.reporting import read_report

_DRIVER_COLUMNS = [
    "parent_category",
    "label",
    "n_recent",
    "baseline_per_day",
    "recent_per_day",
    "growth_ratio",
    "share_of_increase",
    "poisson_p",
]


def _run_now(recent: int, baseline: int, backend: str) -> None:
    """Run the discovery + attribution pipeline inline, with progress."""
    from itsm.pipelines.root_cause import run

    with st.status("Tagging sub-topics and attributing the increase...", expanded=True) as status:
        st.write("Embedding designations and clustering within each category")
        report = run(
            state.settings(),
            backend=backend,  # type: ignore[arg-type]
            recent_days=recent,
            baseline_days=baseline,
        )
        st.write(f"{report['n_subtopics']} sub-topics, {report['coverage']:.0%} coverage")
        status.update(label="Done", state="complete", expanded=False)
    st.cache_data.clear()
    st.rerun()


def render() -> None:
    """Draw the root-cause page."""
    st.title("What is driving the increase")
    st.caption(
        "Declared categories are a routing taxonomy, too coarse to name a root cause. "
        "This discovers sub-topics inside each category from the text - grouping wordings "
        "that share a meaning, not just vocabulary - then decomposes the recent change in "
        "volume across them (`itsm.analysis.root_cause`)."
    )

    report = read_report("root_cause", state.settings())

    with st.expander("Run / re-run", expanded=report is None):
        columns = st.columns(3)
        recent = columns[0].number_input("Recent window (days)", min_value=3, value=14)
        baseline = columns[1].number_input("Baseline window (days)", min_value=7, value=56)
        backend = columns[2].selectbox("Text representation", ["embeddings", "tfidf"])
        st.caption(
            "`embeddings` groups tickets by meaning across languages and wordings - the "
            "point of this page. `tfidf` is the offline fallback."
        )
        if st.button("Discover sub-topics and attribute the increase", type="primary"):
            _run_now(int(recent), int(baseline), backend)

    if report is None:
        st.info(
            "No report yet. Run it above, or from a terminal: `itsm analyze root-cause`.",
            icon=":material/travel_explore:",
        )
        return

    columns = st.columns(4)
    columns[0].metric("Sub-topics", report["n_subtopics"])
    columns[1].metric("Coverage", f"{report['coverage']:.0%}")
    columns[2].metric(
        "Significant drivers",
        report["n_drivers"],
        help="p <= 0.01 and at least 5% of the total increase - large and unlikely to be noise.",
    )
    columns[3].metric("Window explained", f"last {report['recent_days']} days")

    ranking = pd.DataFrame(report["ranking"])
    if ranking.empty:
        st.success("No sub-topic shows a significant change over this window.")
        return

    st.subheader("What changed")
    ranking["driver"] = (ranking["poisson_p"] <= 0.01) & (ranking["share_of_increase"] >= 0.05)
    st.dataframe(
        ranking[["driver", *_DRIVER_COLUMNS]],
        hide_index=True,
        column_config={
            "driver": st.column_config.CheckboxColumn(
                "driver", help="Large and statistically significant."
            ),
            "parent_category": "category",
            "label": "sub-topic",
            "n_recent": "recent tickets",
            "baseline_per_day": st.column_config.NumberColumn("was / day", format="%.2f"),
            "recent_per_day": st.column_config.NumberColumn("now / day", format="%.2f"),
            "growth_ratio": st.column_config.NumberColumn("x", format="%.1f"),
            "share_of_increase": st.column_config.ProgressColumn(
                "share of increase", min_value=0.0, max_value=1.0, format="%.0f%%"
            ),
            "poisson_p": st.column_config.NumberColumn("p", format="%.1e"),
        },
    )

    drivers = pd.DataFrame(report["drivers"])
    if not drivers.empty:
        st.subheader("Ranked by absolute contribution")
        st.caption(
            "What to fix first: a sub-topic going from 20 to 30 tickets a day contributes "
            "ten times more than one going from 1 to 3, even though the second tripled."
        )
        bar_data = drivers.assign(
            per_day_delta=drivers["recent_per_day"] - drivers["baseline_per_day"],
            topic=drivers["parent_category"] + " - " + drivers["label"].str.slice(0, 40),
        )
        st.altair_chart(
            charts.category_bars(bar_data, value="per_day_delta", label="topic"),
            use_container_width=True,
        )

    emerging = pd.DataFrame(report["emerging"])
    if not emerging.empty:
        st.subheader("Emerging - absent from the baseline entirely")
        st.caption("No growth ratio to compute: these did not exist a baseline-window ago.")
        st.dataframe(
            emerging[["parent_category", "label", "n_recent", "recent_per_day"]],
            hide_index=True,
            column_config={"parent_category": "category", "label": "sub-topic"},
        )

    with st.expander("All discovered sub-topics"):
        subtopics = pd.DataFrame(report["subtopics"])
        st.caption(
            f"{len(subtopics)} sub-topics across {subtopics['parent_category'].nunique()} "
            "categories. A ticket with no close match to any of these is left unassigned "
            "rather than forced into the nearest one."
        )
        st.dataframe(
            subtopics[["topic_id", "parent_category", "label", "size", "keywords"]],
            hide_index=True,
            column_config={"topic_id": "id", "parent_category": "category"},
        )
