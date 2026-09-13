"""Families page.

What the designations cluster into, and how that compares to the declared
taxonomy.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from itsm.app import charts, state
from itsm.pipelines.reporting import read_report


def render() -> None:
    """Draw the ticket-families page."""
    st.title("Families of tickets")
    st.caption(
        "Unsupervised discovery on the designation: normalise, vectorise, reduce, "
        "cluster, then name each family with class-based TF-IDF "
        "(`itsm.analysis.clustering`)."
    )

    model = state.clusterer()
    summaries = model.summaries_
    if not summaries:
        st.warning("The stored clusterer carries no summaries; re-run `itsm analyze designations`.")
        return

    report = read_report("designation_analysis", state.settings()) or {}
    metrics = report.get("metrics", {})

    columns = st.columns(4)
    columns[0].metric("Families", len(summaries))
    columns[1].metric(
        "Silhouette",
        metrics.get("silhouette", "-"),
        help=(
            "Geometric separation. On short text, 0.05-0.25 is normal; "
            "read it as a relative measure."
        ),
    )
    columns[2].metric(
        "Adjusted Rand",
        metrics.get("adjusted_rand", "-"),
        help="Agreement with the declared taxonomy, corrected for chance.",
    )
    columns[3].metric(
        "Mean purity",
        f"{report.get('taxonomy_alignment', {}).get('mean_cluster_purity', 0):.0%}",
        help="Share of the dominant declared category inside each family.",
    )

    frame = pd.DataFrame(
        [
            {
                "cluster": summary.cluster_id,
                "label": summary.name,
                "tickets": summary.size,
                "share": summary.share,
                "dominant category": summary.dominant_category,
                "purity": summary.category_purity,
            }
            for summary in summaries
        ]
    ).sort_values("tickets", ascending=False)

    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "share": st.column_config.ProgressColumn(
                "share", min_value=0.0, max_value=1.0, format="%.1f%%"
            ),
            "purity": st.column_config.ProgressColumn(
                "purity", min_value=0.0, max_value=1.0, format="%.0f%%"
            ),
        },
    )

    st.divider()
    detail, comparison = st.columns([2, 3])

    with detail:
        st.subheader("Inspect a family")
        choice = st.selectbox(
            "Family",
            frame["cluster"].tolist(),
            format_func=lambda cluster_id: (
                f"{cluster_id} - {frame.loc[frame['cluster'] == cluster_id, 'label'].iloc[0]}"
            ),
        )
        summary = next(s for s in summaries if s.cluster_id == int(choice))
        st.markdown("**Discriminative terms**")
        st.write(" ".join(f"`{term}`" for term in summary.top_terms))
        st.markdown("**Representative tickets**")
        for designation in summary.representatives:
            st.markdown(f"- {designation}")

    with comparison:
        st.subheader("Families vs declared categories")
        st.caption(
            "A family split across two categories means the taxonomy separates what "
            "users experience as one problem; a category split across families means "
            "the bucket is too coarse to route on."
        )
        tickets = state.tickets()
        labels = model.predict(tickets["designation"].astype(str).tolist())
        contingency = (
            pd.crosstab(pd.Series(labels, name="cluster"), tickets["category"], normalize="index")
            .stack()
            .rename("share")
            .reset_index()
        )
        counts = (
            pd.crosstab(pd.Series(labels, name="cluster"), tickets["category"])
            .stack()
            .rename("tickets")
            .reset_index()
        )
        merged = contingency.merge(counts, on=["cluster", "category"])
        merged = merged[merged["share"] > 0.01]
        st.altair_chart(charts.purity_heatmap(merged), use_container_width=True)
