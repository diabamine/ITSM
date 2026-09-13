"""2D ticket map: sentence embeddings, UMAP, HDBSCAN.

A companion to the *Ticket families* page: instead of the persisted
TF-IDF/SVD/KMeans clusterer used to route new tickets, this page fits a
throwaway, interactively-tunable projection on the designation **and**
description, live, and lets the analyst see - not just read a table of - how
tickets group and where they disagree with the declared taxonomy.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from itsm.analysis.embedding_map import EmbeddingMap
from itsm.app import charts, state

_COLOR_CANDIDATES = (
    "category",
    "channel",
    "priority",
    "assignment_group",
    "language",
    "subcategory",
)


def _cluster_label(cluster_id: int, names: dict[int, str]) -> str:
    return "noise" if cluster_id == -1 else f"{cluster_id} · {names[cluster_id]}"


def _plot_frame(frame: pd.DataFrame, projection: EmbeddingMap) -> pd.DataFrame:
    plot = frame.copy()
    plot["x"] = projection.coords[:, 0]
    plot["y"] = projection.coords[:, 1]
    plot["membership"] = projection.probabilities.round(3)
    plot["cluster"] = [_cluster_label(c, projection.cluster_names) for c in projection.labels]
    # Plotly hover boxes do not wrap: a full designation would stretch the
    # tooltip across the screen.
    plot["title"] = plot["designation"].astype(str).str.slice(0, 70)
    return plot


def render() -> None:
    """Draw the 2D ticket map page."""
    st.title("Ticket map (2D)")
    st.caption(
        "Sentence embeddings on the designation and the description, projected to 2D with "
        "UMAP, then clustered in that same 2D space with HDBSCAN - so the picture below *is* "
        "the space the clusters were found in. Compare against the declared category, channel "
        "or assignment group by changing the color."
    )

    tickets = state.tickets()
    settings = state.settings()

    with st.expander("Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        backend = col1.selectbox(
            "Text representation",
            ["tfidf", "embeddings"],
            index=["tfidf", "embeddings"].index(settings.embedding_backend),
            help=(
                "`tfidf`: word+char n-grams reduced with SVD, offline, seconds. "
                "`embeddings`: multilingual sentence-transformer, needs the `hf` extra and a "
                "one-time model download."
            ),
        )
        max_sample = min(len(tickets), 8000)
        sample_n = col2.slider(
            "Sample size",
            min_value=500,
            max_value=max_sample,
            value=min(3000, max_sample),
            step=500,
        )
        seed = col3.number_input("Seed", min_value=0, value=42, step=1)

        col4, col5, col6 = st.columns(3)
        n_neighbors = col4.slider(
            "UMAP neighbours",
            min_value=5,
            max_value=100,
            value=15,
            help="Higher favours global structure over fine local detail.",
        )
        min_dist = col5.slider(
            "UMAP min distance",
            min_value=0.0,
            max_value=0.99,
            value=0.1,
            help="Lower packs points of the same cluster tighter together.",
        )
        min_cluster_size = col6.slider(
            "HDBSCAN min cluster size",
            min_value=5,
            max_value=200,
            value=25,
            help="Smallest group of points HDBSCAN will call a cluster; the rest is noise.",
        )

    try:
        with st.spinner("Embedding, projecting and clustering..."):
            projection = state.embedding_map(
                backend,
                int(sample_n),
                int(seed),
                int(n_neighbors),
                float(min_dist),
                int(min_cluster_size),
            )
    except RuntimeError as error:
        st.error(str(error), icon=":material/error:")
        st.stop()

    frame = state.embedding_sample(int(sample_n), int(seed))
    plot = _plot_frame(frame, projection)

    metrics = projection.metrics
    columns = st.columns(5)
    columns[0].metric("Tickets mapped", f"{len(plot):,}".replace(",", " "))
    columns[1].metric("Clusters", projection.n_clusters)
    columns[2].metric("Noise", f"{metrics.get('noise_ratio', 0):.0%}")
    columns[3].metric(
        "Silhouette (2D)",
        metrics.get("silhouette", "-"),
        help="Geometric separation of the clusters in the 2D layout itself.",
    )
    columns[4].metric(
        "Adjusted Rand vs category",
        metrics.get("adjusted_rand", "-"),
        help="Agreement with the declared category, corrected for chance.",
    )

    color_options = ["cluster", *(c for c in _COLOR_CANDIDATES if c in plot.columns)]
    color = st.selectbox("Color by", color_options)
    noise_value = "noise" if color == "cluster" else None
    scale = charts.categorical_scale(plot[color].astype(str), noise_value=noise_value)
    hover_columns = [
        "title",
        "category",
        "cluster",
        "membership",
        *(c for c in _COLOR_CANDIDATES if c in plot.columns and c != color),
    ]
    st.plotly_chart(
        charts.embedding_scatter(plot, color=color, color_map=scale, hover_columns=hover_columns),
        key=f"map-{color}",
    )
    st.caption(
        "Hover a point for the ticket, drag to zoom, double-click to reset, and click a "
        "legend entry to hide that group (double-click to isolate it)."
    )

    st.divider()
    detail, comparison = st.columns([2, 3])

    summary = (
        plot.groupby("cluster")
        .agg(
            tickets=("cluster", "size"),
            dominant_category=("category", lambda s: s.value_counts().index[0]),
        )
        .reset_index()
    )
    summary["share"] = (summary["tickets"] / len(plot)).round(4)
    summary["purity"] = [
        float((plot.loc[plot["cluster"] == c, "category"] == dominant).mean())
        for c, dominant in zip(summary["cluster"], summary["dominant_category"], strict=True)
    ]
    summary = summary.sort_values("tickets", ascending=False)

    with detail:
        st.subheader("Inspect a cluster")
        clusters_only = summary[summary["cluster"] != "noise"]
        if clusters_only.empty:
            st.info("HDBSCAN found no cluster here - only noise. Try a lower min cluster size.")
        else:
            choice = st.selectbox("Cluster", clusters_only["cluster"].tolist())
            cluster_id = int(choice.split(" · ")[0])
            st.markdown("**Discriminative terms**")
            st.write(
                " ".join(f"`{term}`" for term, _ in projection.cluster_terms.get(cluster_id, []))
            )
            st.markdown("**Most representative tickets** (highest HDBSCAN membership)")
            for index in projection.representative_indices(n=5).get(cluster_id, []):
                row = frame.iloc[index]
                st.markdown(f"- {row['designation']} _(declared: {row['category']})_")

    with comparison:
        st.subheader("Clusters vs declared categories")
        st.caption(
            "A cluster split across several categories means the taxonomy separates what users "
            "experience as one problem; a category spread across clusters means the bucket is "
            "coarser than what the text (title + description) actually distinguishes."
        )
        contingency = (
            pd.crosstab(plot["cluster"], plot["category"], normalize="index")
            .stack()
            .rename("share")
            .reset_index()
        )
        counts = (
            pd.crosstab(plot["cluster"], plot["category"]).stack().rename("tickets").reset_index()
        )
        merged = contingency.merge(counts, on=["cluster", "category"])
        merged = merged[merged["share"] > 0.01]
        st.plotly_chart(charts.purity_heatmap(merged))

    st.divider()
    st.subheader("Cluster summary")
    st.dataframe(
        summary[["cluster", "tickets", "share", "dominant_category", "purity"]],
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
