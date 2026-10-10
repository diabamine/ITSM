"""Plotly figures for the dashboard.

Built as plain functions that return a figure and never call Streamlit, so a
chart can be reused or tested on its own. Colours come from the app theme
(navy text, cyan accent); Streamlit applies fonts and background itself.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

NAVY, CYAN, ORANGE = "#2C2F73", "#5EBFD4", "#E8833A"
PALETTE = [NAVY, CYAN, ORANGE, "#7B61FF", "#2BB673", "#D64545", "#9AA3B5", "#B07CC6"]
BAND = "rgba(94,191,212,0.22)"

_LAYOUT = dict(
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    hovermode="x unified",
    xaxis_title=None,
)


def _style(figure: go.Figure, height: int) -> go.Figure:
    figure.update_layout(height=height, **_LAYOUT)
    figure.update_yaxes(rangemode="tozero", title="tickets")
    return figure


def volume_chart(history: pd.DataFrame, projection: pd.DataFrame | None, height: int = 340) -> go.Figure:
    """Total volume over time, with an optional projection and its uncertainty band.

    Args:
        history: ``period`` and ``tickets`` of the complete periods.
        projection: Output of :func:`backend.analytics.project`, or ``None``.
        height: Figure height in pixels.
    """
    figure = go.Figure()
    figure.add_scatter(
        x=history["period"],
        y=history["tickets"],
        name="Tickets",
        mode="lines+markers",
        line=dict(color=NAVY, width=2.2),
        marker=dict(size=5),
    )
    if projection is not None and len(history):
        # Start the projection at the last real point so the two lines join.
        last = history.iloc[-1]
        x = [last["period"], *projection["period"]]
        band_x = [*x, *projection["period"][::-1], last["period"]]
        band_y = [last["tickets"], *projection["upper"], *projection["lower"][::-1], last["tickets"]]
        figure.add_scatter(
            x=band_x,
            y=band_y,
            mode="lines",  # plotly adds markers by default under 20 points
            fill="toself",
            fillcolor=BAND,
            line=dict(width=0),
            name="95% range",
            hoverinfo="skip",
        )
        figure.add_scatter(
            x=x,
            y=[last["tickets"], *projection["forecast"]],
            name="Projection",
            mode="lines+markers",
            line=dict(color=CYAN, width=2.4, dash="dash"),
            marker=dict(size=5),
        )
    return _style(figure, height)


def series_chart(long: pd.DataFrame, by: str, projections: dict[str, pd.DataFrame], height: int = 380) -> go.Figure:
    """One line per value of ``by``; each one continues as a dashed projection.

    Args:
        long: ``period``, ``by`` and ``tickets`` (complete periods only).
        by: Name of the splitting column.
        projections: ``{value: projection frame}``; values without one are drawn without.
        height: Figure height in pixels.
    """
    figure = go.Figure()
    order = long.groupby(by)["tickets"].sum().sort_values(ascending=False).index  # biggest first
    for position, value in enumerate(order):
        colour = PALETTE[position % len(PALETTE)]
        group = long[long[by] == value].sort_values("period")
        figure.add_scatter(
            x=group["period"],
            y=group["tickets"],
            name=str(value),
            mode="lines",
            line=dict(color=colour, width=2),
            legendgroup=str(value),
        )
        projection = projections.get(value)
        if projection is not None and len(group):
            x = [group["period"].iloc[-1], *projection["period"]]
            figure.add_scatter(
                x=x,
                y=[group["tickets"].iloc[-1], *projection["forecast"]],
                mode="lines",
                line=dict(color=colour, width=2, dash="dot"),
                legendgroup=str(value),
                showlegend=False,
                name=f"{value} (projection)",
            )
    return _style(figure, height)


def distribution_chart(share: pd.DataFrame, height: int = 380) -> go.Figure:
    """Horizontal bars of the share of tickets per value (first column = label)."""
    label = share.columns[0]
    data = share.iloc[::-1]
    figure = go.Figure(
        go.Bar(
            x=data["share"] * 100,
            y=data[label],
            orientation="h",
            marker_color=CYAN,
            text=[f"{s:.0%}" for s in data["share"]],
            textposition="outside",
            cliponaxis=False,
            customdata=data["tickets"],
            hovertemplate="%{y}: %{customdata} tickets (%{x:.1f}%)<extra></extra>",
        )
    )
    figure.update_layout(
        height=height, margin=dict(l=10, r=40, t=10, b=10), xaxis_title="% of tickets", yaxis_title=None
    )
    return figure
