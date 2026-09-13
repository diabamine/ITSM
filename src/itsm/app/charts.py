"""Plotly figure builders.

Plotly rather than a static renderer because these charts are read by someone
looking for an explanation, not an illustration: hovering a day gives the exact
observed/expected pair, the legend toggles a series off, the time axis zooms and
pans, and a spike can be clicked to pull up the tickets behind it.

Two conventions hold across the module:

* **Builders are pure.** They take dataframes, return a ``go.Figure`` and never
  call ``st.*`` - so they stay reusable, unit-testable, and the page decides how
  to display them.
* **Every anomaly chart shows the expected baseline - and, where the scale is
  known, the alert band - next to the observed line.** An alert without the
  baseline it departed from is not actionable.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_OBSERVED = "#2563eb"
COLOR_EXPECTED = "#94a3b8"
COLOR_RESOLVED = "#10b981"
COLOR_ALERT = "#dc2626"
COLOR_EMERGING = "#f59e0b"
#: Points HDBSCAN refused to assign. Deliberately the dullest colour on the
#: page: noise must be visible without competing with the real clusters.
COLOR_NOISE = "#cbd5e1"

#: Qualitative palette for categorical colouring, long enough for ~30 clusters.
_QUALITATIVE = (*px.colors.qualitative.Set2, *px.colors.qualitative.Dark24)

#: Applied to every figure: tight margins, legend on top, full hover labels.
_LAYOUT: dict[str, object] = {
    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
    "hoverlabel": {"namelength": -1},
}


def _style(figure: go.Figure, height: int = 300, unified_hover: bool = True) -> go.Figure:
    """Apply the shared layout to a figure."""
    figure.update_layout(height=height, **_LAYOUT)
    if unified_hover:
        figure.update_layout(hovermode="x unified")
    return figure


def _time_controls(figure: go.Figure, slider: bool = False) -> go.Figure:
    """Add zoom presets - and optionally a range slider - to a time axis.

    A year of daily points does not fit a screen legibly. The presets are how a
    reader goes from "the whole year" to "the week of the incident" without
    leaving the page.
    """
    figure.update_xaxes(
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                {"step": "all", "label": "all"},
            ],
            "y": 1.06,
            "yanchor": "bottom",
            "bgcolor": "rgba(148,163,184,0.15)",
        },
        rangeslider={"visible": slider, "thickness": 0.08},
    )
    return figure


def volume_line(
    frame: pd.DataFrame, x: str = "period", y: str = "tickets", color: str | None = None
) -> go.Figure:
    """Ticket volume over time, optionally split by a dimension.

    With a ``color`` split the legend doubles as a filter: click one family to
    hide it, double-click to isolate it.
    """
    figure = px.line(frame, x=x, y=y, color=color, markers=len(frame) < 60)
    figure.update_traces(hovertemplate="%{y} tickets<extra>%{fullData.name}</extra>")
    figure.update_yaxes(title="tickets", rangemode="tozero")
    figure.update_xaxes(title=None)
    return _time_controls(_style(figure, height=300))


def category_bars(
    frame: pd.DataFrame, value: str = "tickets", label: str = "category"
) -> go.Figure:
    """Horizontal ranking of a value by category."""
    ordered = frame.sort_values(value)
    figure = px.bar(ordered, x=value, y=label, orientation="h", text=value)
    figure.update_traces(
        marker_color=COLOR_OBSERVED,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    )
    figure.update_xaxes(title=value.replace("_", " "))
    figure.update_yaxes(title=None, type="category")
    return _style(figure, height=max(240, 30 * len(frame)), unified_hover=False)


def backlog_area(frame: pd.DataFrame) -> go.Figure:
    """Arrivals vs departures, with the running backlog on a second axis.

    The bars answer "how much came in and went out"; the line answers "is the
    queue growing" - two questions that only make sense together.
    """
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_bar(
        x=frame["period"],
        y=frame["created"],
        name="created",
        marker_color=COLOR_OBSERVED,
        opacity=0.85,
        hovertemplate="created: %{y}<extra></extra>",
    )
    figure.add_bar(
        x=frame["period"],
        y=frame["resolved"],
        name="resolved",
        marker_color=COLOR_RESOLVED,
        opacity=0.85,
        hovertemplate="resolved: %{y}<extra></extra>",
    )
    figure.add_scatter(
        x=frame["period"],
        y=frame["open_backlog"],
        name="open backlog",
        mode="lines",
        line={"color": COLOR_ALERT, "width": 2},
        hovertemplate="backlog: %{y}<extra></extra>",
        secondary_y=True,
    )
    figure.update_layout(barmode="group", bargap=0.15)
    figure.update_yaxes(title="tickets", secondary_y=False, rangemode="tozero")
    figure.update_yaxes(title="open backlog", secondary_y=True, showgrid=False, color=COLOR_ALERT)
    figure.update_xaxes(title=None)
    return _time_controls(_style(figure, height=330))


def anomaly_chart(
    diagnostics: pd.DataFrame, alerts: pd.DataFrame, z_spike: float | None = None
) -> go.Figure:
    """Observed vs expected counts, the alert band, and the alerts themselves.

    Args:
        diagnostics: Per-period frame of a single series, with ``period``,
            ``observed``, ``expected`` and - when available - ``scale``.
        alerts: Alerts for that series, with ``period``, ``kind``, ``observed``,
            ``expected`` and ``severity``.
        z_spike: Spike threshold. Given together with ``scale``, it draws the
            level a single day has to cross to be flagged, which makes the
            decision rule visible instead of implicit.

    Returns:
        A figure layering the threshold, the expected line, the observed line
        and one marker trace per alert kind.
    """
    figure = go.Figure()

    if z_spike is not None and "scale" in diagnostics.columns:
        figure.add_scatter(
            x=diagnostics["period"],
            y=diagnostics["expected"] + z_spike * diagnostics["scale"],
            name=f"spike threshold (z={z_spike:g})",
            mode="lines",
            line={"color": "rgba(220,38,38,0.45)", "dash": "dot", "width": 1},
            hovertemplate="threshold: %{y:.1f}<extra></extra>",
        )

    figure.add_scatter(
        x=diagnostics["period"],
        y=diagnostics["expected"],
        name="expected",
        mode="lines",
        line={"color": COLOR_EXPECTED, "dash": "dash", "width": 1.5},
        hovertemplate="expected: %{y:.1f}<extra></extra>",
    )
    figure.add_scatter(
        x=diagnostics["period"],
        y=diagnostics["observed"],
        name="observed",
        mode="lines",
        line={"color": COLOR_OBSERVED, "width": 2},
        hovertemplate="observed: %{y}<extra></extra>",
    )

    for kind, color, symbol in (
        ("spike", COLOR_ALERT, "circle"),
        ("emerging", COLOR_EMERGING, "diamond"),
    ):
        subset = alerts[alerts["kind"] == kind] if len(alerts) else pd.DataFrame()
        if not len(subset):
            continue
        figure.add_scatter(
            x=subset["period"],
            y=subset["observed"],
            name=kind,
            mode="markers",
            marker={
                "color": color,
                "size": 11,
                "symbol": symbol,
                "line": {"width": 1, "color": "white"},
            },
            customdata=subset[["expected", "severity"]].to_numpy(),
            hovertemplate=(
                f"<b>{kind}</b><br>observed: %{{y}}<br>expected: %{{customdata[0]:.1f}}"
                "<br>severity: %{customdata[1]:.1f}<extra></extra>"
            ),
        )

    figure.update_yaxes(title="tickets per period", rangemode="tozero")
    figure.update_xaxes(title=None)
    return _time_controls(_style(figure, height=360), slider=True)


def residual_chart(diagnostics: pd.DataFrame, ewma_threshold: float) -> go.Figure:
    """EWMA of standardised residuals against its decision threshold."""
    figure = go.Figure()
    figure.add_scatter(
        x=diagnostics["period"],
        y=diagnostics["ewma"],
        name="EWMA",
        mode="lines",
        line={"color": COLOR_EMERGING, "width": 1.5},
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.35)",
        hovertemplate="EWMA: %{y:.2f}<extra></extra>",
    )
    figure.add_hline(
        y=ewma_threshold,
        line={"color": COLOR_ALERT, "dash": "dash", "width": 1.5},
        annotation_text=f"threshold {ewma_threshold:g}",
        annotation_position="top left",
    )
    figure.update_yaxes(title="EWMA of residuals", rangemode="tozero")
    figure.update_xaxes(title=None)
    return _style(figure, height=200)


def probability_bars(frame: pd.DataFrame) -> go.Figure:
    """Class probabilities of a single prediction."""
    ordered = frame.sort_values("score")
    figure = px.bar(
        ordered,
        x="score",
        y="label",
        orientation="h",
        text=[f"{value:.0%}" for value in ordered["score"]],
    )
    figure.update_traces(
        marker_color=COLOR_OBSERVED,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    )
    figure.update_xaxes(title="probability", range=[0, 1.15], tickformat=".0%")
    figure.update_yaxes(title=None, type="category")
    return _style(figure, height=max(180, 46 * len(frame)), unified_hover=False)


def categorical_scale(values: pd.Series, noise_value: str | None = None) -> dict[str, str]:
    """Build a stable colour map for a categorical column.

    Args:
        values: The column that will drive the colour.
        noise_value: Category to render in grey - HDBSCAN's ``noise`` bucket,
            which should stay legible without competing with real clusters.

    Returns:
        A ``category -> colour`` mapping for Plotly's ``color_discrete_map``.
    """
    categories = sorted({str(value) for value in values})
    if noise_value is not None:
        categories = [value for value in categories if value != noise_value]

    mapping = {
        category: _QUALITATIVE[index % len(_QUALITATIVE)]
        for index, category in enumerate(categories)
    }
    if noise_value is not None:
        mapping[noise_value] = COLOR_NOISE
    return mapping


def embedding_scatter(
    frame: pd.DataFrame,
    color: str,
    color_map: dict[str, str] | None = None,
    hover_columns: tuple[str, ...] | list[str] = (),
) -> go.Figure:
    """2D ticket map: one point per ticket, coloured by any attribute.

    The axes are hidden on purpose: UMAP coordinates have no unit and no
    meaning beyond relative position, and labelling them invites readers to
    over-interpret distance. Rendered through WebGL so a few thousand points
    stay smooth to pan and zoom.

    Args:
        frame: Must carry ``x`` and ``y`` plus the colour and hover columns.
        color: Column driving the colour.
        color_map: Output of :func:`categorical_scale`.
        hover_columns: Extra columns to show on hover.
    """
    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color=color,
        color_discrete_map=color_map or {},
        hover_data=list(hover_columns),
        render_mode="webgl",
        opacity=0.75,
    )
    figure.update_traces(marker={"size": 6, "line": {"width": 0}})
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    styled = _style(figure, height=560, unified_hover=False)
    # A vertical legend on the right: with 20+ clusters a horizontal one eats
    # half the plot.
    styled.update_layout(
        legend={"orientation": "v", "yanchor": "top", "y": 1, "x": 1.01, "title": color}
    )
    return styled


def purity_heatmap(frame: pd.DataFrame) -> go.Figure:
    """Discovered family x declared category contingency, normalised per family.

    Args:
        frame: Long-format frame with ``cluster``, ``category``, ``share`` and
            ``tickets``.
    """
    shares = frame.pivot_table(index="cluster", columns="category", values="share", fill_value=0.0)
    counts = frame.pivot_table(
        index="cluster", columns="category", values="tickets", fill_value=0
    ).reindex(index=shares.index, columns=shares.columns, fill_value=0)

    figure = go.Figure(
        go.Heatmap(
            z=shares.to_numpy(),
            x=[str(column) for column in shares.columns],
            y=[str(index) for index in shares.index],
            customdata=counts.to_numpy(),
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar={"title": "share", "tickformat": ".0%"},
            hovertemplate=(
                "family %{y} x %{x}<br>share: %{z:.0%}<br>tickets: %{customdata}<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title=None, tickangle=-40)
    figure.update_yaxes(title="family", type="category")
    return _style(figure, height=max(260, 22 * len(shares)), unified_hover=False)
