"""Altair chart builders.

Altair ships with Streamlit, so no extra dependency. Charts are built as
functions returning a chart object rather than drawn in place: they stay
testable and reusable across pages.

Two conventions:

* a **diverging** scale only where a sign is meaningful (sentiment, residuals);
* every anomaly chart shows the **expected band next to the observed line** -
  an alert without its baseline is not actionable.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

COLOR_OBSERVED = "#2563eb"
COLOR_EXPECTED = "#94a3b8"
COLOR_ALERT = "#dc2626"
COLOR_EMERGING = "#f59e0b"
COLOR_NOISE = "#9aa1ac"

#: Tableau 20 - enough distinguishable hues for a couple dozen HDBSCAN clusters.
TABLEAU20 = [
    "#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D",
    "#B6992D", "#F1CE63", "#499894", "#86BCB6", "#E15759", "#FF9D9A",
    "#79706E", "#BAB0AC", "#D37295", "#FABFD2", "#B07AA1", "#D4A6C8",
    "#9D7660", "#D7B5A6",
]  # fmt: skip


def volume_line(
    frame: pd.DataFrame, x: str = "period", y: str = "tickets", color: str | None = None
) -> alt.Chart:
    """Ticket volume over time, optionally split by a dimension."""
    encoding: dict[str, object] = {
        "x": alt.X(f"{x}:T", title=None),
        "y": alt.Y(f"{y}:Q", title="tickets"),
        "tooltip": [alt.Tooltip(f"{x}:T"), alt.Tooltip(f"{y}:Q")],
    }
    if color:
        encoding["color"] = alt.Color(f"{color}:N", title=None)
    return alt.Chart(frame).mark_line(point=False).encode(**encoding).properties(height=280)


def category_bars(
    frame: pd.DataFrame, value: str = "tickets", label: str = "category"
) -> alt.Chart:
    """Horizontal ranking of a value by category."""
    return (
        alt.Chart(frame)
        .mark_bar(color=COLOR_OBSERVED)
        .encode(
            x=alt.X(f"{value}:Q", title=value.replace("_", " ")),
            y=alt.Y(f"{label}:N", sort="-x", title=None),
            tooltip=list(frame.columns),
        )
        .properties(height=max(220, 26 * len(frame)))
    )


def backlog_area(frame: pd.DataFrame) -> alt.Chart:
    """Arrivals vs departures, with the running backlog on a second axis."""
    melted = frame.melt(
        id_vars="period", value_vars=["created", "resolved"], var_name="flow", value_name="tickets"
    )
    bars = (
        alt.Chart(melted)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("period:T", title=None),
            y=alt.Y("tickets:Q", title="tickets"),
            color=alt.Color(
                "flow:N",
                scale=alt.Scale(domain=["created", "resolved"], range=[COLOR_OBSERVED, "#10b981"]),
                title=None,
            ),
            xOffset="flow:N",
            tooltip=["period:T", "flow:N", "tickets:Q"],
        )
    )
    line = (
        alt.Chart(frame)
        .mark_line(color=COLOR_ALERT, strokeWidth=2)
        .encode(
            x="period:T",
            y=alt.Y("open_backlog:Q", title="open backlog", axis=alt.Axis(titleColor=COLOR_ALERT)),
            tooltip=["period:T", "open_backlog:Q"],
        )
    )
    return alt.layer(bars, line).resolve_scale(y="independent").properties(height=300)


def anomaly_chart(
    diagnostics: pd.DataFrame, alerts: pd.DataFrame
) -> alt.LayerChart | alt.FacetChart:
    """Observed vs expected counts, with alert markers.

    Args:
        diagnostics: Per-day frame for a single series, with ``period``,
            ``observed`` and ``expected``.
        alerts: Alerts for the same series, with ``period``, ``kind`` and
            ``observed``.
    """
    base = alt.Chart(diagnostics).encode(x=alt.X("period:T", title=None))
    expected = base.mark_line(color=COLOR_EXPECTED, strokeDash=[4, 3]).encode(
        y=alt.Y("expected:Q", title="tickets per period"),
        tooltip=["period:T", "expected:Q"],
    )
    observed = base.mark_line(color=COLOR_OBSERVED, strokeWidth=2).encode(
        y="observed:Q",
        tooltip=["period:T", "observed:Q", "expected:Q", "robust_z:Q", "ewma:Q"],
    )
    layers = [expected, observed]

    if len(alerts):
        markers = (
            alt.Chart(alerts)
            .mark_point(size=110, filled=True, opacity=0.9)
            .encode(
                x="period:T",
                y="observed:Q",
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(
                        domain=["spike", "emerging"], range=[COLOR_ALERT, COLOR_EMERGING]
                    ),
                    title="alert",
                ),
                shape=alt.Shape("kind:N", legend=None),
                tooltip=["period:T", "kind:N", "observed:Q", "expected:Q", "severity:Q"],
            )
        )
        layers.append(markers)
    return alt.layer(*layers).properties(height=320).interactive(bind_y=False)


def residual_chart(
    diagnostics: pd.DataFrame, ewma_threshold: float
) -> alt.LayerChart | alt.FacetChart:
    """EWMA of standardised residuals, with the decision threshold."""
    line = (
        alt.Chart(diagnostics)
        .mark_area(color=COLOR_EMERGING, opacity=0.45)
        .encode(
            x=alt.X("period:T", title=None),
            y=alt.Y("ewma:Q", title="EWMA of residuals"),
            tooltip=["period:T", "ewma:Q", "robust_z:Q"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"threshold": [ewma_threshold]}))
        .mark_rule(color=COLOR_ALERT, strokeDash=[5, 4])
        .encode(y="threshold:Q")
    )
    return alt.layer(line, rule).properties(height=160)


def probability_bars(frame: pd.DataFrame) -> alt.Chart:
    """Class probabilities of a single prediction."""
    return (
        alt.Chart(frame)
        .mark_bar(color=COLOR_OBSERVED)
        .encode(
            x=alt.X("score:Q", title="probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=["label:N", "score:Q"],
        )
        .properties(height=28 * max(len(frame), 2))
    )


def categorical_scale(values: pd.Series | list[str], noise_value: str | None = None) -> alt.Scale:
    """Fixed-order categorical color scale, with a neutral gray for ``noise_value``.

    Colors are assigned in a fixed sorted order rather than left to Altair's
    default (which reorders as the data changes), and HDBSCAN's noise label - if
    present - always renders as the same neutral gray instead of a cycled hue.
    """
    unique = sorted(str(v) for v in set(values) if v != noise_value)
    colors = [TABLEAU20[i % len(TABLEAU20)] for i in range(len(unique))]
    if noise_value is not None and noise_value in set(values):
        return alt.Scale(domain=[noise_value, *unique], range=[COLOR_NOISE, *colors])
    return alt.Scale(domain=unique, range=colors)


def embedding_scatter(
    frame: pd.DataFrame, color: str, color_scale: alt.Scale, tooltip: list[str]
) -> alt.Chart:
    """Interactive 2D layout (e.g. a UMAP projection), points coloured by ``color``.

    The axes are UMAP components: their scale and orientation carry no meaning
    on their own, only relative distance does, so their ticks are hidden.
    ``.interactive()`` enables pan/zoom; the tooltip is the per-point detail.
    """
    return (
        alt.Chart(frame)
        .mark_circle(size=42, opacity=0.7)
        .encode(
            x=alt.X("x:Q", title=None, axis=alt.Axis(labels=False, ticks=False, grid=False)),
            y=alt.Y("y:Q", title=None, axis=alt.Axis(labels=False, ticks=False, grid=False)),
            color=alt.Color(
                f"{color}:N", scale=color_scale, legend=alt.Legend(title=None, symbolLimit=40)
            ),
            tooltip=tooltip,
        )
        .properties(height=560)
        .interactive()
    )


def purity_heatmap(frame: pd.DataFrame) -> alt.Chart:
    """Cluster x declared-category contingency, normalised per cluster."""
    return (
        alt.Chart(frame)
        .mark_rect()
        .encode(
            x=alt.X("category:N", title=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("cluster:N", title="family"),
            color=alt.Color("share:Q", scale=alt.Scale(scheme="blues"), title="share"),
            tooltip=["cluster:N", "category:N", "share:Q", "tickets:Q"],
        )
        .properties(height=max(240, 24 * frame["cluster"].nunique()))
    )
