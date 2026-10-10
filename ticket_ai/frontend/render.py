"""Plotly figures and the HTML report. Draws what ``backend`` computes; computes nothing itself."""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go

from backend import analytics as an
from backend import data as d

NAVY, CYAN, ORANGE = "#2C2F73", "#5EBFD4", "#E8833A"  # the app theme
PALETTE = [NAVY, CYAN, ORANGE, "#7B61FF", "#2BB673", "#D64545"]
LEVEL_COLOURS = {"error": "#D64545", "warning": ORANGE, "info": CYAN}


def fmt(value: float | None, suffix: str = "", digits: int = 0) -> str:
    """Format a number for display; ``None`` (empty column) becomes ``n/a``."""
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(rangemode="tozero", title="tickets")
    return fig


def volume_fig(history: pd.Series, projection: pd.DataFrame | None = None) -> go.Figure:
    """Volume over time, with an optional projection (dashed) and its 95% band."""
    fig = go.Figure(
        go.Scatter(
            x=history.index,
            y=history,
            name="Tickets",
            mode="lines+markers",
            line=dict(color=NAVY, width=2.2),
            marker=dict(size=5),
        )
    )
    if projection is not None:
        x = [history.index[-1], *projection.index]  # start at the last real point so the lines join
        fig.add_scatter(
            x=[*x, *projection.index[::-1], x[0]],
            y=[history.iloc[-1], *projection["upper"], *projection["lower"][::-1], history.iloc[-1]],
            fill="toself",
            fillcolor="rgba(94,191,212,0.22)",
            line=dict(width=0),
            mode="lines",
            name="95% range",
            hoverinfo="skip",
        )
        fig.add_scatter(
            x=x,
            y=[history.iloc[-1], *projection["forecast"]],
            name="Projection",
            mode="lines+markers",
            line=dict(color=CYAN, width=2.4, dash="dash"),
        )
    return _style(fig, 340)


def lines_fig(wide: pd.DataFrame, projections: dict[str, pd.DataFrame]) -> go.Figure:
    """One line per column of ``wide``, each continued by a dotted projection when there is one."""
    fig = go.Figure()
    for i, column in enumerate(wide):
        colour = PALETTE[i % len(PALETTE)]
        fig.add_scatter(
            x=wide.index,
            y=wide[column],
            name=str(column),
            mode="lines",
            line=dict(color=colour, width=2),
            legendgroup=str(column),
        )
        if column in projections:
            p = projections[column]
            fig.add_scatter(
                x=[wide.index[-1], *p.index],
                y=[wide[column].iloc[-1], *p["forecast"]],
                mode="lines",
                line=dict(color=colour, width=2, dash="dot"),
                legendgroup=str(column),
                showlegend=False,
            )
    return _style(fig, 380)


def bar_fig(share: pd.Series) -> go.Figure:
    """Horizontal bars of shares (0-1) indexed by label."""
    share = share.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=share * 100,
            y=share.index.astype(str),
            orientation="h",
            marker_color=CYAN,
            text=[f"{s:.0%}" for s in share],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(height=max(200, 36 * len(share) + 60), margin=dict(l=10, r=40, t=10, b=10), xaxis_title="% of tickets")
    return fig


_CSS = f"""
body {{ font-family: system-ui, "Segoe UI", Arial, sans-serif; color: {NAVY}; max-width: 920px; margin: 0 auto; padding: 24px; }}
h2 {{ border-bottom: 2px solid {CYAN}; padding-bottom: 4px; margin-top: 28px; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; }}
.kpi {{ background: #F0F4F8; border-radius: 8px; padding: 10px 12px; }} .kpi b {{ display: block; font-size: 22px; }}
.alert {{ border-left: 5px solid; background: #F8FAFC; padding: 8px 12px; margin: 8px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ padding: 6px 8px; border-bottom: 1px solid #E3E8EF; text-align: left; }}
.muted {{ color: #6B7390; }} button {{ float: right; }} @media print {{ button {{ display: none; }} }}
"""


def report_html(batch: pd.DataFrame, title: str = "Ticket batch report") -> str:
    """One self-contained HTML page: key figures, alerts, global trend and what is growing.

    Plotly's JavaScript is embedded, so the file opens offline. To get a PDF, use the page's
    "Print / save as PDF" button (or Ctrl+P): charts and layout print as they are.
    Every value coming from the data is HTML-escaped.
    """
    esc = html.escape
    k = an.kpis(batch)
    cards = [
        ("Tickets", k["tickets"]),
        ("Still open", k["open"]),
        ("Median resolution", fmt(k["median_resolution_h"], " h", 1)),
        ("SLA missed", fmt(k["sla_missed_pct"], "%")),
        ("Reopened", fmt(k["reopened_pct"], "%")),
        ("Negative tone", fmt(k["negative_tone_pct"], "%")),
    ]
    kpi_html = "".join(f'<div class="kpi"><b>{esc(str(v))}</b>{esc(label)}</div>' for label, v in cards)

    alerts = an.batch_alerts(batch)
    alert_html = (
        "".join(
            f'<div class="alert" style="border-color:{LEVEL_COLOURS[a.level]}"><b>{esc(a.title)}</b><br>{esc(a.detail)}</div>'
            for a in alerts
        )
        or '<p class="muted">No alert was raised on this batch.</p>'
    )

    figures: list[tuple[str, go.Figure]] = []
    weekly = an.volume(batch, "W")["Tickets"]
    if len(weekly) >= 3:
        fig = volume_fig(weekly)
        fig.add_scatter(x=weekly.index, y=weekly.rolling(4).mean(), name="4-week average", line=dict(color=CYAN, width=3))
        figures.append(("Weekly volume", fig))
    types = batch[d.INCIDENT_TYPE].dropna().astype(str).value_counts(normalize=True).head(8)
    if not types.empty:
        figures.append(("Incident types", bar_fig(types)))
    chart_html = "".join(
        f"<h2>{esc(caption)}</h2>" + fig.to_html(full_html=False, include_plotlyjs=i == 0)
        for i, (caption, fig) in enumerate(figures)
    )

    tables = ""
    for column in (d.AFFECTED_SYSTEM, d.ROOT_CAUSE):
        g = an.growth(batch, column).head(5)
        if not g.empty:
            tables += f"<h2>Fastest growing: {esc(column)}</h2>" + g.reset_index(names=column).round(2).to_html(
                index=False, border=0, na_rep="new"
            )

    period = f"{batch[d.OPENED].min():%d %b %Y} - {batch[d.OPENED].max():%d %b %Y}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>{_CSS}</style></head><body>
<button onclick="window.print()">Print / save as PDF</button>
<h1>{esc(title)}</h1><p class="muted">{esc(period)}</p>
<h2>Key figures</h2><div class="kpis">{kpi_html}</div>
<h2>Alerts</h2>{alert_html}{chart_html}{tables}
<p class="muted">Weeks run Monday to Sunday and only complete weeks are drawn. Growth compares the last 28 days
with the 84 days before; "share_of_increase" is a topic's part of the total rise and "p_value" the chance of
seeing that many tickets if nothing had changed.</p></body></html>"""
