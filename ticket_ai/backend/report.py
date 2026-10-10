"""Batch report: the trend and the alerts of a set of tickets, as HTML or PDF.

The report is a plain :class:`BatchReport` object (numbers, alerts, figures,
tables) rendered by :meth:`BatchReport.to_html`. The PDF is that same HTML run
through WeasyPrint, so the two can never differ.

Charts are **static SVG** drawn with matplotlib, not Plotly: a report is a
document that must open anywhere, offline, and print. SVG stays sharp at any
zoom, needs no JavaScript, and is embedded inline, so the HTML file is fully
self-contained.
"""

from __future__ import annotations

import html
import io
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd

from backend.alerts import Alert
from backend.analytics import TOPIC_COLUMNS, batch_alerts, distribution, growth_by, kpis, volume_series
from backend.llm import LLMClient, summarise_batch
from backend.schema import NEGATIVE_TONES, Col

NAVY, CYAN, GREY, ORANGE = "#2C2F73", "#5EBFD4", "#9AA3B5", "#E8833A"
LEVEL_COLOURS = {"critical": "#D64545", "warning": ORANGE, "info": CYAN}

plt.rcParams.update(
    {
        "svg.fonttype": "none",  # keep text as text: smaller files, crisp in print
        "font.family": "sans-serif",
        "axes.edgecolor": GREY,
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E3E8EF",
    }
)


@dataclass
class BatchReport:
    """Everything shown in the report, ready to render.

    Attributes:
        title: Report title.
        period: Human-readable period covered by the batch.
        filters: ``{label: value}`` description of the selection, for the appendix.
        indicators: Output of :func:`backend.analytics.kpis`.
        alerts: Batch-level alerts, most severe first.
        summary: Executive summary text.
        summary_source: ``"llm"`` or ``"offline"``.
        figures: ``{caption: inline SVG}``.
        tables: ``{caption: DataFrame}``.
    """

    title: str
    period: str
    filters: dict[str, str]
    indicators: dict[str, float | int | None]
    alerts: list[Alert]
    summary: str
    summary_source: str
    figures: dict[str, str] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    def to_html(self) -> str:
        """Render the report as one self-contained HTML document."""
        return _render_html(self)

    def to_pdf(self) -> bytes:
        """Render the report as PDF (needs WeasyPrint, see :func:`pdf_available`)."""
        if not pdf_available():
            raise RuntimeError("PDF export needs WeasyPrint: pip install weasyprint")
        from weasyprint import HTML

        return HTML(string=self.to_html()).write_pdf()


def pdf_available() -> bool:
    """Whether WeasyPrint (and the system libraries it needs) can be imported."""
    try:
        import weasyprint  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


# --------------------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------------------
def build_report(
    tickets: pd.DataFrame,
    title: str = "Ticket batch report",
    filters: dict[str, str] | None = None,
    client: LLMClient | None = None,
) -> BatchReport:
    """Analyse a batch and assemble its report.

    Args:
        tickets: The selected batch (already filtered).
        title: Title shown on the report.
        filters: Description of how the batch was selected.
        client: LLM client for the executive summary; ``None`` uses a template.
            Only aggregated numbers are ever sent, never ticket text.

    Raises:
        ValueError: if the batch is empty.
    """
    if tickets.empty:
        raise ValueError("The selection contains no ticket.")

    alerts = batch_alerts(tickets)
    indicators = kpis(tickets)
    weekly = volume_series(tickets, "Week")
    period = f"{tickets[Col.OPENED].min():%d %b %Y} - {tickets[Col.OPENED].max():%d %b %Y}"

    facts = {
        "period": period,
        "tickets": len(tickets),
        "change_vs_previous_pct": _recent_change(weekly),
        "sla_breach_pct": indicators["sla_breach_pct"],
        "negative_tone_pct": indicators["negative_tone_pct"],
        "top_incident_types": distribution(tickets, Col.INCIDENT_TYPE, 3)[[Col.INCIDENT_TYPE, "share"]]
        .round(2)
        .to_dict("records"),
        "alerts": [a.title for a in alerts[:5]],
    }
    summary, source = summarise_batch(client, facts)

    figures = {
        "Weekly ticket volume": _volume_chart(weekly),
        "Where the tickets come from": _distribution_chart(distribution(tickets, Col.INCIDENT_TYPE, 8)),
    }
    quality = _quality_chart(tickets)
    if quality:
        figures["SLA breaches and negative tone, by week"] = quality

    return BatchReport(
        title=title,
        period=period,
        filters=filters or {},
        indicators=indicators,
        alerts=alerts,
        summary=summary,
        summary_source=source,
        figures=figures,
        tables=_tables(tickets),
    )


def _recent_change(weekly: pd.DataFrame) -> float | None:
    """Percent change of the last 4 complete weeks against the 4 before them."""
    counts = weekly[weekly["complete"]]["tickets"].to_numpy(dtype=float)
    if len(counts) < 8 or counts[-8:-4].sum() == 0:
        return None
    return float((counts[-4:].sum() - counts[-8:-4].sum()) / counts[-8:-4].sum() * 100)


def _tables(tickets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The two tables of the report: what is growing, and the biggest root causes."""
    tables: dict[str, pd.DataFrame] = {}
    for column in (Col.AFFECTED_SYSTEM, Col.ROOT_CAUSE):
        growth = growth_by(tickets, column).head(5)
        if not growth.empty:
            tables[f"Fastest growing - {TOPIC_COLUMNS[column]} (last 28 days)"] = pd.DataFrame(
                {
                    TOPIC_COLUMNS[column].capitalize(): growth[column],
                    "Tickets": growth["recent"],
                    "Was / day": growth["baseline_per_day"].round(2),
                    "Now / day": growth["recent_per_day"].round(2),
                    "Growth": growth["ratio"].map(lambda r: "new" if pd.isna(r) else f"x{r:.1f}"),
                    "Share of increase": growth["share_of_increase"].map("{:.0%}".format),
                }
            )

    known = tickets[tickets[Col.ROOT_CAUSE].notna() & (tickets[Col.ROOT_CAUSE].astype(str) != "")]
    if not known.empty:
        grouped = known.groupby(Col.ROOT_CAUSE)
        table = (
            pd.DataFrame(
                {
                    "Tickets": grouped.size(),
                    "Median resolution (h)": grouped[Col.RESOLUTION_HOURS].median().round(1),
                    "Typical resolution": grouped[Col.RESOLUTION_SUMMARY].agg(
                        lambda s: next(iter(s.dropna().astype(str).loc[lambda x: x != ""].mode()), "")
                    ),
                }
            )
            .sort_values("Tickets", ascending=False)
            .head(8)
        )
        tables["Largest root causes"] = table.reset_index().rename(columns={Col.ROOT_CAUSE: "Root cause"})
    return tables


# --------------------------------------------------------------------------------------
# Charts (static SVG)
# --------------------------------------------------------------------------------------
def _svg(fig: plt.Figure) -> str:
    """Serialise a figure to inline SVG (without the XML prolog)."""
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    text = buffer.getvalue()
    return text[text.index("<svg") :]


def _volume_chart(weekly: pd.DataFrame) -> str:
    done = weekly[weekly["complete"]]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(done["period"], done["tickets"], color=NAVY, linewidth=1.8, label="Tickets per week")
    ax.plot(done["period"], done["tickets"].rolling(4).mean(), color=CYAN, linewidth=2.4, label="4-week average")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    return _svg(fig)


def _distribution_chart(share: pd.DataFrame) -> str:
    column = share.columns[0]
    data = share.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(data) + 0.8))
    ax.barh(data[column], data["share"] * 100, color=CYAN)
    for position, value in enumerate(data["share"] * 100):
        ax.text(value + 0.4, position, f"{value:.0f}%", va="center", fontsize=9)
    ax.set_xlabel("% of tickets")
    ax.grid(axis="y", visible=False)
    return _svg(fig)


def _quality_chart(tickets: pd.DataFrame) -> str | None:
    """Weekly share of SLA breaches and of negative-tone tickets; ``None`` if undefined."""
    frame = tickets.assign(week=tickets[Col.OPENED].dt.to_period("W").dt.start_time)
    done = frame[frame[Col.SLA].isin(["Respecté", "Dépassé"])]
    tones = frame[frame[Col.TONE].notna()]
    if done.empty or tones.empty:
        return None
    breach = done.groupby("week")[Col.SLA].apply(lambda s: (s == "Dépassé").mean() * 100)
    negative = tones.groupby("week")[Col.TONE].apply(lambda s: s.isin(NEGATIVE_TONES).mean() * 100)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(breach.index, breach.rolling(3, min_periods=1).mean(), color=ORANGE, linewidth=2, label="SLA missed (%)")
    ax.plot(
        negative.index, negative.rolling(3, min_periods=1).mean(), color=NAVY, linewidth=2, label="Negative tone (%)"
    )
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    return _svg(fig)


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------
_CSS = f"""
@page {{ size: A4; margin: 15mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: system-ui, "Segoe UI", Helvetica, Arial, sans-serif; color: {NAVY};
       margin: 0 auto; max-width: 920px; padding: 24px; line-height: 1.45; }}
h1 {{ margin: 0 0 4px; font-size: 26px; }}
h2 {{ font-size: 17px; margin: 28px 0 10px; padding-bottom: 4px; border-bottom: 2px solid {CYAN}; }}
.muted {{ color: #6B7390; font-size: 13px; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; }}
.kpi {{ background: #F0F4F8; border-radius: 8px; padding: 10px 12px; }}
.kpi b {{ display: block; font-size: 22px; }}
.kpi span {{ font-size: 12px; color: #6B7390; }}
.alert {{ border-left: 5px solid; background: #F8FAFC; padding: 8px 12px; margin: 8px 0; border-radius: 4px;
         break-inside: avoid; }}
.alert b {{ display: block; }}
.badge {{ font-size: 11px; background: #F0F4F8; border-radius: 10px; padding: 2px 8px; color: #6B7390; }}
figure {{ margin: 8px 0 14px; break-inside: avoid; }}
figure svg {{ width: 100%; height: auto; }}
figcaption {{ font-size: 13px; color: #6B7390; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 6px 0 14px; break-inside: avoid; }}
th {{ text-align: left; background: #F0F4F8; }}
th, td {{ padding: 6px 8px; border-bottom: 1px solid #E3E8EF; vertical-align: top; }}
footer {{ margin-top: 30px; font-size: 12px; color: #6B7390; }}
"""


def _render_html(report: BatchReport) -> str:
    """Assemble the report document. All dynamic text is HTML-escaped."""
    esc = html.escape
    ind = report.indicators

    def fmt(value: float | None, suffix: str = "", digits: int = 0) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}{suffix}"

    cards = [
        ("Tickets", f"{ind['tickets']}"),
        ("Still open", f"{ind['open']}"),
        ("Median resolution", fmt(ind["median_resolution_h"], " h", 1)),
        ("SLA missed", fmt(ind["sla_breach_pct"], "%")),
        ("Reopened", fmt(ind["reopen_pct"], "%")),
        ("Negative tone", fmt(ind["negative_tone_pct"], "%")),
    ]
    kpi_html = "".join(f'<div class="kpi"><b>{esc(v)}</b><span>{esc(k)}</span></div>' for k, v in cards)

    if report.alerts:
        alert_html = "".join(
            f'<div class="alert" style="border-color:{LEVEL_COLOURS[a.level]}">'
            f"<b>{esc(a.title)}</b>{esc(a.detail)}</div>"
            for a in report.alerts
        )
    else:
        alert_html = '<p class="muted">No alert was raised on this batch.</p>'

    figures = "".join(
        f"<figure>{svg}<figcaption>{esc(caption)}</figcaption></figure>" for caption, svg in report.figures.items()
    )
    tables = "".join(
        f"<h2>{esc(caption)}</h2>{frame.to_html(index=False, border=0)}" for caption, frame in report.tables.items()
    )
    filters = "".join(f"<li><b>{esc(k)}</b>: {esc(v)}</li>" for k, v in report.filters.items()) or "<li>None</li>"
    origin = "written by the LLM from aggregated figures" if report.summary_source == "llm" else "template (no LLM)"

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(report.title)}</title>
<style>{_CSS}</style></head><body>
<h1>{esc(report.title)}</h1>
<div class="muted">{esc(report.period)} - generated {esc(report.generated_at)}</div>
<h2>Key figures</h2><div class="kpis">{kpi_html}</div>
<h2>Summary <span class="badge">{esc(origin)}</span></h2><p>{esc(report.summary)}</p>
<h2>Alerts</h2>{alert_html}
<h2>Trend</h2>{figures}
{tables}
<h2>About this report</h2><p class="muted">Selection:</p><ul class="muted">{filters}</ul>
<footer>Volumes count tickets by opening date. Weeks are Monday-Sunday and only complete weeks are drawn.
Growth compares the last 28 days with the 84 days before; "share of increase" is a topic's part of the total
rise, and alerts are raised only when a rise is also unlikely to be noise (Poisson test).</footer>
</body></html>"""
