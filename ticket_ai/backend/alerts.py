"""Alert rules for a single ticket.

Alerts are **rules, not model outputs**. The LLM supplies facts (tone, scope,
incident type) and the history supplies context (how many similar tickets
recently, how they went); the decision to raise a flag is plain code that anyone
can read, test and tune. That is what makes an alert explainable to the person
who receives it.

Batch-level alerts (volume spikes, growing topics) live in
:mod:`backend.analytics`, next to the statistics they depend on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from backend.llm import Enrichment
from backend.schema import NEGATIVE_TONES, WIDE_SCOPES, Col

LEVEL_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class Alert:
    """One flag raised on a ticket or a batch.

    Attributes:
        level: ``critical``, ``warning`` or ``info``.
        code: Stable identifier, useful for tests and for filtering.
        title: One-line message.
        detail: Why it was raised, with the numbers behind it.
    """

    level: str
    code: str
    title: str
    detail: str = ""


@dataclass(frozen=True)
class Thresholds:
    """Tunable limits of the ticket rules."""

    recurring_similarity: float = 0.6  # a past ticket counts as "similar" above this
    recurring_days: int = 14  # ... if it was opened within this window
    recurring_min: int = 5  # warning at this many similar recent tickets
    recurring_critical: int = 10
    hard_reopen_rate: float = 0.5  # mean reopenings among close neighbours
    hard_sla_breach: float = 0.4  # share of close neighbours that missed their SLA


DEFAULT_THRESHOLDS = Thresholds()  # frozen, so safe to share as a default argument


def ticket_alerts(
    ticket: Mapping[str, str],
    enrichment: Enrichment,
    neighbours: pd.DataFrame,
    reference_date: pd.Timestamp,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> list[Alert]:
    """Raise the flags that apply to a new ticket.

    Args:
        ticket: The submitted form (``Urgence`` and ``Impact`` are read from it).
        enrichment: What the LLM (or fallback) inferred.
        neighbours: Output of :meth:`SimilarTicketIndex.search`, ideally with
            ``k`` around 30 so recurrence can be counted.
        reference_date: "Now" in the history's own timeline.
        thresholds: Rule limits.

    Returns:
        Alerts sorted from most to least severe.
    """
    alerts: list[Alert] = []

    if ticket.get(Col.URGENCY) == "High" and ticket.get(Col.IMPACT) == "High":
        alerts.append(
            Alert("critical", "MAJOR_INCIDENT", "Possible major incident", "Urgency and impact are both High.")
        )
    if enrichment.incident_type == "Sécurité":
        alerts.append(
            Alert(
                "critical",
                "SECURITY",
                "Security-related ticket",
                "Route to the security team and avoid asking the user to forward the content.",
            )
        )
    if enrichment.scope in WIDE_SCOPES:
        alerts.append(
            Alert(
                "warning",
                "WIDE_SCOPE",
                f"Wide impact: {enrichment.scope}",
                "The text suggests several users are affected.",
            )
        )
    if enrichment.tone in NEGATIVE_TONES:
        alerts.append(
            Alert(
                "warning",
                "NEGATIVE_TONE",
                f"Escalation risk: tone is {enrichment.tone.lower()}",
                "Consider a proactive call-back before the SLA expires.",
            )
        )

    alerts += _history_alerts(neighbours, reference_date, thresholds)

    if enrichment.pii_found:
        alerts.append(
            Alert(
                "info",
                "PII",
                "Personal data detected in the text",
                "Found: " + ", ".join(enrichment.pii_found) + ". Masked before any LLM call.",
            )
        )
    return sorted(alerts, key=lambda a: LEVEL_ORDER[a.level])


def _history_alerts(neighbours: pd.DataFrame, reference_date: pd.Timestamp, t: Thresholds) -> list[Alert]:
    """Rules that need the history: recurrence and difficulty of similar tickets."""
    alerts: list[Alert] = []
    if neighbours.empty:
        return alerts

    close = neighbours[neighbours["Similarity"] >= t.recurring_similarity]
    recent = close[close[Col.OPENED] >= reference_date - pd.Timedelta(days=t.recurring_days)]
    if len(recent) >= t.recurring_min:
        level = "critical" if len(recent) >= t.recurring_critical else "warning"
        alerts.append(
            Alert(
                level,
                "RECURRING",
                f"Emerging pattern: {len(recent)} similar tickets in {t.recurring_days} days",
                "Several recent tickets describe the same problem; this may be one incident, not many.",
            )
        )

    top = neighbours.head(10)
    reopen = float(pd.to_numeric(top[Col.REOPEN_COUNT], errors="coerce").mean())
    breach = float((top[Col.SLA] == "Dépassé").mean())
    if reopen >= t.hard_reopen_rate or breach >= t.hard_sla_breach:
        alerts.append(
            Alert(
                "warning",
                "HARD_TO_RESOLVE",
                "Similar tickets were hard to resolve",
                f"Among the 10 closest: {breach:.0%} missed their SLA, {reopen:.1f} reopenings on average.",
            )
        )
    return alerts
