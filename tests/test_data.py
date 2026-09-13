"""The generator must produce a dataset the pipelines can trust."""

from __future__ import annotations

import pandas as pd

from itsm.data.schema import TERMINAL_STATES, TICKET_COLUMNS, TicketState
from itsm.data.synthetic import generate


def test_generator_is_deterministic() -> None:
    first = generate(n_tickets=300, days=60, seed=3).tickets
    second = generate(n_tickets=300, days=60, seed=3).tickets
    pd.testing.assert_frame_equal(first, second)


def test_ticket_frame_matches_schema(tickets: pd.DataFrame) -> None:
    assert set(TICKET_COLUMNS).issubset(tickets.columns)
    assert tickets["ticket_id"].is_unique
    assert tickets["designation"].str.len().gt(0).all()
    assert tickets["opened_at"].is_monotonic_increasing


def test_events_are_consistent_with_tickets(tickets: pd.DataFrame, events: pd.DataFrame) -> None:
    assert set(events["ticket_id"]).issubset(set(tickets["ticket_id"]))
    first_events = events.groupby("ticket_id")["state"].first()
    assert (first_events == TicketState.NEW.value).all()
    # Chronology must hold inside each ticket.
    assert events.groupby("ticket_id")["timestamp"].apply(lambda s: s.is_monotonic_increasing).all()


def test_backlog_and_resolution_are_coherent(tickets: pd.DataFrame) -> None:
    closed = tickets[tickets["state"].isin({s.value for s in TERMINAL_STATES})]
    assert closed["resolution_hours"].dropna().gt(0).all()
    # A realistic extract always contains open tickets, but not mostly.
    open_ratio = 1 - len(closed) / len(tickets)
    assert 0.0 < open_ratio < 0.35


def test_anomalies_raise_the_volume_of_their_category() -> None:
    dataset = generate(n_tickets=6000, days=300, seed=11)
    spike = next(a for a in dataset.anomalies if a.kind == "spike")

    opened = pd.to_datetime(dataset.tickets["opened_at"]).dt.date
    affected = dataset.tickets["category"] == spike.category
    during = affected & (opened >= spike.start) & (opened <= spike.end)
    before = affected & (opened < spike.start)

    days_during = (spike.end - spike.start).days + 1
    days_before = (spike.start - opened.min()).days
    rate_during = during.sum() / days_during
    rate_before = before.sum() / max(days_before, 1)
    assert rate_during > 2 * rate_before
