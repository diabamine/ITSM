"""Shared fixtures: a small, deterministic extract and an offline configuration."""

from __future__ import annotations

import pytest

from backend.data import TicketStore
from backend.mock_data import generate_tickets
from backend.similarity import SimilarTicketIndex, TextEmbedder

END = "2026-06-30"


@pytest.fixture(scope="session")
def tickets():
    """~1500 synthetic tickets over 300 days, ending on a fixed date."""
    return generate_tickets(n=1500, seed=3, end=END, days=300)


@pytest.fixture(scope="session")
def store(tickets):
    return TicketStore(tickets, source="test")


@pytest.fixture(scope="session")
def index(store):
    """Similarity index on TF-IDF: offline and fast."""
    return SimilarTicketIndex(store.tickets, TextEmbedder("tfidf"))


@pytest.fixture
def offline_env(monkeypatch):
    """Run the app offline: synthetic data, TF-IDF, no LLM."""
    for name in ("TICKETS_PATH", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MOCK_TICKETS", "900")
    monkeypatch.setenv("EMBEDDING_BACKEND", "tfidf")
