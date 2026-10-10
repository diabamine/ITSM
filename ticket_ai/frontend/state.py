"""Shared, cached resources.

Streamlit re-runs the whole script on every interaction, so anything expensive is
built once per process here and reused: the ticket store, the similarity index
(the slow one: it embeds every ticket) and the LLM clients.
"""

from __future__ import annotations

import streamlit as st

from backend.config import Settings
from backend.data import TicketStore
from backend.llm import LLMClient, TicketEnricher
from backend.similarity import SimilarTicketIndex, TextEmbedder


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    """Settings read from the environment (or ``.env``)."""
    return Settings.from_env()


@st.cache_resource(show_spinner="Loading tickets...")
def get_store() -> TicketStore:
    """The ticket extract: the file in ``TICKETS_PATH``, or a synthetic one."""
    return TicketStore.from_settings(get_settings())


@st.cache_resource(show_spinner="Indexing tickets for similarity search (the first run can take a minute)...")
def get_index() -> SimilarTicketIndex:
    """Similarity index over the whole history."""
    settings = get_settings()
    embedder = TextEmbedder(settings.embedding_backend, settings.embedding_model, settings.cache_dir)
    return SimilarTicketIndex(get_store().tickets, embedder)


@st.cache_resource(show_spinner=False)
def get_enricher() -> TicketEnricher:
    """Ticket enricher (LLM when configured, local heuristic otherwise)."""
    return TicketEnricher.from_settings(get_settings())


@st.cache_resource(show_spinner=False)
def get_client() -> LLMClient | None:
    """The LLM client used for report summaries, or ``None`` in offline mode."""
    return LLMClient.from_settings(get_settings())
