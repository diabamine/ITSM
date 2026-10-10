"""The two places where YOUR models plug in. Everything else in the app is plain analytics.

Until you implement them, the ticket page tells you so and keeps working with what it
has; the report and the dashboard do not need them at all.
"""

from __future__ import annotations

import numpy as np


def embed(texts: list[str]) -> np.ndarray:
    """Return one embedding vector per text, as an array of shape ``(len(texts), dimension)``.

    Called on the whole ticket history once (the result is cached) and on each new
    ticket. Texts are ``"<short description>. <description>"``. Vectors do not need to
    be normalised: the similarity search does it.
    """
    raise NotImplementedError("Implement embed() in backend/ai.py to find similar tickets.")


def extract_fields(ticket: dict[str, str]) -> dict[str, str]:
    """Ask your LLM about a new ticket and return the fields it infers.

    Args:
        ticket: The submitted form, keyed by the extract's column names: ``Description courte``,
            ``Description``, and ``Service``, ``Urgence``, ``Impact``... when chosen.

    Returns:
        ``{column: value}`` using the names of the LLM columns of the extract, for example
        ``{"IncidentType": ..., "Tone": ..., "RequestType": ..., "Scope": ..., "AffectedSystem": ...,
        "RootCause": ...}``. The page displays every key, and the alert rules and the similarity
        search use ``IncidentType``, ``Tone``, ``Scope`` and ``AffectedSystem`` when present.
    """
    raise NotImplementedError("Implement extract_fields() in backend/ai.py to analyse a new ticket.")
