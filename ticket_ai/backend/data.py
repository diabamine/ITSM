"""Loading the extract and slicing it for the pages.

One small class, :class:`TicketStore`, owns the DataFrame. The frontend never
reads files or filters rows itself: it asks the store.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from backend.config import Settings
from backend.mock_data import generate_tickets
from backend.schema import (
    ALL_COLUMNS,
    REQUIRED_COLUMNS,
    Col,
    coerce_types,
    normalise_columns,
)


class MissingColumnsError(ValueError):
    """Raised when an extract lacks columns the app cannot work without."""


def load_extract(path: Path) -> pd.DataFrame:
    """Read a CSV or XLSX extract and bring it to the canonical schema.

    CSV separators and encodings are auto-detected because French exports are
    usually ``;``-separated and Windows-1252 encoded.

    Args:
        path: Location of the file.

    Returns:
        A frame with canonical headers, parsed dates and numbers. Columns that
        are absent from the file (typically the LLM ones) are added empty, so
        every page can rely on them existing.

    Raises:
        MissingColumnsError: if a column in :data:`~backend.schema.REQUIRED_COLUMNS`
            is missing.
    """
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    else:
        try:
            raw = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw = pd.read_csv(path, sep=None, engine="python", encoding="cp1252")

    frame = coerce_types(normalise_columns(raw))
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise MissingColumnsError(f"The extract is missing required columns: {missing}")
    for column in ALL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame


class TicketStore:
    """The ticket extract plus the few queries the pages need.

    Attributes:
        tickets: The full frame, sorted by opening date.
        source: Human-readable origin (file name or ``synthetic``), shown in the UI.
    """

    def __init__(self, tickets: pd.DataFrame, source: str = "unknown") -> None:
        self.tickets = tickets.sort_values(Col.OPENED).reset_index(drop=True)
        self.source = source

    @classmethod
    def from_settings(cls, settings: Settings) -> TicketStore:
        """Load ``TICKETS_PATH`` if set, otherwise generate a synthetic extract."""
        if settings.tickets_path:
            return cls(load_extract(settings.tickets_path), source=settings.tickets_path.name)
        return cls(generate_tickets(n=settings.mock_tickets), source="synthetic")

    @property
    def reference_date(self) -> pd.Timestamp:
        """The 'now' of the analysis: the latest opening date in the extract.

        Using the data rather than the wall clock keeps the app correct on a
        historical extract.
        """
        return self.tickets[Col.OPENED].max()

    @property
    def has_llm_columns(self) -> bool:
        """Whether the LLM-inferred columns are actually populated."""
        return bool(self.tickets[Col.INCIDENT_TYPE].notna().any())

    def options(self, column: str) -> list[str]:
        """Sorted distinct non-empty values of a column (for select boxes)."""
        values = self.tickets[column].dropna().astype(str)
        return sorted(v for v in values.unique() if v.strip())

    def filter(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        selections: Mapping[str, Sequence[str]] | None = None,
    ) -> pd.DataFrame:
        """Return the tickets opened in ``[start, end]`` matching every selection.

        Args:
            start: First day included (inclusive, from midnight).
            end: Last day included (inclusive, until 23:59:59).
            selections: ``{column: allowed values}``. An empty list means "no
                restriction" for that column, which is what an untouched
                multiselect produces.
        """
        mask = pd.Series(True, index=self.tickets.index)
        opened = self.tickets[Col.OPENED]
        if start is not None:
            mask &= opened >= pd.Timestamp(start).normalize()
        if end is not None:
            mask &= opened < pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
        for column, allowed in (selections or {}).items():
            if allowed:
                mask &= self.tickets[column].astype(str).isin([str(v) for v in allowed])
        return self.tickets[mask]
