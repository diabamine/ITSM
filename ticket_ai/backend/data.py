"""Load the ticket extract (CSV or XLSX) and filter it.

This is the only file that knows the names of your columns and the labels the
alert rules look for. If your export differs, edit the constants below.
"""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"  # put your extract (CSV or XLSX) here

# --- columns of the extract ---------------------------------------------------------
ID, URGENCY, IMPACT, SERVICE = "Numéro", "Urgence", "Impact", "Service"
OPENED, RESOLVED = "Ouvert le", "Résolu"
TITLE, DESCRIPTION = "Description courte", "Description"
GROUP, ENTITY, SITE, CONTACT = "Groupe d'affectation", "Entité", "Site", "Type de contact"
RESOLUTION_H, SLA, REOPENED = "Temps de resolution", "SLA", "Nombre de reouverture"
# --- columns inferred by the LLM ----------------------------------------------------
INCIDENT_TYPE, TONE, REQUEST_TYPE, SCOPE = "IncidentType", "Tone", "RequestType", "Scope"
ROOT_CAUSE, RESOLUTION_SUMMARY, AFFECTED_SYSTEM = "RootCause", "ResolutionSummary", "AffectedSystem"
# --- labels the alert rules look for: adapt them to the values of your data ---------
SLA_BREACHED = "Dépassé"
NEGATIVE_TONES = {"Frustré", "Urgent", "Agressif"}
WIDE_SCOPES = {"Site", "Multi-sites"}
SECURITY = "Sécurité"
LEVELS = ["Low", "Medium", "High"]  # order of Urgence / Impact

# fmt: off
COLUMNS = [ID, URGENCY, IMPACT, SERVICE, OPENED, RESOLVED, TITLE, DESCRIPTION, GROUP, ENTITY, SITE, CONTACT,
           RESOLUTION_H, SLA, REOPENED, INCIDENT_TYPE, TONE, REQUEST_TYPE, SCOPE, ROOT_CAUSE,
           RESOLUTION_SUMMARY, AFFECTED_SYSTEM]
# fmt: on


def _key(name: object) -> str:
    """Reduce a header to lowercase letters and digits: ``"Temps écoulé "`` -> ``"tempsecoule"``."""
    ascii_only = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_only.lower())


def find_extract() -> Path | None:
    """The extract stored in ``DATA_DIR``: the first ``.csv`` or ``.xlsx`` file, or ``None`` if there is none."""
    files = sorted(p for p in DATA_DIR.glob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"} and not p.name.startswith("~$"))
    return files[0] if files else None


def load(path: Path) -> pd.DataFrame:
    """Read an extract and bring it to the column names above.

    Headers are matched ignoring accents, case and spacing. CSV separators and
    encodings are detected (French exports are usually ``;`` and Windows-1252).
    Columns missing from the file are added empty so the pages never fail on them.

    Raises:
        ValueError: if the opening date, title or description column is missing.
    """
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    else:
        content = path.read_bytes()
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("cp1252")
        raw = pd.read_csv(io.StringIO(text), sep=None, engine="python")

    known = {_key(column): column for column in COLUMNS}
    df = raw.rename(columns=lambda column: known.get(_key(column), column))
    missing = [c for c in (OPENED, TITLE, DESCRIPTION) if c not in df]
    if missing:
        raise ValueError(f"The extract has no column {missing}. Found: {list(raw.columns)}")

    df = df.reindex(columns=[*df.columns, *[c for c in COLUMNS if c not in df]])
    for column in (OPENED, RESOLVED):
        df[column] = pd.to_datetime(df[column], dayfirst=True, errors="coerce")
    for column in (RESOLUTION_H, REOPENED):  # accept decimal commas
        df[column] = pd.to_numeric(df[column].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df.dropna(subset=[OPENED]).sort_values(OPENED).reset_index(drop=True)


def options(df: pd.DataFrame, column: str, order: list[str] = ()) -> list[str]:  # type: ignore[assignment]
    """Distinct non-empty values of a column, for a select box. ``order`` lists values to put first."""
    values = {v for v in df[column].dropna().astype(str) if v.strip()}
    return sorted(values, key=lambda v: (order.index(v) if v in order else len(order), v))


def select(df: pd.DataFrame, start=None, end=None, filters: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Tickets opened from ``start`` to ``end`` (both days included) matching every filter.

    An empty list for a column means "no restriction", which is what an untouched
    multiselect returns.
    """
    keep = pd.Series(True, index=df.index)
    if start is not None:
        keep &= df[OPENED] >= pd.Timestamp(start)
    if end is not None:
        keep &= df[OPENED] < pd.Timestamp(end) + pd.Timedelta(days=1)
    for column, allowed in (filters or {}).items():
        if allowed:
            keep &= df[column].astype(str).isin(allowed)
    return df[keep]
