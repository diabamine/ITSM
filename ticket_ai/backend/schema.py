"""Column names, allowed values and small helpers shared by backend and frontend.

The extract uses French headers. They are declared **once**, here, and every
other module refers to ``Col.<NAME>`` instead of repeating a string: a renamed
column is then a one-line change.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


class Col:
    """Canonical column names, spelled exactly as in the source extract."""

    # --- native ITSM columns --------------------------------------------------
    ID = "Numéro"
    URGENCY = "Urgence"
    SERVICE = "Service"
    IMPACT = "Impact"
    PRIORITY = "Priority"
    STAGE = "Etape"
    OPENED = "Ouvert le"
    RESOLVED = "Résolu"
    CLOSED = "Fermé le"
    TITLE = "Description courte"
    DESCRIPTION = "Description"
    GROUP = "Groupe d'affectation"
    ENTITY = "Entité"
    RESOLUTION_HOURS = "Temps de resolution"
    SLA = "SLA"
    ELAPSED_HOURS = "Temps ecoulé"
    WORK_NOTES = "Notes de travail"
    COMMENT = "Commentaire additiionnel"
    RESOLUTION_CODE = "Code de resolution"
    CLOSE_NOTES = "Note de fermeture"
    REOPEN_COUNT = "Nombre de reouverture"
    CALLER_FOLLOWUPS = "Nombre de relance à l'appelant"
    SUPPORT_FOLLOWUPS = "Nombre de relances au support"
    CONTACT_TYPE = "Type de contact"
    TGI = "TGI"
    COMPANY = "Société"
    SITE = "Site"
    GBU = "GBU"
    # --- columns inferred by the LLM -------------------------------------------
    INCIDENT_TYPE = "IncidentType"
    TONE = "Tone"
    REQUEST_TYPE = "RequestType"
    SCOPE = "Scope"
    ROOT_CAUSE = "RootCause"
    RESOLUTION_SUMMARY = "ResolutionSummary"
    AFFECTED_SYSTEM = "AffectedSystem"


ALL_COLUMNS: list[str] = [v for k, v in vars(Col).items() if k.isupper()]
DATE_COLUMNS = [Col.OPENED, Col.RESOLVED, Col.CLOSED]
NUMERIC_COLUMNS = [
    Col.RESOLUTION_HOURS,
    Col.ELAPSED_HOURS,
    Col.REOPEN_COUNT,
    Col.CALLER_FOLLOWUPS,
    Col.SUPPORT_FOLLOWUPS,
]
LLM_COLUMNS = [
    Col.INCIDENT_TYPE,
    Col.TONE,
    Col.REQUEST_TYPE,
    Col.SCOPE,
    Col.ROOT_CAUSE,
    Col.RESOLUTION_SUMMARY,
    Col.AFFECTED_SYSTEM,
]
#: The minimum an extract needs for the app to be useful at all.
REQUIRED_COLUMNS = [Col.ID, Col.OPENED, Col.TITLE, Col.DESCRIPTION]

#: Categorical columns blended into the similarity score next to the text.
SIMILARITY_FEATURES = [Col.SERVICE, Col.GROUP, Col.INCIDENT_TYPE, Col.AFFECTED_SYSTEM, Col.SITE]

# --- allowed values ----------------------------------------------------------------
LEVELS = ("Low", "Medium", "High")  # Urgence, Impact
PRIORITIES = ("1 - Critical", "2 - High", "3 - Moderate", "4 - Low")
STAGES = ("Nouveau", "Affecté", "En cours", "En attente utilisateur", "Résolu", "Clos")
INCIDENT_TYPES = (
    "Réseau",
    "Accès & Comptes",
    "Poste de travail",
    "Messagerie",
    "Application métier",
    "Logiciel",
    "Impression",
    "Sécurité",
    "Stockage",
    "Autre",
)
REQUEST_TYPES = ("Incident", "Demande de service", "Demande d'information", "Changement")
TONES = ("Neutre", "Courtois", "Frustré", "Urgent", "Agressif")
SCOPES = ("Utilisateur unique", "Équipe", "Site", "Multi-sites")
NEGATIVE_TONES = frozenset({"Frustré", "Urgent", "Agressif"})
WIDE_SCOPES = frozenset({"Site", "Multi-sites"})

#: Resolution target in hours, per priority, used to derive the SLA status.
SLA_TARGET_HOURS = {"1 - Critical": 4.0, "2 - High": 8.0, "3 - Moderate": 24.0, "4 - Low": 72.0}


def derive_priority(impact: str, urgency: str) -> str:
    """Return the ITIL-style priority for an impact x urgency pair.

    Args:
        impact: One of :data:`LEVELS`.
        urgency: One of :data:`LEVELS`.

    Returns:
        One of :data:`PRIORITIES`. Unknown values count as ``Medium``.
    """
    rank = {level: index for index, level in enumerate(LEVELS)}
    score = rank.get(impact, 1) + rank.get(urgency, 1)
    return PRIORITIES[min(len(PRIORITIES) - 1, max(0, 4 - score))]


# --- loading helpers -----------------------------------------------------------------
def _key(name: object) -> str:
    """Reduce a header to lowercase ASCII letters and digits for fuzzy matching."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", ascii_only.lower())


_ALIASES = {_key(column): column for column in ALL_COLUMNS}
_ALIASES |= {
    _key("Commentaire additionnel"): Col.COMMENT,  # the extract's header is misspelled
    _key("Priorité"): Col.PRIORITY,
}


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename headers to their canonical spelling, ignoring accents, case and spacing.

    A real export rarely matches byte for byte (``Temps écoulé`` vs
    ``Temps ecoulé``, trailing spaces, non-breaking spaces...). Matching on a
    reduced key makes the loader forgiving without guessing.
    """
    mapping = {c: _ALIASES[_key(c)] for c in frame.columns if _key(c) in _ALIASES}
    return frame.rename(columns=mapping)


def coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse dates (day-first) and numbers (comma decimals) in place-safe fashion."""
    out = frame.copy()
    for column in DATE_COLUMNS:
        if column in out and not pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = pd.to_datetime(out[column], dayfirst=True, errors="coerce")
    for column in NUMERIC_COLUMNS:
        if column in out and not pd.api.types.is_numeric_dtype(out[column]):
            text = out[column].astype("string").str.replace(",", ".", regex=False)
            out[column] = pd.to_numeric(text, errors="coerce")
    return out


def fold(text: object) -> str:
    """Lowercase and strip accents, keeping spaces: ``"Réseau Lent"`` -> ``"reseau lent"``.

    Used wherever free text or labels are compared, so an accent or a capital
    never decides a match.
    """
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
