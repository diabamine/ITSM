"""LLM access: structured extraction on one ticket, and a short narrative on a batch.

Three pieces, each small:

* :class:`LLMClient` - a thin HTTP client for any OpenAI-compatible chat API
  (OpenAI, an Azure/enterprise gateway, vLLM, Ollama...). It is the **only** place
  that knows about HTTP; to use another provider, change ``LLMClient.complete``.
* :class:`OfflineHeuristic` - a keyword fallback. It makes the demo run with no
  API key and keeps the app usable when the API is down. It is deliberately
  modest and the UI always says when it was used instead of the LLM.
* :class:`TicketEnricher` - asks the LLM, validates what comes back against the
  allowed values, and falls back to the heuristic on any failure.

Only the ticket text, masked by :func:`backend.privacy.redact_pii`, is sent to
the API. The batch narrative sends **aggregated numbers only**, never tickets.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import requests

from backend.config import Settings
from backend.privacy import redact_pii
from backend.schema import (
    INCIDENT_TYPES,
    REQUEST_TYPES,
    SCOPES,
    TONES,
    Col,
    fold,
)


@dataclass
class Enrichment:
    """What the LLM (or the fallback) inferred about a ticket.

    The first six fields are the LLM-inferred columns of the extract, so a new
    ticket is described in exactly the same terms as the history it is compared to.
    """

    incident_type: str = "Autre"
    request_type: str = "Incident"
    tone: str = "Neutre"
    scope: str = "Utilisateur unique"
    affected_system: str = "Non identifié"
    root_cause: str = "À déterminer"
    source: str = "offline"  # "llm" or "offline"
    notice: str = ""  # why the fallback was used, when it was
    pii_found: list[str] = field(default_factory=list)

    def as_columns(self) -> dict[str, str]:
        """Return the inferred values keyed by the extract's column names."""
        return {
            Col.INCIDENT_TYPE: self.incident_type,
            Col.REQUEST_TYPE: self.request_type,
            Col.TONE: self.tone,
            Col.SCOPE: self.scope,
            Col.AFFECTED_SYSTEM: self.affected_system,
            Col.ROOT_CAUSE: self.root_cause,
        }


def choose(value: object, allowed: tuple[str, ...], default: str) -> str:
    """Map a model answer onto the allowed values, ignoring case and accents."""
    wanted = fold(value).strip()
    return next((a for a in allowed if fold(a) == wanted), default)


# --------------------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------------------
class LLMClient:
    """Minimal client for an OpenAI-compatible ``/chat/completions`` endpoint.

    Args:
        base_url: API root, e.g. ``https://api.openai.com/v1``.
        api_key: Sent as a Bearer token when not empty.
        model: Model name.
        timeout: Seconds before a call is abandoned.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMClient | None:
        """Return a client, or ``None`` when no API is configured (offline mode)."""
        if not (settings.llm_base_url and settings.llm_model):
            return None
        return cls(settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout)

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        """Send one chat request and return the assistant's text.

        Raises:
            requests.RequestException: on network errors and HTTP error statuses.
        """
        payload: dict = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = self._post(payload)
        if response.status_code == 400 and json_mode:
            # Some servers reject response_format: retry once without it.
            payload.pop("response_format")
            response = self._post(payload)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])

    def complete_json(self, system: str, user: str) -> dict:
        """Like :meth:`complete`, but parse the answer as a JSON object."""
        return parse_json_object(self.complete(system, user, json_mode=True))

    def _post(self, payload: dict) -> requests.Response:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)


def parse_json_object(text: str) -> dict:
    """Extract the first JSON object from a model answer.

    Models wrap JSON in code fences or add a sentence around it, even when asked
    not to. Taking everything between the first ``{`` and the last ``}`` is
    tolerant of both.

    Raises:
        ValueError: if no JSON object can be found or parsed.
    """
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the model answer")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("the model answer is not a JSON object")
    return parsed


# --------------------------------------------------------------------------------------
# Offline fallback
# --------------------------------------------------------------------------------------
_INCIDENT_KEYWORDS = {
    "Réseau": ["vpn", "wifi", "wi-fi", "reseau", "latence", "bande passante"],
    "Accès & Comptes": [
        "mot de passe",
        "password",
        "verrouille",
        "mfa",
        "double authentification",
        "droits",
        "habilitation",
        "acces au",
        "role ",
    ],
    "Poste de travail": [
        "portable",
        "ordinateur",
        "ecran",
        "station d'accueil",
        "batterie",
        "clavier",
        "souris",
        "demarre",
    ],
    "Messagerie": ["outlook", "teams", "mail", "courriel", "calendrier", "boite aux lettres", "reunion"],
    "Application métier": ["sap", "plm", "portail rh", "erp", "note de frais"],
    "Logiciel": ["licence", "installation", "installer", "mise a jour", "logiciel"],
    "Impression": ["imprim", "copieur", "scanner", "toner"],
    "Sécurité": ["phishing", "antivirus", "malware", "virus", "certificat", "suspect"],
    "Stockage": ["onedrive", "partage", "lecteur", "sauvegarde", "restaur", "quota"],
}
_SYSTEMS = [
    ("vpn", "VPN"),
    ("wi-fi", "Wi-Fi"),
    ("wifi", "Wi-Fi"),
    ("outlook", "Outlook"),
    ("teams", "Teams"),
    ("sap", "SAP"),
    ("onedrive", "OneDrive"),
    ("plm", "PLM"),
    ("portail rh", "Portail RH"),
    ("imprimante", "Imprimante"),
    ("copieur", "Imprimante"),
    ("antivirus", "Antivirus"),
    ("certificat", "Certificat"),
    ("mfa", "MFA"),
    ("windows", "Windows"),
    ("portable", "Ordinateur portable"),
    ("calendrier", "Calendrier"),
]
_TONE_KEYWORDS = [
    ("Agressif", ["inadmissible", "inacceptable", "scandaleux", "lamentable", "honteux"]),
    (
        "Frustré",
        [
            "troisieme fois",
            "toujours rien",
            "toujours pas",
            "relance",
            "je suis bloque",
            "personne ne",
            "depuis des jours",
        ],
    ),
    ("Urgent", ["urgent", "en priorite", "asap", "immediat", "production est impactee", "livraison client"]),
    ("Courtois", ["merci", "cordialement", "s'il vous plait"]),
]
_SCOPE_KEYWORDS = [
    ("Multi-sites", ["plusieurs sites", "tous les sites", "toute l'entreprise", "all sites"]),
    ("Site", ["tout le site", "tout le batiment", "tous les utilisateurs", "toute l'usine"]),
    ("Équipe", ["mon equipe", "l'equipe", "collegues", "plateau", "plusieurs utilisateurs"]),
]
_CAUSE_RULES = [
    (["apres la mise a jour", "depuis la mise a jour", "derniere mise a jour"], "Mise à jour récente"),
    (["expire"], "Expiration (licence, certificat ou mot de passe)"),
    (["mot de passe", "verrouille"], "Compte verrouillé ou mot de passe oublié"),
    (["nouveau telephone", "nouveau poste", "nouveau portable"], "Changement de matériel"),
]


class OfflineHeuristic:
    """Keyword-based stand-in for the LLM, returning the same :class:`Enrichment`."""

    def enrich(self, title: str, description: str) -> Enrichment:
        """Infer the six fields from keywords in the (accent-folded) text."""
        text = fold(f"{title} {description}")
        scores = {t: sum(text.count(k) for k in keys) for t, keys in _INCIDENT_KEYWORDS.items()}
        best = max(scores, key=lambda t: scores[t])

        found = [(text.find(k), label) for k, label in _SYSTEMS if k in text]
        if text.lstrip().startswith(("comment ", "je cherche comment")) or "how do i" in text:
            request = "Demande d'information"
        elif any(k in text for k in ("installer", "installation", "demande d'acces", "ajouter le role")):
            request = "Demande de service"
        else:
            request = "Incident"

        return Enrichment(
            incident_type=best if scores[best] else "Autre",
            request_type=request,
            tone=next((t for t, keys in _TONE_KEYWORDS if any(k in text for k in keys)), "Neutre"),
            scope=next((s for s, keys in _SCOPE_KEYWORDS if any(k in text for k in keys)), "Utilisateur unique"),
            affected_system=min(found)[1] if found else "Non identifié",
            root_cause=next((c for keys, c in _CAUSE_RULES if any(k in text for k in keys)), "À déterminer"),
            source="offline",
        )


# --------------------------------------------------------------------------------------
# Ticket enrichment
# --------------------------------------------------------------------------------------
_EXTRACTION_SYSTEM = f"""You are an IT service-desk analyst. Read the ticket and fill the fields below.
Answer with ONE JSON object and nothing else.

Fields:
- incident_type: one of {list(INCIDENT_TYPES)}
- request_type: one of {list(REQUEST_TYPES)}
- tone: one of {list(TONES)} (the tone of the requester's wording)
- scope: one of {list(SCOPES)} (how many people are affected according to the text)
- affected_system: short name of the application or equipment (e.g. VPN, Outlook, SAP), or "Non identifié"
- root_cause: your best hypothesis in at most 8 words, in French, or "À déterminer"

Use only what the ticket supports; do not invent facts. The ticket may be in French or English."""


class TicketEnricher:
    """Ask the LLM about a ticket, validate the answer, fall back when needed.

    Args:
        client: The LLM client, or ``None`` for offline mode.
        redact: Mask formatted personal data before the text leaves the machine.
    """

    def __init__(self, client: LLMClient | None, redact: bool = True) -> None:
        self.client = client
        self.redact = redact
        self._fallback = OfflineHeuristic()

    @classmethod
    def from_settings(cls, settings: Settings) -> TicketEnricher:
        """Build an enricher from the application settings."""
        return cls(LLMClient.from_settings(settings), redact=settings.llm_redact_pii)

    @property
    def mode(self) -> str:
        """``"llm"`` when an API is configured, ``"offline"`` otherwise."""
        return "llm" if self.client else "offline"

    def enrich(self, ticket: Mapping[str, str]) -> Enrichment:
        """Infer the LLM columns for one ticket.

        Args:
            ticket: At least ``Description courte`` and ``Description``; ``Service``,
                ``Urgence`` and ``Impact`` are passed to the model as context when present.
        """
        title, found_title = redact_pii(str(ticket.get(Col.TITLE, "")))
        description, found_body = redact_pii(str(ticket.get(Col.DESCRIPTION, "")))
        pii = list(dict.fromkeys(found_title + found_body))
        if not self.redact:
            title, description = str(ticket.get(Col.TITLE, "")), str(ticket.get(Col.DESCRIPTION, ""))

        if self.client is None:
            result = self._fallback.enrich(title, description)
            result.notice = "No LLM configured: local keyword heuristic used (demo mode)."
        else:
            try:
                result = self._ask(title, description, ticket)
            except (requests.RequestException, ValueError, KeyError) as error:
                result = self._fallback.enrich(title, description)
                result.notice = f"LLM call failed ({type(error).__name__}): heuristic used instead."
        result.pii_found = pii
        return result

    def _ask(self, title: str, description: str, ticket: Mapping[str, str]) -> Enrichment:
        assert self.client is not None
        context = ", ".join(f"{c}: {ticket[c]}" for c in (Col.SERVICE, Col.URGENCY, Col.IMPACT) if ticket.get(c))
        user = f"Title: {title}\nDescription: {description}\nContext: {context}"
        data = self.client.complete_json(_EXTRACTION_SYSTEM, user)
        return Enrichment(
            incident_type=choose(data.get("incident_type"), INCIDENT_TYPES, "Autre"),
            request_type=choose(data.get("request_type"), REQUEST_TYPES, "Incident"),
            tone=choose(data.get("tone"), TONES, "Neutre"),
            scope=choose(data.get("scope"), SCOPES, "Utilisateur unique"),
            affected_system=str(data.get("affected_system") or "Non identifié")[:60],
            root_cause=str(data.get("root_cause") or "À déterminer")[:80],
            source="llm",
        )


# --------------------------------------------------------------------------------------
# Batch narrative
# --------------------------------------------------------------------------------------
def summarise_batch(client: LLMClient | None, facts: Mapping[str, object]) -> tuple[str, str]:
    """Write a short executive summary of a batch from **aggregated facts only**.

    Args:
        client: LLM client, or ``None`` for the template fallback.
        facts: Numbers and labels computed by the backend (no ticket text).

    Returns:
        ``(summary, source)`` where source is ``"llm"`` or ``"offline"``.
    """
    if client is not None:
        system = (
            "You write executive summaries for an IT service-desk manager. Using ONLY the JSON "
            "facts provided, write 4 to 6 sentences in English, no bullet points. Mention the "
            "most important alert first, then the trend, then what to watch. Do not invent numbers."
        )
        try:
            return client.complete(system, json.dumps(facts, ensure_ascii=False, default=str)).strip(), "llm"
        except (requests.RequestException, ValueError, KeyError):
            pass
    return _template_summary(facts), "offline"


def _template_summary(facts: Mapping[str, object]) -> str:
    """Deterministic summary used when no LLM is available."""
    parts = [f"{facts.get('tickets', 0)} tickets were opened over {facts.get('period', 'the period')}."]
    change = facts.get("change_vs_previous_pct")
    if isinstance(change, (int, float)):
        direction = "up" if change >= 0 else "down"
        parts.append(f"Volume is {direction} {abs(change):.0f}% compared with the previous period.")
    alerts = facts.get("alerts") or []
    if alerts:
        parts.append("Main alert: " + str(alerts[0]) + ".")  # type: ignore[index]
    else:
        parts.append("No alert was raised on this batch.")
    if facts.get("sla_breach_pct") is not None:
        parts.append(f"{facts['sla_breach_pct']:.0f}% of resolved tickets missed their SLA.")
    return " ".join(parts)
