"""Runtime settings, read from environment variables (or a ``.env`` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: a .env file is a convenience, never a requirement
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

APP_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Everything configurable, with defaults that make the app run offline.

    Attributes:
        tickets_path: CSV/XLSX extract to load. ``None`` generates a synthetic one.
        cache_dir: Where computed embeddings are stored between runs.
        llm_base_url: Base URL of an OpenAI-compatible API (``.../v1``). Empty
            means *offline demo mode*: a local keyword heuristic replaces the LLM.
        llm_api_key: Bearer token for that API.
        llm_model: Model name sent in the request.
        llm_timeout: Seconds before a call is abandoned.
        llm_redact_pii: Mask emails, phones, IBANs and card numbers before any
            text leaves the machine.
        embedding_backend: ``auto`` (sentence-transformers if installed, else
            TF-IDF), ``sbert`` or ``tfidf``.
        embedding_model: Sentence-transformers model, multilingual by default.
        mock_tickets: Size of the synthetic extract.
    """

    tickets_path: Path | None = None
    cache_dir: Path = APP_ROOT / ".cache"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout: int = 30
    llm_redact_pii: bool = True
    embedding_backend: str = "auto"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    mock_tickets: int = 5000

    @classmethod
    def from_env(cls) -> Settings:
        """Build the settings from ``TICKETS_PATH``, ``LLM_*``, ``EMBEDDING_*``..."""
        path = os.getenv("TICKETS_PATH", "").strip()
        return cls(
            tickets_path=Path(path) if path else None,
            cache_dir=Path(os.getenv("CACHE_DIR", str(APP_ROOT / ".cache"))),
            llm_base_url=os.getenv("LLM_BASE_URL", "").strip(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            llm_timeout=int(os.getenv("LLM_TIMEOUT", "30")),
            llm_redact_pii=os.getenv("LLM_REDACT_PII", "true").lower() != "false",
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "auto").strip().lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            mock_tickets=int(os.getenv("MOCK_TICKETS", "5000")),
        )
