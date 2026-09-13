"""Application settings.

A single :class:`Settings` object is the only place where paths, dataset choice
and model backends are configured. Everything is overridable through
environment variables prefixed with ``ITSM_`` or a local ``.env`` file, so the
library, the CLI and the API always agree on the same configuration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DatasetName = Literal["synthetic", "csv", "huggingface", "uci_incident_log"]
EmbeddingBackend = Literal["tfidf", "embeddings"]
SentimentBackend = Literal["lexicon", "transformer"]


class Settings(BaseSettings):
    """Runtime configuration, loaded from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="ITSM_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    # --- paths -----------------------------------------------------------
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")

    # --- dataset ---------------------------------------------------------
    dataset: DatasetName = "synthetic"
    csv_path: Path | None = None
    hf_dataset_id: str = "Tobi-Bueck/customer-support-tickets"
    synthetic_n_tickets: int = 8000
    synthetic_days: int = 365
    synthetic_seed: int = 42

    # --- text / model backends ------------------------------------------
    embedding_backend: EmbeddingBackend = "tfidf"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    sentiment_backend: SentimentBackend = "lexicon"
    sentiment_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    # --- API -------------------------------------------------------------
    api_title: str = "ITSM Insights API"
    api_version: str = "0.1.0"
    cors_origins: list[str] = Field(default_factory=list)
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create the working directories if they do not exist yet."""
        for path in (self.data_dir, self.artifacts_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()
