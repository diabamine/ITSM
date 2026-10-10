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
AnonymisationBackend = Literal["rules", "transformer", "presidio", "gliner"]


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

    # --- privacy ----------------------------------------------------------
    anonymisation_backend: AnonymisationBackend = "rules"
    ner_model: str = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    #: Language the Presidio analyser is configured for. One engine serves one
    #: language, so on a bilingual corpus this is a measurable trade, not a
    #: formality - see ``docs/07-privacy.md``.
    presidio_language: str = "en"
    #: spaCy pipeline for that language. ``None`` uses the default known for
    #: the language; set it when that pipeline is not the one installed.
    presidio_spacy_model: str | None = None
    #: Zero-shot GLiNER checkpoint. Multilingual and PII-tuned by default,
    #: which is what an FR/EN ticket corpus needs.
    gliner_model: str = "urchade/gliner_multi_pii-v1"
    #: When true, every pipeline reads the anonymised extract instead of the
    #: raw one. That is the switch a production deployment leaves on.
    use_anonymised: bool = False

    # --- API -------------------------------------------------------------
    api_title: str = "ITSM Insights API"
    api_version: str = "0.1.0"
    cors_origins: list[str] = Field(default_factory=list)
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create the working directories if they do not exist yet."""
        for path in (self.data_dir, self.artifacts_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)

    def pii_model(self, backend: AnonymisationBackend) -> str | None:
        """Return the model override for an anonymisation backend.

        Each backend loads a different kind of model - a Hugging Face NER
        checkpoint, a spaCy pipeline, or nothing at all - so handing them a
        single ``model_name`` would pass a transformer id to spaCy. This keeps
        the choice in one place instead of at every call site.
        """
        if backend == "transformer":
            return self.ner_model
        if backend == "presidio":
            return self.presidio_spacy_model
        if backend == "gliner":
            return self.gliner_model
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()
