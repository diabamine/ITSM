"""Persistence of trained models.

Deliberately file-based: a directory per artifact, holding the serialised
estimator and a JSON sidecar with everything needed to explain where a
prediction came from (training date, dataset, parameters, evaluation metrics).
That sidecar is what the API exposes on ``/v1/models`` - a model whose scores
cannot be inspected at runtime should not be in production.

Swapping this for MLflow or a model registry means reimplementing
:class:`ArtifactStore`; nothing else in the codebase touches the filesystem.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from itsm.config import get_settings
from itsm.logging import get_logger

logger = get_logger("models.artifacts")

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class ArtifactMetadata:
    """Provenance and quality record attached to a stored model."""

    name: str
    kind: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    dataset: str = "synthetic"
    n_train: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> str:
        """Serialise the metadata."""
        return json.dumps(asdict(self), indent=2, default=str)


class ArtifactStore:
    """Read/write trained models under ``artifacts/``.

    Args:
        root: Base directory; defaults to the configured artifacts directory.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_settings().artifacts_dir

    def path(self, name: str) -> Path:
        """Return the directory holding artifact ``name``."""
        return self.root / name

    def exists(self, name: str) -> bool:
        """Return whether a usable artifact is stored under ``name``."""
        return (self.path(name) / MODEL_FILE).exists()

    def save(self, name: str, model: Any, metadata: ArtifactMetadata) -> Path:
        """Persist a model and its metadata, returning the artifact directory."""
        directory = self.path(name)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, directory / MODEL_FILE)
        (directory / METADATA_FILE).write_text(metadata.to_json(), encoding="utf-8")
        logger.info("saved artifact %s to %s", name, directory)
        return directory

    def load(self, name: str) -> Any:
        """Load a model. Raises ``FileNotFoundError`` if it was never trained."""
        target = self.path(name) / MODEL_FILE
        if not target.exists():
            raise FileNotFoundError(
                f"artifact {name!r} not found in {self.root}. Train it first: `itsm train {name}`"
            )
        return joblib.load(target)

    def load_metadata(self, name: str) -> dict[str, Any]:
        """Return the metadata sidecar, or an empty dict if absent."""
        target = self.path(name) / METADATA_FILE
        return json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}

    def list_artifacts(self) -> list[dict[str, Any]]:
        """List every stored artifact with its metadata."""
        if not self.root.exists():
            return []
        return [
            {"name": directory.name, **self.load_metadata(directory.name)}
            for directory in sorted(self.root.iterdir())
            if (directory / MODEL_FILE).exists()
        ]
