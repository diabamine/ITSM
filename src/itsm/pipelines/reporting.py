"""Report writing shared by the pipelines.

Every pipeline returns a plain dictionary *and* persists it under
``reports/``. The dictionary feeds the API and the tests; the file is the
audit trail that lets a result be reread months later without rerunning the
training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from itsm.config import Settings, get_settings
from itsm.logging import get_logger

logger = get_logger("pipelines.reporting")


def _to_jsonable(value: Any) -> Any:
    """Convert numpy / pandas objects into JSON-serialisable primitives."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def write_report(name: str, payload: dict[str, Any], settings: Settings | None = None) -> Path:
    """Persist a pipeline report as JSON and return its path."""
    settings = settings or get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    path = settings.reports_dir / f"{name}.json"
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, default=str), encoding="utf-8")
    logger.info("wrote report %s", path)
    return path


def read_report(name: str, settings: Settings | None = None) -> dict[str, Any] | None:
    """Return a previously written report, or ``None`` if absent."""
    settings = settings or get_settings()
    path = settings.reports_dir / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
