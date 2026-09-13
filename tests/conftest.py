"""Shared fixtures.

Every test runs against a small synthetic dataset in a temporary directory, so
the suite is fast, hermetic and independent from whatever is cached in the
developer's ``data/`` folder.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

N_TICKETS = 1800
N_DAYS = 150


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    """Session settings pointing at a temporary workspace."""
    from itsm.config import get_settings

    root = tmp_path_factory.mktemp("itsm")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ITSM_DATA_DIR", str(root / "data"))
        patch.setenv("ITSM_ARTIFACTS_DIR", str(root / "artifacts"))
        patch.setenv("ITSM_REPORTS_DIR", str(root / "reports"))
        patch.setenv("ITSM_DATASET", "synthetic")
        patch.setenv("ITSM_SYNTHETIC_N_TICKETS", str(N_TICKETS))
        patch.setenv("ITSM_SYNTHETIC_DAYS", str(N_DAYS))
        get_settings.cache_clear()
        current = get_settings()
        current.ensure_dirs()
        yield current
        get_settings.cache_clear()


@pytest.fixture(scope="session")
def dataset(settings: object) -> object:
    """A generated synthetic dataset, shared by the whole session."""
    from itsm.data import synthetic

    return synthetic.generate(n_tickets=N_TICKETS, days=N_DAYS, seed=7)


@pytest.fixture(scope="session")
def tickets(dataset: object) -> pd.DataFrame:
    """Ticket table of the session dataset."""
    from itsm.data.schema import validate_tickets

    return validate_tickets(dataset.tickets)  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def events(dataset: object) -> pd.DataFrame:
    """Event table of the session dataset."""
    from itsm.data.schema import validate_events

    return validate_events(dataset.events)  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def trained_workspace(settings: object) -> Path:
    """Build the dataset and train every artifact the API needs."""
    from itsm.pipelines.run_analysis import run_designation_analysis
    from itsm.pipelines.train_classifier import run as train_classifier

    train_classifier(settings, compare_baselines=False)  # type: ignore[arg-type]
    run_designation_analysis(settings, n_clusters=10)  # type: ignore[arg-type]
    return Path(settings.artifacts_dir)  # type: ignore[attr-defined]
