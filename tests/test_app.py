"""Streamlit layer: the pure parts.

The pages themselves are exercised end-to-end with ``streamlit.testing`` only
when artifacts exist; what is tested unconditionally here is everything that
does not need a Streamlit runtime - the chart builders and the helpers - since
that is where a silent breakage would go unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from itsm.analysis.lifecycle import backlog_timeline, volume_timeline
from itsm.app import charts
from itsm.app.views.triage import _priority_from
from itsm.data.schema import Priority


def test_volume_and_backlog_charts_build(tickets: pd.DataFrame) -> None:
    volume = volume_timeline(tickets, freq="W")
    assert charts.volume_line(volume).to_dict()["mark"]["type"] == "line"
    assert charts.backlog_area(backlog_timeline(tickets, freq="W")).to_dict()["layer"]


def test_anomaly_chart_handles_empty_alerts() -> None:
    diagnostics = pd.DataFrame(
        {
            "period": pd.date_range("2026-01-01", periods=10, freq="D"),
            "observed": range(10),
            "expected": [5.0] * 10,
            "robust_z": [0.0] * 10,
            "ewma": [0.0] * 10,
        }
    )
    chart = charts.anomaly_chart(diagnostics, pd.DataFrame())
    assert len(chart.to_dict()["layer"]) == 2

    alerts = pd.DataFrame(
        {
            "period": [pd.Timestamp("2026-01-05")],
            "observed": [40],
            "expected": [5.0],
            "kind": ["spike"],
            "severity": [6.0],
        }
    )
    assert len(charts.anomaly_chart(diagnostics, alerts).to_dict()["layer"]) == 3


@pytest.mark.parametrize(
    ("impact", "urgency", "expected"),
    [(1, 1, Priority.P1), (1, 2, Priority.P2), (2, 2, Priority.P3), (3, 3, Priority.P4)],
)
def test_priority_matrix(impact: int, urgency: int, expected: Priority) -> None:
    assert _priority_from(impact, urgency) is expected


@pytest.mark.parametrize("page", ["overview", "families", "signals", "triage", "privacy"])
def test_pages_render_without_exception(
    trained_workspace: object, tmp_path: Path, page: str
) -> None:
    """Each page must run end to end against a trained workspace.

    Streamlit executes a page as a script, so the test does the same: a one
    line entry point per page, run through ``AppTest``. A broken widget call or
    a bad dataframe access then fails here rather than in front of a user.
    """
    from streamlit.testing.v1 import AppTest

    script = tmp_path / f"{page}_page.py"
    script.write_text(f"from itsm.app.views import {page}\n{page}.render()\n", encoding="utf-8")
    app = AppTest.from_file(str(script), default_timeout=180).run()
    assert not app.exception, [error.value for error in app.exception]
