"""Streamlit layer: the Plotly builders, the helpers, and every page end to end.

The builders are pure functions returning ``go.Figure``, so they are asserted
on directly - trace count, trace types and the encoded data - which is where a
silent breakage would otherwise go unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from itsm.analysis.lifecycle import aging_backlog, backlog_timeline, volume_timeline
from itsm.app import charts
from itsm.app.views.triage import _priority_from
from itsm.data.schema import Priority


@pytest.fixture
def diagnostics() -> pd.DataFrame:
    """A ten-day diagnostic frame as the detector produces it."""
    return pd.DataFrame(
        {
            "period": pd.date_range("2026-01-01", periods=10, freq="D"),
            "observed": range(10),
            "expected": [5.0] * 10,
            "scale": [1.5] * 10,
            "robust_z": [0.0] * 10,
            "ewma": [0.0] * 10,
        }
    )


def test_volume_line_is_a_time_series(tickets: pd.DataFrame) -> None:
    figure = charts.volume_line(volume_timeline(tickets, freq="W"))
    assert isinstance(figure, go.Figure)
    assert figure.data[0].type in {"scatter", "scattergl"}
    assert figure.layout.hovermode == "x unified"
    # Zoom presets are what makes a year of daily points readable.
    assert figure.layout.xaxis.rangeselector.buttons


def test_volume_line_splits_by_category(tickets: pd.DataFrame) -> None:
    split = volume_timeline(tickets, freq="ME", by="category")
    figure = charts.volume_line(split, color="category")
    assert len(figure.data) == tickets["category"].nunique()


def test_backlog_chart_pairs_bars_with_a_backlog_line(tickets: pd.DataFrame) -> None:
    figure = charts.backlog_area(backlog_timeline(tickets, freq="W"))
    kinds = [trace.type for trace in figure.data]
    assert kinds.count("bar") == 2
    assert kinds.count("scatter") == 1
    assert figure.layout.barmode == "group"


def test_category_bars_are_sorted_ascending(tickets: pd.DataFrame) -> None:
    figure = charts.category_bars(aging_backlog(tickets), value="tickets", label="age_bucket")
    values = list(figure.data[0].x)
    assert values == sorted(values)


def test_anomaly_chart_layers(diagnostics: pd.DataFrame) -> None:
    without_alerts = charts.anomaly_chart(diagnostics, pd.DataFrame())
    assert [trace.name for trace in without_alerts.data] == ["expected", "observed"]

    with_band = charts.anomaly_chart(diagnostics, pd.DataFrame(), z_spike=3.5)
    threshold = with_band.data[0]
    assert "threshold" in threshold.name
    # expected + z * scale = 5.0 + 3.5 * 1.5
    assert threshold.y[0] == pytest.approx(10.25)

    alerts = pd.DataFrame(
        {
            "period": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-07")],
            "observed": [40, 30],
            "expected": [5.0, 5.0],
            "kind": ["spike", "emerging"],
            "severity": [6.0, 3.0],
        }
    )
    full = charts.anomaly_chart(diagnostics, alerts, z_spike=3.5)
    assert [trace.name for trace in full.data] == [
        "spike threshold (z=3.5)",
        "expected",
        "observed",
        "spike",
        "emerging",
    ]
    assert full.layout.xaxis.rangeslider.visible


def test_residual_chart_draws_the_threshold(diagnostics: pd.DataFrame) -> None:
    figure = charts.residual_chart(diagnostics, ewma_threshold=1.1)
    assert figure.data[0].fill == "tozeroy"
    assert figure.layout.shapes[0].y0 == 1.1


def test_probability_bars_span_the_full_range() -> None:
    frame = pd.DataFrame({"label": ["Network & VPN", "Access"], "score": [0.9, 0.1]})
    figure = charts.probability_bars(frame)
    assert figure.layout.xaxis.range == (0, 1.15)
    assert list(figure.data[0].y) == ["Access", "Network & VPN"]


def test_categorical_scale_greys_out_noise() -> None:
    values = pd.Series(["a", "b", "noise", "a"])
    scale = charts.categorical_scale(values, noise_value="noise")
    assert scale["noise"] == charts.COLOR_NOISE
    assert len({scale["a"], scale["b"]}) == 2
    assert charts.COLOR_NOISE not in {scale["a"], scale["b"]}


def test_embedding_scatter_hides_its_axes() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 0.5],
            "cluster": ["0 - vpn", "0 - vpn", "noise"],
            "title": ["a", "b", "c"],
        }
    )
    scale = charts.categorical_scale(frame["cluster"], noise_value="noise")
    figure = charts.embedding_scatter(
        frame, color="cluster", color_map=scale, hover_columns=["title"]
    )
    assert len(figure.data) == 2
    assert figure.layout.xaxis.visible is False
    assert figure.layout.yaxis.scaleanchor == "x"


def test_purity_heatmap_is_normalised() -> None:
    frame = pd.DataFrame(
        {
            "cluster": [0, 0, 1],
            "category": ["Network", "Access", "Network"],
            "share": [0.7, 0.3, 1.0],
            "tickets": [7, 3, 10],
        }
    )
    figure = charts.purity_heatmap(frame)
    assert figure.data[0].type == "heatmap"
    assert (figure.data[0].zmin, figure.data[0].zmax) == (0, 1)


@pytest.mark.parametrize(
    ("impact", "urgency", "expected"),
    [(1, 1, Priority.P1), (1, 2, Priority.P2), (2, 2, Priority.P3), (3, 3, Priority.P4)],
)
def test_priority_matrix(impact: int, urgency: int, expected: Priority) -> None:
    assert _priority_from(impact, urgency) is expected


@pytest.mark.parametrize("page", ["overview", "families", "signals", "triage"])
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


@pytest.mark.slow
def test_embedding_map_page_renders(trained_workspace: object, tmp_path: Path) -> None:
    """The 2D map page, which fits UMAP and HDBSCAN live.

    Marked slow: the projection is refitted for the test's sample rather than
    read from an artifact.
    """
    from streamlit.testing.v1 import AppTest

    script = tmp_path / "embedding_map_page.py"
    script.write_text(
        "from itsm.app.views import embedding_map\nembedding_map.render()\n", encoding="utf-8"
    )
    app = AppTest.from_file(str(script), default_timeout=600).run()
    assert not app.exception, [error.value for error in app.exception]
