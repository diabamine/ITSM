"""Frontend smoke tests: every page renders, and the ticket form works end to end.

Each page is run the way Streamlit runs it (as a script, through ``AppTest``)
against the offline configuration: synthetic data, TF-IDF, no LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _page(tmp_path: Path, module: str) -> AppTest:
    """An ``AppTest`` that renders one page module."""
    script = tmp_path / f"{module}.py"
    script.write_text(
        f"import sys\nsys.path.insert(0, {str(ROOT)!r})\nfrom frontend.views import {module}\n{module}.render()\n",
        encoding="utf-8",
    )
    return AppTest.from_file(str(script), default_timeout=120)


@pytest.mark.parametrize("module", ["ticket_form", "batch_report", "dashboard"])
def test_page_renders_without_exception(offline_env, tmp_path, module):
    app = _page(tmp_path, module).run()
    assert not app.exception, [e.value for e in app.exception]


def test_submitting_a_ticket_shows_analysis_alerts_and_similar_tickets(offline_env, tmp_path):
    app = _page(tmp_path, "ticket_form").run()
    app.text_input[0].set_value("VPN qui se déconnecte")
    app.text_area[0].set_value("Le VPN se coupe sans arrêt, c'est la troisième fois, c'est urgent. Tel 06 12 34 56 78")
    app.button[0].click().run()

    assert not app.exception, [e.value for e in app.exception]
    headings = [h.value for h in app.subheader]
    assert headings == ["AI analysis", "Alerts", "Similar past tickets"]
    assert any(
        "demo mode" in w.value.lower() or "no llm" in w.value.lower() for w in app.warning
    )  # honest about offline mode
    assert any("Personal data" in str(m.value) for m in app.info)  # the phone number was flagged
    assert len(app.dataframe) == 1


def test_an_empty_form_is_refused(offline_env, tmp_path):
    app = _page(tmp_path, "ticket_form").run()
    app.button[0].click().run()
    assert [e.value for e in app.error] == ["Short description and description are required."]


def test_report_can_be_generated_and_previewed(offline_env, tmp_path):
    app = _page(tmp_path, "batch_report").run()
    generate = next(b for b in app.button if b.label == "Generate report")
    generate.click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert "report" in app.session_state
    assert "Key figures" in app.session_state["report"].to_html()
