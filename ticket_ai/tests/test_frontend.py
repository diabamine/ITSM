"""Frontend tests: each page renders from an extract, and the page-1 hooks work end to end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from backend import ai
from backend import data as d

ROOT = Path(__file__).resolve().parents[1]


def page(csv_path: str, name: str) -> AppTest:
    """Run one page function of ``frontend/app.py`` on the extract, the way Streamlit runs a script."""
    script = Path(csv_path).with_name(f"{name}.py")
    lines = [
        "import sys",
        f"sys.path.insert(0, {str(ROOT)!r})",
        "from pathlib import Path",
        "from backend import data",
        "from frontend import app",
        f"df = data.load(Path({csv_path!r}))",
        f"app.{name}(df)",
    ]
    script.write_text("\n".join(lines), encoding="utf-8")
    return AppTest.from_file(str(script), default_timeout=120)


@pytest.mark.parametrize("name", ["new_ticket", "batch_report", "dashboard"])
def test_page_renders(csv_path, name):
    at = page(csv_path, name).run()
    assert not at.exception, [e.value for e in at.exception]


def test_the_form_is_populated_from_the_extract(csv_path):
    at = page(csv_path, "new_ticket").run()
    by_label = {s.label: s for s in at.selectbox}
    assert by_label["Site"].options == ["-", "Brest", "Paris"]
    assert by_label["Urgency"].options == ["-", *d.LEVELS]  # Low, Medium, High: not alphabetical


def test_an_incomplete_form_is_refused(csv_path):
    at = page(csv_path, "new_ticket").run()
    at.button[0].click().run()
    assert [e.value for e in at.error] == ["Short description and description are required."]


def submit(at: AppTest) -> AppTest:
    at.text_input[0].set_value("VPN qui se déconnecte")
    at.text_area[0].set_value("Le VPN se coupe sans arrêt depuis ce matin")
    return at.button[0].click().run()


def test_submitting_before_the_hooks_exist_explains_what_to_implement(csv_path):
    at = submit(page(csv_path, "new_ticket").run())
    assert not at.exception, [e.value for e in at.exception]
    messages = " ".join(i.value for i in at.info)
    assert "extract_fields()" in messages and "embed()" in messages
    assert [s.value for s in at.subheader] == ["Alerts"] and len(at.success) == 1  # nothing to flag yet
    assert len(at.dataframe) == 0


def test_the_page_uses_your_hooks_once_they_are_implemented(csv_path, monkeypatch):
    monkeypatch.setattr(ai, "embed", lambda texts: np.array([[1.0, 0.0] if "VPN" in t else [0.0, 1.0] for t in texts]))
    monkeypatch.setattr(
        ai, "extract_fields", lambda ticket: {d.INCIDENT_TYPE: d.SECURITY, d.TONE: "Frustré", d.AFFECTED_SYSTEM: "VPN"}
    )

    at = submit(page(csv_path, "new_ticket").run())
    assert not at.exception, [e.value for e in at.exception]
    assert [s.value for s in at.subheader] == ["AI analysis", "Alerts", "Similar past tickets"]
    assert [m.label for m in at.metric] == [d.INCIDENT_TYPE, d.TONE, d.AFFECTED_SYSTEM]
    assert any("Security-related" in e.value for e in at.error) and any("Escalation risk" in w.value for w in at.warning)
    table = at.dataframe[0].value
    assert len(table) == 5 and (table[d.TITLE] == "Déconnexions fréquentes du VPN").all()  # the VPN tickets, by text


def test_report_page_generates_a_report(csv_path):
    at = page(csv_path, "batch_report").run()
    next(b for b in at.button if b.label == "Generate report").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Key figures" in at.session_state["report"]


def run_app(data_dir: Path, monkeypatch) -> AppTest:
    """The whole app (``frontend/app.py``), reading its extract from ``data_dir``."""
    monkeypatch.setattr(d, "DATA_DIR", data_dir)
    return AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=60).run()


def test_the_app_loads_the_extract_stored_next_to_the_code(csv_path, tmp_path, monkeypatch):
    (tmp_path / "tickets.csv").write_bytes(Path(csv_path).read_bytes())
    at = run_app(tmp_path, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "New ticket"
    assert "tickets.csv" in at.sidebar.caption[0].value and "1,201 tickets" in at.sidebar.caption[0].value
    assert not at.sidebar.text_input and not at.sidebar.get("file_uploader")  # nothing to upload


def test_without_an_extract_the_app_says_where_to_put_it(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert "No extract found" in at.error[0].value and str(tmp_path) in at.error[0].value


def test_a_broken_extract_is_reported_not_crashed(tmp_path, monkeypatch):
    (tmp_path / "bad.csv").write_text("Numero;Urgence\nINC1;High\n", encoding="utf-8")
    at = run_app(tmp_path, monkeypatch)
    assert not at.exception and "no column" in at.error[0].value
