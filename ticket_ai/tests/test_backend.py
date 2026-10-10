"""Backend tests: loading, filtering, volumes, projection, growth, alerts, nearest tickets, report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend import analytics as an
from backend import data as d
from frontend import render

# --- data ---------------------------------------------------------------------------------


def test_csv_export_is_read_with_dates_and_decimal_commas(csv_path, tickets):
    df = d.load(Path(csv_path))
    assert len(df) == len(tickets)
    assert df[d.OPENED].is_monotonic_increasing and df[d.OPENED].iloc[0] == tickets[d.OPENED].min().floor("s")  # day first
    assert np.allclose(sorted(df[d.RESOLUTION_H]), sorted(tickets[d.RESOLUTION_H]))  # "4,5" -> 4.5 (the loader sorts by date)


def test_headers_are_matched_ignoring_accents_case_and_spacing_and_windows_encoding(tmp_path):
    raw = "NUMERO;ouvert LE;Description  courte;Description;Entité\nINC1;12/03/2026 09:00;VPN coupé;Le VPN se coupe;Nord\n"
    path = tmp_path / "e.csv"
    path.write_bytes(raw.encode("cp1252"))  # not valid UTF-8: the loader must fall back
    df = d.load(path)
    assert df.loc[0, d.ID] == "INC1" and df.loc[0, d.ENTITY] == "Nord" and df.loc[0, d.TITLE] == "VPN coupé"
    assert df.loc[0, d.OPENED] == pd.Timestamp("2026-03-12 09:00")
    assert d.TONE in df.columns and df[d.TONE].isna().all()  # a missing column is added empty


def test_excel_files_are_read(tickets, tmp_path):
    path = tmp_path / "extract.xlsx"
    tickets.head(50).to_excel(path, index=False)
    assert len(d.load(path)) == 50


def test_a_file_without_the_essential_columns_is_refused(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Numero;Urgence\nINC1;High\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no column"):
        d.load(path)


def test_the_extract_is_found_in_the_data_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "DATA_DIR", tmp_path)
    assert d.find_extract() is None
    (tmp_path / "README.md").write_text("not data")
    (tmp_path / "~$lock.xlsx").write_text("excel lock file")
    assert d.find_extract() is None
    (tmp_path / "b.csv").write_text("x")
    (tmp_path / "a.xlsx").write_text("x")
    assert d.find_extract().name == "a.xlsx"  # the first one, in name order


def test_options_and_select(tickets):
    assert d.options(tickets, d.URGENCY, d.LEVELS) == d.LEVELS  # not alphabetical
    window = d.select(tickets, "2026-03-01", "2026-03-31", {d.AFFECTED_SYSTEM: ["Outlook"], d.SITE: []})
    assert not window.empty and (window[d.AFFECTED_SYSTEM] == "Outlook").all()
    assert window[d.OPENED].between("2026-03-01", "2026-04-01").all()  # the last day is included


# --- analytics ----------------------------------------------------------------------------


def test_volume_fills_empty_weeks_and_drops_partial_edges():
    opened = pd.to_datetime(["2026-01-07 10:00", "2026-01-08 10:00", "2026-01-26 10:00", "2026-01-27 10:00"])
    weekly = an.volume(pd.DataFrame({d.OPENED: opened}), "W")["Tickets"]
    assert weekly.tolist() == [0, 0]  # data starts on a Wednesday and ends on a Tuesday: both edge weeks dropped,
    assert weekly.index.tolist() == [pd.Timestamp("2026-01-12"), pd.Timestamp("2026-01-19")]  # the empty ones kept


def test_volume_by_category(tickets):
    wide = an.volume(tickets, "W", d.INCIDENT_TYPE)
    assert set(wide.columns) == set(tickets[d.INCIDENT_TYPE])
    assert wide.sum().sum() <= len(tickets)


def test_projection_follows_the_trend_with_a_widening_band():
    rng = np.random.default_rng(1)
    series = pd.Series(
        50 + 1.5 * np.arange(40) + rng.normal(0, 3, 40), index=pd.date_range("2025-01-06", periods=40, freq="W-MON")
    )
    p = an.project(series, 8)
    assert len(p) == 8 and p.index[0] == series.index[-1] + pd.Timedelta(days=7)
    assert (p["lower"] <= p["forecast"]).all() and (p["forecast"] <= p["upper"]).all()
    assert (p["upper"] - p["lower"]).is_monotonic_increasing
    assert p["forecast"].iloc[0] > series.iloc[-10:].mean()


def test_projection_handles_short_and_falling_series():
    short = pd.Series([10, 12, 11], index=pd.date_range("2025-01-06", periods=3, freq="W-MON"))
    assert an.project(short, 3)["forecast"].nunique() == 1
    falling = pd.Series(np.linspace(30, 1, 30), index=pd.date_range("2025-01-06", periods=30, freq="W-MON"))
    assert (an.project(falling, 12)[["forecast", "lower"]] >= 0).all().all()


def test_growth_finds_the_vpn_surge(tickets):
    g = an.growth(tickets, d.AFFECTED_SYSTEM)
    vpn = g.loc["VPN"]
    assert g.index[0] == "VPN" and vpn["share_of_increase"] > 0.9 and vpn["p_value"] < 0.001
    assert pd.isna(vpn["ratio"])  # VPN did not exist before the surge: "new", there is no ratio
    assert g.drop("VPN")["share_of_increase"].sum() < 0.1  # the stable topics explain none of the rise
    # A topic that existed before gets a real ratio.
    grown = tickets.assign(
        **{d.AFFECTED_SYSTEM: np.where(tickets[d.AFFECTED_SYSTEM] == "VPN", "Wi-Fi", tickets[d.AFFECTED_SYSTEM])}
    )
    assert an.growth(grown, d.AFFECTED_SYSTEM).loc["Wi-Fi", "ratio"] > 3
    assert an.growth(tickets.head(40), d.AFFECTED_SYSTEM).empty  # too short for a baseline


def test_batch_alerts_flag_the_surge_and_ignore_small_batches(tickets):
    alerts = an.batch_alerts(tickets)
    assert any("VPN" in a.title and "(new)" in a.title for a in alerts)
    assert [a.level for a in alerts] == sorted((a.level for a in alerts), key=lambda x: {"error": 0, "warning": 1}[x])
    assert an.batch_alerts(tickets.head(10)) == []


def test_kpis_tolerate_empty_columns(tickets):
    k = an.kpis(tickets)
    assert k["tickets"] == len(tickets) and 0 <= k["sla_missed_pct"] <= 100
    empty = tickets.assign(**{d.TONE: pd.NA, d.SLA: pd.NA})
    assert an.kpis(empty)["negative_tone_pct"] is None and an.kpis(empty)["sla_missed_pct"] is None


# --- similar tickets and ticket alerts -----------------------------------------------------


def test_nearest_ranks_by_text_and_returns_recent_first_on_ties(tickets):
    vectors = np.array([[1.0, 0.0] if s == "VPN" else [0.0, 1.0] for s in tickets[d.AFFECTED_SYSTEM]])
    found = an.nearest(tickets, vectors, np.array([1.0, 0.0]), {}, k=10)
    assert (found[d.AFFECTED_SYSTEM] == "VPN").all() and found["Similarity"].eq(0.7).all()  # 0.7 x cosine 1.0
    assert found[d.OPENED].is_monotonic_decreasing  # equal scores: the most recent first


def test_matching_context_raises_the_score(tickets):
    same = np.ones((len(tickets), 2))
    found = an.nearest(tickets, same, np.ones(2), {d.SERVICE: "Connectivité", d.AFFECTED_SYSTEM: "VPN"}, k=len(tickets))
    assert found.iloc[0]["Similarity"] == pytest.approx(0.7 + 0.3 * 2 / len(an.CONTEXT))
    assert found["Similarity"].is_monotonic_decreasing and found.iloc[-1]["Similarity"] == pytest.approx(0.7)


def _similar(n, days_ago, now, similarity=0.8, **extra):
    return pd.DataFrame({"Similarity": similarity, d.OPENED: [now - pd.Timedelta(days=days_ago)] * n,
                         d.REOPENED: 0, d.SLA: "Respecté", **extra})  # fmt: skip


def test_each_ticket_rule_fires_on_its_own_trigger():
    now = pd.Timestamp("2026-06-30")
    top = d.LEVELS[-1]
    fields = {d.INCIDENT_TYPE: d.SECURITY, d.SCOPE: "Multi-sites", d.TONE: "Agressif"}
    alerts = an.ticket_alerts({d.URGENCY: top, d.IMPACT: top}, fields, _similar(0, 1, now), now)
    assert [a.title.split(":")[0] for a in alerts] == [
        "Possible major incident",
        "Security-related ticket",
        "Wide impact",
        "Escalation risk",
    ]
    assert [a.level for a in alerts] == ["error", "error", "warning", "warning"]
    assert an.ticket_alerts({}, {}, _similar(0, 1, now), now) == []  # nothing implemented yet: nothing raised


def test_recurring_pattern_is_graded_and_needs_to_be_recent():
    now = pd.Timestamp("2026-06-30")
    level = lambda similar: [a.level for a in an.ticket_alerts({}, {}, similar, now) if "Emerging" in a.title]  # noqa: E731
    assert level(_similar(5, 3, now)) == ["warning"] and level(_similar(10, 3, now)) == ["error"]
    assert level(_similar(10, 60, now)) == [] and level(_similar(10, 3, now, similarity=0.3)) == []


def test_hard_to_resolve_similar_tickets_are_flagged():
    now = pd.Timestamp("2026-06-30")
    similar = _similar(10, 90, now).assign(**{d.SLA: d.SLA_BREACHED})
    assert any("hard to resolve" in a.title for a in an.ticket_alerts({}, {}, similar, now))


# --- report -------------------------------------------------------------------------------


def test_report_has_its_sections_and_embeds_plotly(tickets):
    page = render.report_html(tickets, "My report")
    for text in (
        "My report",
        "Key figures",
        "Alerts",
        "Weekly volume",
        "Incident types",
        "Fastest growing",
        "Growing",
        "window.print",
    ):
        assert text in page
    assert "<script src=" not in page and len(page) > 1_000_000  # Plotly's JavaScript is inline: it opens offline


def test_report_escapes_what_comes_from_the_data(tickets):
    hostile = tickets.copy()
    hostile.loc[hostile.index[-300:], d.AFFECTED_SYSTEM] = "<script>alert(1)</script>"
    hostile[d.INCIDENT_TYPE] = hostile[d.INCIDENT_TYPE].replace("Réseau", "<img src=x onerror=alert(2)>")
    page = render.report_html(hostile, "<b>t</b>")
    assert "<script>alert(1)" not in page and "<img src=x" not in page and "<b>t</b>" not in page
    assert "&lt;script&gt;alert(1)" in page  # it reached the alerts and the table, escaped
