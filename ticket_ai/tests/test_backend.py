"""Backend tests: schema, data, privacy, LLM layer, similarity, alerts, analytics, report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import requests

from backend.alerts import Alert, ticket_alerts
from backend.analytics import batch_alerts, change_vs_previous, distribution, growth_by, kpis, project, volume_series
from backend.data import MissingColumnsError, TicketStore, load_extract
from backend.llm import Enrichment, TicketEnricher, choose, parse_json_object
from backend.mock_data import generate_tickets
from backend.privacy import redact_pii
from backend.report import build_report, pdf_available
from backend.schema import ALL_COLUMNS, Col, derive_priority, normalise_columns

# --------------------------------------------------------------------------- schema


def test_headers_are_matched_ignoring_accents_case_and_the_known_typo():
    frame = pd.DataFrame(columns=["numero", "TEMPS ÉCOULÉ", "Commentaire additionnel", "Priorité", "inconnu"])
    renamed = list(normalise_columns(frame).columns)
    assert renamed == [Col.ID, Col.ELAPSED_HOURS, Col.COMMENT, Col.PRIORITY, "inconnu"]


@pytest.mark.parametrize(
    ("impact", "urgency", "expected"),
    [
        ("High", "High", "1 - Critical"),
        ("High", "Medium", "2 - High"),
        ("Medium", "Medium", "3 - Moderate"),
        ("Low", "Medium", "4 - Low"),
        ("Low", "Low", "4 - Low"),
    ],  # Low/Low once indexed out of range
)
def test_priority_matrix(impact, urgency, expected):
    assert derive_priority(impact, urgency) == expected


# --------------------------------------------------------------------------- data


def test_mock_extract_has_every_column_and_is_deterministic(tickets):
    assert list(tickets.columns) == ALL_COLUMNS
    again = generate_tickets(n=1500, seed=3, end="2026-06-30", days=300)
    pd.testing.assert_frame_equal(tickets, again)


def test_mock_extract_is_internally_consistent(tickets):
    assert tickets[Col.ID].is_unique
    resolved = tickets[tickets[Col.RESOLVED].notna()]
    assert (resolved[Col.RESOLUTION_HOURS] > 0).all()
    assert (tickets[Col.OPENED] <= pd.Timestamp("2026-06-30 15:00")).all()  # nothing opens after the extract
    assert (
        tickets[Col.PRIORITY].eq(tickets.apply(lambda r: derive_priority(r[Col.IMPACT], r[Col.URGENCY]), axis=1)).all()
    )
    assert 0 < tickets[Col.STAGE].isin(["Nouveau", "Affecté", "En cours", "En attente utilisateur"]).sum() < 60


def test_store_filters_by_date_and_selection(store):
    window = store.filter(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-31"), {Col.INCIDENT_TYPE: ["Réseau"]})
    assert not window.empty
    assert window[Col.OPENED].between("2026-03-01", "2026-04-01").all()
    assert (window[Col.INCIDENT_TYPE] == "Réseau").all()
    assert len(store.filter(selections={Col.INCIDENT_TYPE: []})) == len(store.tickets)  # empty = no restriction


def test_load_extract_handles_a_french_export(tmp_path):
    path = tmp_path / "export.csv"
    path.write_text(
        "Numéro;Ouvert le;Description courte;Description;Temps de resolution;Urgence\n"
        "INC1;12/03/2026 14:05:33;VPN coupé;Le VPN se coupe;4,5;High\n",
        encoding="utf-8",
    )
    frame = load_extract(path)
    assert frame.loc[0, Col.OPENED] == pd.Timestamp("2026-03-12 14:05:33")  # day first
    assert frame.loc[0, Col.RESOLUTION_HOURS] == 4.5  # decimal comma
    assert Col.INCIDENT_TYPE in frame.columns and frame[Col.INCIDENT_TYPE].isna().all()  # absent LLM column added


def test_load_extract_rejects_a_file_without_the_essentials(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Numéro;Urgence\nINC1;High\n", encoding="utf-8")
    with pytest.raises(MissingColumnsError):
        load_extract(path)


# --------------------------------------------------------------------------- privacy


def test_formatted_personal_data_is_masked():
    text = (
        "Mail jean.martin@exemple.fr, tel 06 12 34 56 78, "
        "IBAN FR76 3000 6000 0112 3456 7890 189, carte 4970 1012 3456 7890"
    )
    masked, found = redact_pii(text)
    for secret in ("jean.martin", "06 12 34 56 78", "3000 6000", "4970 1012"):
        assert secret not in masked
    assert set(found) == {"EMAIL", "PHONE", "IBAN", "CREDIT_CARD"}


def test_text_without_personal_data_is_untouched():
    text = "Le VPN se coupe toutes les 10 minutes depuis la version 5.2"
    assert redact_pii(text) == (text, [])


# --------------------------------------------------------------------------- llm layer


class FakeClient:
    """Stands in for LLMClient; records what it was asked."""

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.prompts = answer, error, []

    def complete_json(self, system, user):
        self.prompts.append(user)
        if self.error:
            raise self.error
        return self.answer


def test_json_is_extracted_from_a_chatty_answer():
    answer = 'Sure!\n```json\n{"tone": "Neutre", "scope": "Site"}\n```\nHope it helps'
    assert parse_json_object(answer) == {"tone": "Neutre", "scope": "Site"}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_answers_are_mapped_onto_allowed_values():
    assert choose("frustre", ("Neutre", "Frustré"), "Neutre") == "Frustré"  # accent-insensitive
    assert choose("angry", ("Neutre", "Frustré"), "Neutre") == "Neutre"  # unknown -> default


def test_enricher_validates_what_the_llm_returns():
    client = FakeClient(
        {
            "incident_type": "reseau",
            "request_type": "???",
            "tone": "Frustré",
            "scope": "Site",
            "affected_system": "VPN",
            "root_cause": "Mise à jour",
        }
    )
    result = TicketEnricher(client).enrich({Col.TITLE: "VPN", Col.DESCRIPTION: "coupé"})
    assert (result.source, result.incident_type, result.request_type) == ("llm", "Réseau", "Incident")
    assert result.tone == "Frustré" and result.affected_system == "VPN"


def test_enricher_masks_personal_data_before_calling_the_api():
    client = FakeClient({})
    enrichment = TicketEnricher(client, redact=True).enrich(
        {Col.TITLE: "VPN", Col.DESCRIPTION: "Appelez-moi au 06 12 34 56 78 ou jean@exemple.fr"}
    )
    assert "06 12 34 56 78" not in client.prompts[0] and "jean@exemple.fr" not in client.prompts[0]
    assert enrichment.pii_found == ["EMAIL", "PHONE"]


def test_enricher_falls_back_when_the_api_fails():
    result = TicketEnricher(FakeClient(error=requests.ConnectionError("down"))).enrich(
        {Col.TITLE: "VPN coupé", Col.DESCRIPTION: "le vpn se coupe"}
    )
    assert result.source == "offline" and "failed" in result.notice
    assert result.incident_type == "Réseau"


def test_offline_heuristic_reads_the_text():
    result = TicketEnricher(None).enrich(
        {
            Col.TITLE: "Courriel suspect",
            Col.DESCRIPTION: "C'est inadmissible, un phishing demandant mon mot de passe, toute l'entreprise est visée",
        }
    )
    assert (result.incident_type, result.tone, result.scope) == ("Sécurité", "Agressif", "Multi-sites")


# --------------------------------------------------------------------------- similarity


def test_search_returns_ranked_neighbours_with_an_explainable_score(index, store):
    query = {
        Col.TITLE: "Déconnexions fréquentes du VPN",
        Col.DESCRIPTION: "le VPN se coupe en télétravail",
        Col.AFFECTED_SYSTEM: "VPN",
        Col.INCIDENT_TYPE: "Réseau",
    }
    found = index.search(query, k=10)
    assert len(found) == 10 and found["Similarity"].is_monotonic_decreasing
    assert (found[Col.AFFECTED_SYSTEM] == "VPN").mean() > 0.8
    blend = 0.7 * found["Text match"] + 0.3 * found["Category match"]
    assert np.allclose(blend, found["Similarity"])


def test_matching_context_raises_the_score(index):
    base = {Col.TITLE: "Imprimante hors ligne", Col.DESCRIPTION: "l'imprimante du plateau est hors ligne"}
    with_context = index.search({**base, Col.INCIDENT_TYPE: "Impression", Col.AFFECTED_SYSTEM: "Imprimante"}, k=1)
    without = index.search(base, k=1)
    assert with_context["Similarity"].iloc[0] > without["Similarity"].iloc[0]


# --------------------------------------------------------------------------- alerts


def _neighbours(similarity, days_ago, now, **extra):
    n = len(similarity)
    return pd.DataFrame(
        {
            "Similarity": similarity,
            Col.OPENED: [now - pd.Timedelta(days=d) for d in days_ago],
            Col.REOPEN_COUNT: [0] * n,
            Col.SLA: ["Respecté"] * n,
            **extra,
        }
    )


def test_each_rule_fires_on_its_own_trigger():
    now = pd.Timestamp("2026-06-30")
    quiet = _neighbours([0.3], [100], now)
    ticket = {Col.URGENCY: "High", Col.IMPACT: "High"}
    codes = {
        a.code
        for a in ticket_alerts(
            ticket,
            Enrichment(incident_type="Sécurité", scope="Multi-sites", tone="Agressif", pii_found=["EMAIL"]),
            quiet,
            now,
        )
    }
    assert codes == {"MAJOR_INCIDENT", "SECURITY", "WIDE_SCOPE", "NEGATIVE_TONE", "PII"}


def test_benign_ticket_raises_nothing():
    now = pd.Timestamp("2026-06-30")
    assert (
        ticket_alerts({Col.URGENCY: "Low", Col.IMPACT: "Low"}, Enrichment(), _neighbours([0.3], [100], now), now) == []
    )


def test_recurring_pattern_is_detected_and_graded():
    now = pd.Timestamp("2026-06-30")
    five = _neighbours([0.8] * 5, [1, 2, 3, 4, 5], now)
    ten = _neighbours([0.8] * 10, list(range(1, 11)), now)
    old = _neighbours([0.8] * 10, [60] * 10, now)
    levels = lambda n: [a.level for a in ticket_alerts({}, Enrichment(), n, now) if a.code == "RECURRING"]  # noqa: E731
    assert levels(five) == ["warning"] and levels(ten) == ["critical"] and levels(old) == []


def test_alerts_are_sorted_by_severity():
    now = pd.Timestamp("2026-06-30")
    alerts = ticket_alerts(
        {Col.URGENCY: "High", Col.IMPACT: "High"},
        Enrichment(tone="Urgent", pii_found=["PHONE"]),
        _neighbours([0.3], [100], now),
        now,
    )
    assert [a.level for a in alerts] == ["critical", "warning", "info"]


# --------------------------------------------------------------------------- analytics


def test_empty_periods_are_zero_and_partial_edges_are_flagged():
    opened = pd.to_datetime(["2026-01-07 10:00", "2026-01-08 10:00", "2026-01-26 10:00", "2026-01-27 10:00"])
    series = volume_series(pd.DataFrame({Col.OPENED: opened}), "Week")
    assert series["tickets"].tolist() == [2, 0, 0, 2]  # the two empty weeks in the middle are kept
    assert series["complete"].tolist() == [False, True, True, False]  # starts on a Wednesday, ends on a Tuesday


def test_projection_is_sane():
    rng = np.random.default_rng(1)
    index = pd.date_range("2025-01-06", periods=40, freq="W-MON")
    series = pd.Series(50 + 1.5 * np.arange(40) + rng.normal(0, 3, 40), index=index)
    result = project(series, 8, "Week")
    assert len(result) == 8 and result["period"].iloc[0] == index[-1] + pd.Timedelta(days=7)
    assert (result["lower"] <= result["forecast"]).all() and (result["forecast"] <= result["upper"]).all()
    assert (result["upper"] - result["lower"]).is_monotonic_increasing  # uncertainty grows with the horizon
    assert result["forecast"].iloc[0] > series.iloc[-10:].mean()  # follows the upward trend


def test_projection_never_goes_negative_and_copes_with_short_series():
    falling = pd.Series(np.linspace(30, 1, 30), index=pd.date_range("2025-01-06", periods=30, freq="W-MON"))
    assert (project(falling, 12, "Week")[["forecast", "lower"]] >= 0).all().all()
    short = pd.Series([10, 12, 11], index=pd.date_range("2025-01-06", periods=3, freq="W-MON"))
    assert project(short, 3, "Week")["forecast"].nunique() == 1  # flat


def test_growth_finds_the_planted_vpn_surge(store):
    growth = growth_by(store.tickets, Col.AFFECTED_SYSTEM)
    top = growth.iloc[0]
    assert top[Col.AFFECTED_SYSTEM] == "VPN" and top["ratio"] > 3 and top["p_value"] < 0.001


def test_batch_alerts_surface_the_planted_events(store):
    titles = " ".join(a.title for a in batch_alerts(store.tickets))
    assert "VPN" in titles
    assert batch_alerts(store.tickets.head(10)) == []  # too small to judge


def test_kpis_distribution_and_change(store):
    indicators = kpis(store.tickets)
    assert indicators["tickets"] == len(store.tickets) and 0 <= indicators["sla_breach_pct"] <= 100
    share = distribution(store.tickets, Col.INCIDENT_TYPE, top_n=3)
    assert (
        len(share) == 4 and share["share"].sum() == pytest.approx(1.0) and share.iloc[-1][Col.INCIDENT_TYPE] == "Autres"
    )
    assert change_vs_previous(store.tickets, "2026-06-01", "2026-06-30") is not None


# --------------------------------------------------------------------------- report


def test_report_contains_the_expected_sections(store):
    report = build_report(store.tickets, "My report", {"Period": "all"})
    document = report.to_html()
    for expected in ("My report", "Key figures", "Alerts", "Weekly ticket volume", "Largest root causes", "<svg"):
        assert expected in document
    assert any(a.code == "GROWING_TOPIC" for a in report.alerts)


def test_report_escapes_text_coming_from_the_data(store):
    """Dynamic text reaches the HTML in alerts, tables and the header: all must be escaped."""
    hostile = store.tickets.copy()
    # Recent tickets, so the value becomes a "growing root cause" alert and a table row.
    hostile.loc[hostile.index[-300:], Col.ROOT_CAUSE] = "<script>alert(1)</script>"
    report = build_report(hostile, "<b>t</b>", {"x": "<i>"})
    document = report.to_html()

    assert any("<script>" in a.title for a in report.alerts)  # the value really reached an alert...
    assert "&lt;script&gt;" in document  # ...and was escaped there,
    assert "<script>" not in document  # nowhere raw,
    assert "<b>t</b>" not in document and "&lt;b&gt;t&lt;/b&gt;" in document  # and in the title too


def test_report_refuses_an_empty_batch(store):
    with pytest.raises(ValueError):
        build_report(store.tickets.iloc[0:0])


@pytest.mark.skipif(not pdf_available(), reason="WeasyPrint not installed")
def test_pdf_export(store):
    assert build_report(store.tickets).to_pdf().startswith(b"%PDF")


def test_store_exposes_reference_date_and_options(store):
    assert isinstance(store, TicketStore)
    assert store.reference_date == store.tickets[Col.OPENED].max()
    assert "VPN" in store.options(Col.AFFECTED_SYSTEM)
    assert isinstance(Alert("info", "X", "t"), Alert)
