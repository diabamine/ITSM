"""Text normalisation, clustering and lifecycle statistics."""

from __future__ import annotations

import pandas as pd
import pytest

from itsm.analysis.clustering import DesignationClusterer
from itsm.analysis.labeling import class_tfidf
from itsm.analysis.lifecycle import (
    aging_backlog,
    backlog_timeline,
    dwell_times,
    followed_ticket,
    kpi_overview,
    kpis_by,
    rework_profile,
    state_transitions,
)
from itsm.text.normalize import normalize_text, tokenize


@pytest.mark.parametrize(
    ("raw", "expected_token"),
    [
        ("Connexion impossible à SAP - erreur 0x80070005", "<code>"),
        ("Voir http://intranet/kb/123 pour le detail", "<url>"),
        ("Contacter jean.dupont@example.com", "<email>"),
        ("Poste PC12345 ne demarre plus", "<asset>"),
    ],
)
def test_normalisation_replaces_identifiers(raw: str, expected_token: str) -> None:
    assert expected_token in normalize_text(raw)


def test_normalisation_is_accent_and_case_insensitive() -> None:
    assert normalize_text("Problème RÉSEAU") == normalize_text("probleme reseau")


def test_tokenize_drops_stopwords() -> None:
    tokens = tokenize(normalize_text("le vpn ne fonctionne plus depuis ce matin"))
    assert "vpn" in tokens
    assert "le" not in tokens


def test_clustering_recovers_the_declared_taxonomy(tickets: pd.DataFrame) -> None:
    clusterer = DesignationClusterer(n_clusters=24, random_state=0)
    result = clusterer.fit(tickets["designation"].tolist(), tickets["category"])

    assert result.n_clusters == 24
    # Clusters are finer than the taxonomy - one category spans several symptom
    # templates and two languages - so the contract is *purity*, not a
    # one-to-one match: completeness, and hence ARI, stay moderate by
    # construction. The absolute level is further capped by the label noise and
    # the ambiguous wordings the generator injects on purpose. See the module
    # docstring of itsm.analysis.clustering.
    assert result.metrics["homogeneity"] > 0.33
    assert result.metrics["adjusted_rand"] > 0.08
    assert result.metrics["silhouette"] > 0.0

    purities = [summary.category_purity or 0.0 for summary in result.summaries]
    assert sum(purities) / len(purities) > 0.50
    assert all(summary.top_terms for summary in result.summaries)

    predicted = clusterer.predict(["Le VPN se deconnecte toutes les cinq minutes"])
    assert predicted.shape == (1,)


def test_cluster_granularity_increases_purity(tickets: pd.DataFrame) -> None:
    """More families means purer families: the trade-off an analyst tunes."""
    designations = tickets["designation"].tolist()
    coarse = DesignationClusterer(n_clusters=12, random_state=0).fit(
        designations, tickets["category"]
    )
    fine = DesignationClusterer(n_clusters=30, random_state=0).fit(
        designations, tickets["category"]
    )

    assert fine.metrics["homogeneity"] > coarse.metrics["homogeneity"]
    assert fine.metrics["completeness"] < coarse.metrics["completeness"] + 0.15


def test_class_tfidf_returns_terms_per_cluster() -> None:
    documents = ["vpn deconnexion reseau"] * 20 + ["mot de passe compte bloque"] * 20
    labels = pd.Series([0] * 20 + [1] * 20).to_numpy()
    terms = class_tfidf(documents, labels, top_n=3)
    assert set(terms) == {0, 1}
    assert any("vpn" in term for term, _ in terms[0])


def test_kpi_overview_is_complete(tickets: pd.DataFrame) -> None:
    overview = kpi_overview(tickets)
    assert overview["tickets"] == len(tickets)
    assert 0 <= overview["sla_breach_rate"] <= 1
    assert overview["mttr_median_hours"] > 0
    assert 0 <= overview["first_time_right_rate"] <= 1


def test_kpis_by_category_covers_every_family(tickets: pd.DataFrame) -> None:
    summary = kpis_by(tickets, "category")
    assert len(summary) == tickets["category"].nunique()
    assert summary["tickets"].sum() == len(tickets)


def test_flow_statistics(tickets: pd.DataFrame) -> None:
    backlog = backlog_timeline(tickets, freq="W")
    assert {"created", "resolved", "net", "open_backlog"}.issubset(backlog.columns)
    assert (backlog["created"] >= 0).all()

    aging = aging_backlog(tickets)
    assert aging["tickets"].sum() > 0


def test_status_update_statistics(events: pd.DataFrame) -> None:
    transitions = state_transitions(events)
    assert transitions["probability"].between(0, 1).all()
    # Row-normalised probabilities must sum to one per source state.
    sums = transitions.groupby("from_state")["probability"].sum()
    assert ((sums - 1).abs() < 1e-6).all()

    dwell = dwell_times(events)
    assert (dwell["median_hours"] >= 0).all()

    rework = rework_profile(events)
    assert 0 <= rework["reopen_rate"] <= 1


def test_followed_ticket_returns_a_timeline(tickets: pd.DataFrame, events: pd.DataFrame) -> None:
    ticket_id = str(events["ticket_id"].iloc[0])
    trace = followed_ticket(tickets, events, ticket_id)
    assert trace["n_updates"] >= 1
    assert trace["timeline"][0]["state"] == "New"

    with pytest.raises(KeyError):
        followed_ticket(tickets, events, "does-not-exist")
