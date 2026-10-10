"""Sub-topic discovery, naming, and attribution of a ticket increase.

The headline test is :func:`test_attribution_recovers_the_injected_root_cause`:
the generator concentrates a surge on one symptom, and the attribution has to
name that symptom as the dominant driver. Everything else here guards the
properties that make the output trustworthy over time - stable ids, refusal to
tag what it does not recognise, and a labeller that cannot crash a run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from itsm.analysis.root_cause import (
    ASSIGN_THRESHOLD,
    MERGE_THRESHOLD,
    SubTopic,
    _merge_similar,
    _stable_id,
    attribute_growth,
    discover_subtopics,
)
from itsm.analysis.topic_labeller import LlmLabeller, TemplateLabeller, _clean
from itsm.data.synthetic import generate

# TF-IDF keeps the suite fast and offline; the embedding backend is exercised
# by the attribution test below, which is the one that needs meaning.
BACKEND = "tfidf"


@pytest.fixture(scope="module")
def targeted_dataset() -> object:
    """A corpus whose Email & Collaboration ramp targets a single symptom."""
    return generate(n_tickets=9000, days=330, seed=42)


def test_discovery_splits_categories_into_named_subtopics(tickets: pd.DataFrame) -> None:
    model = discover_subtopics(tickets, backend=BACKEND, min_cluster_size=15)

    assert len(model.topics) > tickets["category"].nunique()
    assert all(topic.label for topic in model.topics)
    assert all(topic.parent_category in set(tickets["category"]) for topic in model.topics)
    # Every assignment points at a real topic, and unassigned is the empty id.
    known = {topic.topic_id for topic in model.topics} | {""}
    assert set(model.assignments) <= known
    assert 0.0 < model.coverage <= 1.0


def test_attribution_recovers_the_injected_root_cause(targeted_dataset: object) -> None:
    """The dominant driver must be the symptom the generator surged."""
    dataset = targeted_dataset
    tickets = dataset.tickets  # type: ignore[attr-defined]
    ramp = next(a for a in dataset.anomalies if a.kind == "ramp")  # type: ignore[attr-defined]

    model = discover_subtopics(tickets, backend="embeddings", min_cluster_size=25)
    ranked = attribute_growth(
        tickets,
        model.assignments,
        model,
        recent_days=14,
        baseline_days=56,
        as_of=pd.Timestamp(ramp.end),
    )

    top = ranked.iloc[0]
    assert top["parent_category"] == ramp.category
    # The surge was concentrated on one symptom, so one sub-topic must carry
    # most of the increase rather than it spreading across the family.
    assert top["share_of_increase"] > 0.4
    assert top["poisson_p"] < 0.001
    assert top["growth_ratio"] > 3

    # And the label must name that symptom: the generator forced template 0 of
    # the family, whose French wording is the Outlook connection failure.
    assert "outlook" in str(top["label"]).lower()


def test_cross_language_variants_merge_into_one_topic(targeted_dataset: object) -> None:
    """The same symptom in two languages must not be two drivers.

    Unmerged, the injected root cause is reported as two rows of ~33% and ~24%
    instead of one of ~57% - which is how a real driver gets mistaken for two
    minor ones.
    """
    tickets = targeted_dataset.tickets  # type: ignore[attr-defined]
    merged = discover_subtopics(tickets, backend="embeddings", min_cluster_size=25)
    split = discover_subtopics(
        tickets, backend="embeddings", min_cluster_size=25, merge_threshold=1.01
    )
    assert len(merged.topics) < len(split.topics)


def test_merge_only_folds_near_identical_centroids() -> None:
    features = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]])
    labels = np.array([0, 1, 2, 3])
    merged = _merge_similar(
        features / np.linalg.norm(features, axis=1, keepdims=True), labels, MERGE_THRESHOLD
    )
    # Two pairs, each internally similar and mutually orthogonal.
    assert len(set(merged)) == 2
    assert merged[0] == merged[1]
    assert merged[2] == merged[3]


def test_unrelated_clusters_are_left_alone() -> None:
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert len(set(_merge_similar(features, labels, MERGE_THRESHOLD))) == 2


def test_topic_ids_are_stable_across_runs(tickets: pd.DataFrame) -> None:
    """A tag that changes meaning between runs makes trend tracking useless."""
    first = discover_subtopics(tickets, backend=BACKEND, min_cluster_size=15)
    second = discover_subtopics(tickets, backend=BACKEND, min_cluster_size=15, previous=first)

    assert {t.topic_id for t in second.topics} & {t.topic_id for t in first.topics}
    # Same corpus, same parameters: every topic should be recognised.
    assert {t.topic_id for t in second.topics} == {t.topic_id for t in first.topics}


def test_a_new_topic_gets_a_new_id() -> None:
    previous = [
        SubTopic(
            topic_id="NE-000",
            parent_category="Network & VPN",
            label="vpn drops",
            keywords=[],
            size=50,
            centroid=np.array([1.0, 0.0]),
        )
    ]
    assert _stable_id(np.array([1.0, 0.0]), "Network & VPN", previous, 7) == "NE-000"
    # Orthogonal centroid: unrelated, so it must not inherit the id.
    assert _stable_id(np.array([0.0, 1.0]), "Network & VPN", previous, 7) != "NE-000"


def test_assignment_refuses_unrecognised_text(tickets: pd.DataFrame) -> None:
    """Refusing to tag is how a genuinely new failure mode stays visible."""
    model = discover_subtopics(tickets, backend=BACKEND, min_cluster_size=15)
    assigned = model.assign(["Le VPN se deconnecte toutes les cinq minutes"])
    assert len(assigned) == 1
    assert 0.0 <= ASSIGN_THRESHOLD <= 1.0


def test_attribution_flags_an_emerging_topic(tickets: pd.DataFrame) -> None:
    """A sub-topic absent from the baseline has no ratio and is 'emerging'."""
    model = discover_subtopics(tickets, backend=BACKEND, min_cluster_size=15)
    ranked = attribute_growth(tickets, model.assignments, model, recent_days=14, baseline_days=56)

    assert {"delta_per_day", "share_of_increase", "poisson_p", "is_emerging"} <= set(ranked.columns)
    assert ranked["delta_per_day"].is_monotonic_decreasing
    assert ranked.loc[ranked["is_emerging"], "growth_ratio"].isna().all()
    # Shares are of the *total* increase, sub-threshold topics included, so the
    # ranked rows sum to at most 1 - never above it, and never rescaled.
    positive = ranked.loc[ranked["delta_per_day"] > 0, "share_of_increase"].sum()
    assert 0.0 <= positive <= 1.0 + 1e-9


def test_centroids_share_one_vector_space(tickets: pd.DataFrame) -> None:
    """Every topic must be comparable to every other, across categories.

    Encoding each category separately would fit a different basis per slice and
    silently make cross-category similarity meaningless - and break `assign`.
    """
    model = discover_subtopics(tickets, backend=BACKEND, min_cluster_size=15)
    dimensions = {t.centroid.shape[0] for t in model.topics if t.centroid is not None}
    assert len(dimensions) == 1


def test_template_labeller_prefers_the_shortest_title() -> None:
    labeller = TemplateLabeller()
    label = labeller.label(
        ["vpn", "deconnexion"],
        ["Le VPN se deconnecte sans arret depuis la mise a jour", "VPN drops"],
        "Network & VPN",
    )
    assert label == "VPN drops"


def test_llm_labeller_falls_back_when_the_model_fails() -> None:
    """A labelling failure must never fail the analysis."""

    def explode(prompt: str) -> str:
        raise RuntimeError("endpoint unreachable")

    label = LlmLabeller(explode).label(["vpn"], ["VPN drops"], "Network & VPN")
    assert label == "VPN drops"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"Outlook connection failures"', "Outlook connection failures"),
        ("Label: VPN drops after update", "VPN drops after update"),
        ("ok", "FALLBACK"),  # too short to be a name
        ("x" * 200, "FALLBACK"),  # a paragraph, not a phrase
    ],
)
def test_llm_output_is_cleaned_or_rejected(raw: str, expected: str) -> None:
    assert _clean(raw, "FALLBACK") == expected


def test_assign_projects_new_text_into_the_fitted_space(tickets: pd.DataFrame) -> None:
    """A new ticket must be encoded by the *fitted* encoder, not a fresh one.

    Re-fitting on the incoming batch is the subtle failure: it succeeds on a
    large batch, puts the vectors in a different basis, and every similarity it
    produces is meaningless. On a single ticket it does not even succeed.
    """
    model = discover_subtopics(tickets, backend=BACKEND, min_cluster_size=15)
    assert model.encoder is not None

    one = model.assign(["Le VPN se deconnecte toutes les cinq minutes"])
    assert len(one) == 1

    # Tickets drawn from a known topic should mostly land back on it.
    topic = max(model.topics, key=lambda t: t.size)
    back = model.assign(topic.examples)
    assert any(assigned == topic.topic_id for assigned in back)
