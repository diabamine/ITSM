"""Triage page: an ITSM intake form scored live by the models.

This is the page that makes the project concrete: the form a requester would
fill, followed by everything the models can say about it - suggested routing
with an explicit confidence, requester mood, discovered family, comparable past
tickets and the resulting expected handling time.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from itsm.app import charts, state
from itsm.data.catalog import SITES
from itsm.data.schema import Priority

EXAMPLES = {
    "VPN instable (FR)": (
        "Le VPN se deconnecte toutes les cinq minutes en teletravail",
        "Depuis la mise a jour de lundi, je suis deconnecte en permanence. "
        "C'est la troisieme fois que j'ouvre un ticket, delai inacceptable, je suis bloque.",
    ),
    "Password reset (EN)": (
        "Password reset required for Office365",
        "I cannot log in since this morning, could you reset it please? Thanks in advance.",
    ),
    "SAP dump (FR)": (
        "La transaction SAP ST22 se termine en dump",
        "Le dump apparait a chaque validation de commande, plusieurs utilisateurs sont impactes.",
    ),
    "Suspicious email (EN)": (
        "Suspicious email asking for my credentials",
        "I received a message that looks like phishing, asking me to confirm my password.",
    ),
}


def _priority_from(impact: int, urgency: int) -> Priority:
    """ITIL-style priority from the impact x urgency matrix."""
    score = impact + urgency
    if score <= 2:
        return Priority.P1
    if score == 3:
        return Priority.P2
    if score <= 5:
        return Priority.P3
    return Priority.P4


def _render_results(designation: str, description: str, declared_category: str | None) -> None:
    """Run every model on the submitted ticket and display the outcome."""
    text = f"{designation}. {description}".strip()

    prediction = state.classifier().predict_with_confidence([designation])[0]
    mood = state.sentiment().analyze([text])[0]
    family_id = int(state.clusterer().predict([designation])[0])
    families = {summary.cluster_id: summary for summary in state.clusterer().summaries_}

    st.subheader("Suggested routing")
    left, right = st.columns([2, 3])
    with left:
        st.metric("Category", prediction.label, help="Predicted by the trained classifier.")
        st.metric("Confidence", f"{prediction.confidence:.0%}")
        group = state.routing_table().get(prediction.label, "-")
        st.metric("Assignment group", group)

        if prediction.abstained:
            st.warning(
                "**Below the confidence threshold** - route to human triage rather than "
                "trusting this label. Exposing this flag is what makes partial automation safe.",
                icon=":material/pan_tool:",
            )
        else:
            st.success("Confident enough for automatic routing.", icon=":material/check:")

        if (
            declared_category
            and declared_category != "(not declared)"
            and declared_category != prediction.label
        ):
            st.info(
                f"The requester declared **{declared_category}**, the model predicts "
                f"**{prediction.label}**. Disagreements are the misroutes that generate "
                "reassignments - worth reviewing.",
                icon=":material/compare_arrows:",
            )
    with right:
        probabilities = pd.DataFrame(prediction.alternatives, columns=["label", "score"])
        st.altair_chart(charts.probability_bars(probabilities), use_container_width=True)

    st.divider()
    mood_column, family_column = st.columns(2)

    with mood_column:
        st.subheader("Requester sentiment")
        badge = {
            "negative": ":red[frustrated]",
            "neutral": ":gray[neutral]",
            "positive": ":green[positive]",
        }
        st.markdown(f"### {badge[mood.label]}")
        st.progress(
            (mood.score + 1) / 2, text=f"score {mood.score:+.2f} (-1 frustrated, +1 satisfied)"
        )
        if mood.evidence:
            st.caption(
                "Markers: " + ", ".join(f"`{marker}`" for marker in dict.fromkeys(mood.evidence))
            )
        if mood.label == "negative":
            st.warning(
                "Escalation risk: consider a callback before the SLA expires.",
                icon=":material/priority_high:",
            )
        st.caption(f"Backend: `{mood.backend}`. See docs/03-modeling.md for the weak-label caveat.")

    with family_column:
        st.subheader("Discovered family")
        summary = families.get(family_id)
        if summary is None:
            st.info("No family assigned (outlier).")
        else:
            st.markdown(f"### {summary.name}")
            st.caption(
                f"{summary.size} similar tickets ({summary.share:.1%} of volume) - "
                f"dominant declared category: {summary.dominant_category} "
                f"({(summary.category_purity or 0):.0%} pure)"
            )
            st.write(" ".join(f"`{term}`" for term in summary.top_terms))

    st.divider()
    st.subheader("Comparable past tickets")
    st.caption(
        "Nearest neighbours in the same vector space the families were built in - "
        "the basis for suggesting a known resolution."
    )
    neighbours = state.similar_tickets(designation, k=5)
    st.dataframe(
        neighbours,
        use_container_width=True,
        hide_index=True,
        column_config={
            "similarity": st.column_config.ProgressColumn(
                "similarity", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "resolution_hours": st.column_config.NumberColumn("resolved in (h)", format="%.1f"),
        },
    )

    reference = state.resolution_reference()
    if prediction.label in reference.index:
        row = reference.loc[prediction.label]
        st.info(
            f"Tickets of **{prediction.label}** are usually resolved in "
            f"**{row['median_hours']:.1f} h** (p90 {row['p90_hours']:.1f} h).",
            icon=":material/schedule:",
        )


def render() -> None:
    """Draw the triage page."""
    st.title("Ticket intake and triage")
    st.caption(
        "Fill the form as a requester would; every model in the project scores it live "
        "through the same functions the API exposes on `/v1/classify`, `/v1/sentiment` "
        "and `/v1/clusters/assign`."
    )

    example = st.selectbox("Prefill with an example", ["(empty)", *EXAMPLES])
    default_designation, default_description = EXAMPLES.get(example, ("", ""))

    with st.form("ticket"):
        designation = st.text_input(
            "Designation *",
            value=default_designation,
            placeholder="Short description, as typed at intake",
            help="This single field is what the classifier and the clusterer consume.",
        )
        description = st.text_area(
            "Description",
            value=default_description,
            placeholder="Free-text body - used by the sentiment model",
            height=120,
        )
        first, second, third = st.columns(3)
        channel = first.selectbox("Channel", ["portal", "email", "phone", "chat"])
        site = second.selectbox("Site", list(SITES))
        declared = third.selectbox(
            "Declared category",
            ["(not declared)", *sorted(state.tickets()["category"].unique())],
            help="Optional: lets the page compare the requester's choice with the prediction.",
        )
        impact_column, urgency_column = st.columns(2)
        impact = impact_column.select_slider(
            "Impact",
            options=[1, 2, 3],
            value=2,
            format_func=lambda v: {1: "High", 2: "Medium", 3: "Low"}[v],
        )
        urgency = urgency_column.select_slider(
            "Urgency",
            options=[1, 2, 3],
            value=2,
            format_func=lambda v: {1: "High", 2: "Medium", 3: "Low"}[v],
        )
        submitted = st.form_submit_button(
            "Analyse ticket", type="primary", use_container_width=True
        )

    if not submitted:
        return
    if not designation.strip():
        st.error("A designation is required: it is the model input.")
        return

    st.caption(
        f"Declared priority: **{_priority_from(impact, urgency).value}** - "
        f"channel `{channel}`, site `{site}`"
    )
    _render_results(designation.strip(), description.strip(), declared)
