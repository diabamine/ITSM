"""Page 1 - fill in a ticket like a requester would, and see what the AI makes of it.

On submit: the LLM extracts structured fields, a k-NN finds comparable past
tickets, and rules turn both into alerts. This page only collects the form and
displays the result; the work is done in :mod:`backend`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.alerts import ticket_alerts
from backend.llm import Enrichment
from backend.schema import LEVELS, Col, derive_priority
from frontend import state
from frontend.components import render_alerts

EXAMPLES = {
    "VPN keeps dropping (frustrated, contains personal data)": dict(
        title="VPN qui se déconnecte",
        description="Depuis ce matin le VPN se coupe toutes les dix minutes en télétravail et je perds mes sessions. "
        "C'est la troisième fois que je le signale, toujours rien. Je suis joignable au 06 12 34 56 78 "
        "ou sur jean.martin@exemple.fr.",
        service="Connectivité",
        urgency="High",
        impact="High",
    ),
    "Software installation request (neutral)": dict(
        title="Installation d'un logiciel de modélisation",
        description="Bonjour, j'ai besoin que le logiciel de modélisation soit installé sur mon poste pour un projet "
        "qui démarre la semaine prochaine. Merci d'avance.",
        service="Logiciels & licences",
        urgency="Low",
        impact="Low",
    ),
    "Suspicious email (security)": dict(
        title="Courriel suspect demandant mes identifiants",
        description="J'ai reçu un message qui ressemble à du phishing me demandant de confirmer mon mot de passe, "
        "je n'ai pas cliqué sur le lien.",
        service="Sécurité",
        urgency="Medium",
        impact="Medium",
    ),
}
_BLANK = "(blank form)"


def _select(container: st.delta_generator.DeltaGenerator, label: str, options: list[str], default: str = "") -> str:
    """A select box whose first choice is empty (optional field), preselecting ``default``."""
    choices = ["", *options]
    index = choices.index(default) if default in choices else 0
    return container.selectbox(label, choices, index=index, format_func=lambda v: v or "-")


def render() -> None:
    """Draw the ticket form page."""
    st.title("New ticket")
    st.caption(
        "Fill the form as a requester would. On submit, an LLM reads the text and fills the analysis fields, "
        "similar past tickets are retrieved, and alerts are raised when something needs attention."
    )
    store = state.get_store()

    example_name = st.selectbox("Prefill with an example", [_BLANK, *EXAMPLES])
    example = EXAMPLES.get(example_name, {})

    with st.form("ticket_form"):
        title = st.text_input(
            "Short description *", value=example.get("title", ""), placeholder="One line, as typed at intake"
        )
        description = st.text_area(
            "Description *",
            value=example.get("description", ""),
            height=140,
            placeholder="Free text: what happens, since when, who is affected",
        )
        a, b, c = st.columns(3)
        service = _select(a, "Service", store.options(Col.SERVICE), example.get("service", ""))
        group = _select(b, "Assignment group", store.options(Col.GROUP))
        contact = _select(c, "Contact type", store.options(Col.CONTACT_TYPE))
        d, e, f = st.columns(3)
        urgency = d.select_slider("Urgency", LEVELS, value=example.get("urgency", "Medium"))
        impact = e.select_slider("Impact", LEVELS, value=example.get("impact", "Medium"))
        site = _select(f, "Site", store.options(Col.SITE))
        entity = _select(st, "Entity", store.options(Col.ENTITY))
        submitted = st.form_submit_button("Analyse ticket", type="primary", width="stretch")

    if submitted:
        if not title.strip() or not description.strip():
            st.error("Short description and description are required.")
            return
        ticket = {
            Col.TITLE: title.strip(),
            Col.DESCRIPTION: description.strip(),
            Col.SERVICE: service,
            Col.URGENCY: urgency,
            Col.IMPACT: impact,
            Col.PRIORITY: derive_priority(impact, urgency),
            Col.GROUP: group,
            Col.ENTITY: entity,
            Col.SITE: site,
            Col.CONTACT_TYPE: contact,
        }
        with st.spinner("Reading the ticket and searching similar ones..."):
            enrichment = state.get_enricher().enrich(ticket)
            query = {**ticket, **enrichment.as_columns()}
            neighbours = state.get_index().search(query, k=30)
            alerts = ticket_alerts(ticket, enrichment, neighbours, store.reference_date)
        st.session_state["ticket_result"] = dict(
            ticket=ticket, enrichment=enrichment, neighbours=neighbours, alerts=alerts
        )

    result = st.session_state.get("ticket_result")
    if result:
        _render_result(**result)


def _render_result(ticket: dict, enrichment: Enrichment, neighbours: pd.DataFrame, alerts: list) -> None:
    """Display the analysis, the alerts and the similar tickets."""
    st.divider()
    st.subheader("AI analysis")
    if enrichment.source == "llm":
        st.caption("Fields inferred by the LLM, validated against the allowed values.")
    else:
        st.warning(enrichment.notice, icon=":material/smart_toy:")

    fields = [
        ("Incident type", enrichment.incident_type),
        ("Request type", enrichment.request_type),
        ("Tone", enrichment.tone),
        ("Scope", enrichment.scope),
        ("Affected system", enrichment.affected_system),
        ("Probable root cause", enrichment.root_cause),
    ]
    columns = st.columns(3)
    for position, (label, value) in enumerate(fields):
        with columns[position % 3].container(border=True):
            st.caption(label)
            st.markdown(f"**{value}**")
    st.caption(f"Priority derived from impact x urgency: **{ticket[Col.PRIORITY]}**")

    st.subheader("Alerts")
    render_alerts(alerts, "No alert raised for this ticket.")

    st.subheader("Similar past tickets")
    st.caption(
        "Nearest neighbours on the wording (embeddings) and on the context (service, group, type, system, site)."
    )
    shortlist = neighbours.head(5)[
        [
            Col.ID,
            Col.TITLE,
            Col.OPENED,
            Col.STAGE,
            Col.ROOT_CAUSE,
            Col.RESOLUTION_SUMMARY,
            Col.RESOLUTION_HOURS,
            "Similarity",
        ]
    ]
    st.dataframe(
        shortlist,
        hide_index=True,
        column_config={
            Col.OPENED: st.column_config.DatetimeColumn("Opened", format="YYYY-MM-DD HH:mm"),
            Col.RESOLUTION_HOURS: st.column_config.NumberColumn("Resolved in (h)", format="%.1f"),
            "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )

    top = neighbours.head(10)
    summaries = top[Col.RESOLUTION_SUMMARY].dropna().astype(str)
    summaries = summaries[summaries != ""]
    if len(summaries):
        best, count = summaries.value_counts().index[0], int(summaries.value_counts().iloc[0])
        hours = pd.to_numeric(top[Col.RESOLUTION_HOURS], errors="coerce").median()
        st.info(
            f"**Most common resolution among the 10 closest tickets** ({count}/{len(summaries)}): {best} "
            f"Typical time to resolve: {hours:.1f} h.",
            icon=":material/lightbulb:",
        )

    with st.expander("How is similarity computed?"):
        st.markdown(
            "`similarity = 0.7 x text similarity + 0.3 x share of matching categories`.  \n"
            "Text similarity is the cosine between embeddings of *short description + description*. "
            "Categories are service, assignment group, incident type, affected system and site: "
            "the ones the LLM inferred for this ticket count as well."
        )
