"""Privacy page: see what personal data is in the text, and what happens to it.

Anonymisation is the one part of this project that a reviewer will want to
*watch*, not read a metric about. The page therefore does two things: it lets
anyone paste a ticket and see exactly which characters were identified and what
they became, and it shows the measured quality of that process on the whole
corpus.
"""

from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

from itsm.app import state
from itsm.pipelines.reporting import read_report
from itsm.privacy.entities import DIRECT_IDENTIFIERS, PiiSpan

_EXAMPLE = (
    "Le poste de Camille Dubois ne demarre plus depuis la mise a jour. "
    "Merci de la contacter au 06 12 34 56 78 ou sur camille.dubois@groupe-exemple.example. "
    "Son matricule est EMP-45231 et elle est nee le 14/07/1988. "
    "Le RIB a corriger est FR76 3000 1007 9412 3456 7890 185."
)

_COLOURS = {
    "PERSON": "#fecaca",
    "EMAIL": "#bfdbfe",
    "PHONE": "#bbf7d0",
    "EMPLOYEE_ID": "#fde68a",
    "NATIONAL_ID": "#fbcfe8",
    "IBAN": "#ddd6fe",
    "CREDIT_CARD": "#fed7aa",
    "ADDRESS": "#a5f3fc",
    "DATE_OF_BIRTH": "#e9d5ff",
    "IP_ADDRESS": "#d9f99d",
}


def _highlight(text: str, spans: list[PiiSpan]) -> str:
    """Render the text as HTML with every detected span marked.

    Built in a single left-to-right pass so the span offsets - which refer to
    the *raw* string - stay valid, and each slice is HTML-escaped separately so
    the escaping never shifts them.
    """
    pieces: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: item.start):
        if span.start < cursor:  # defensive: spans should already be disjoint
            continue
        colour = _COLOURS.get(str(span.entity), "#e5e7eb")
        pieces.append(html.escape(text[cursor : span.start]))
        pieces.append(
            f'<mark style="background:{colour};padding:0 2px;border-radius:3px" '
            f'title="{html.escape(str(span.entity))} ({span.score:.2f})">'
            f"{html.escape(text[span.start : span.end])}</mark>"
        )
        cursor = span.end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def _render_live_demo(backend: str) -> None:
    """The paste-a-ticket panel."""
    st.subheader("Try it")
    text = st.text_area("Ticket text", value=_EXAMPLE, height=140)
    if not text.strip():
        return

    anonymiser = state.anonymiser(backend)
    anonymised, spans, mapping = anonymiser.anonymise_text(text)

    columns = st.columns(3)
    columns[0].metric("Identifiers found", len(spans))
    columns[1].metric(
        "Direct identifiers",
        sum(1 for span in spans if span.entity in DIRECT_IDENTIFIERS),
        help="One value is enough to name someone: name, email, phone, national ID, IBAN, card.",
    )
    columns[2].metric("Characters replaced", sum(span.length for span in spans))

    st.markdown("**Detected**", help="Hover a highlight for its entity type and confidence.")
    st.markdown(
        f'<div style="line-height:1.9">{_highlight(text, spans)}</div>', unsafe_allow_html=True
    )

    st.markdown("**Anonymised output**")
    st.code(anonymised, language=None, wrap_lines=True)

    if mapping:
        st.markdown("**Replacements**")
        st.caption(
            "Each identifier becomes the tag of its kind. A tag is unmistakable, so a leak "
            "and a correct redaction never look alike - which is not true of a realistic "
            "fake name. Nothing of the value survives, and nothing is reversible."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "entity": str(span.entity),
                        "original": span.text,
                        "replaced by": mapping.get(span.text, "-"),
                    }
                    for span in spans
                    if span.text in mapping
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def _render_corpus_quality() -> None:
    """The measured quality section, read from the evaluation report."""
    report = read_report("privacy", state.settings())
    st.subheader("Measured quality")
    if report is None:
        st.info(
            "No evaluation report yet. Run `itsm privacy eval --backend rules "
            "--backend transformer --backend presidio --isolate` to produce one.",
            icon=":material/science:",
        )
        return

    st.caption(
        f"Scored on {report['n_tickets']} tickets of the `{report['column']}` field against the "
        "generator's character-level ground truth."
    )

    rows: list[dict[str, object]] = []
    for name, entry in report["backends"].items():
        detection = entry.get("detection") or {}
        residual = entry.get("residual") or {}
        rows.append(
            {
                "backend": name,
                "coverage recall": detection.get("recall_coverage"),
                "exact recall": detection.get("recall_strict"),
                "precision": detection.get("precision_partial"),
                "leak rate (direct ids)": detection.get("leak_rate_direct_identifiers"),
                "residual in output": residual.get("residual_rate"),
                "utility cost (macro-F1)": (entry.get("utility") or {}).get("cost_of_redaction"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    entity_rows = [
        {"backend": name, "entity": entity, "metric": metric, "value": values[metric]}
        for name, entry in report["backends"].items()
        for entity, values in ((entry.get("detection") or {}).get("per_entity") or {}).items()
        for metric in ("recall", "precision")
    ]
    if entity_rows:
        st.markdown("**Per entity type**")
        chart = (
            alt.Chart(pd.DataFrame(entity_rows))
            .mark_bar()
            .encode(
                x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                y=alt.Y("entity:N", sort="-x", title=None),
                color=alt.Color("backend:N", title=None),
                row=alt.Row("metric:N", title=None),
                tooltip=["backend:N", "entity:N", "metric:N", "value:Q"],
            )
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)

    probe = report.get("unseen_name_probe")
    if probe:
        st.markdown("**The number that decides which backend to deploy**")
        st.caption(
            f"Recall on {probe['n_names']} names present in neither the generator pools nor the "
            "rule engine's gazetteer. The headline table above flatters the rules, because the "
            "corpus draws its names from the very list the gazetteer holds; this probe does not."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"backend": name, "recall on unseen names": values["recall"]}
                    for name, values in probe["backends"].items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def render() -> None:
    """Draw the privacy page."""
    st.title("Anonymisation")
    st.caption(
        "Ticket free text carries names, mailboxes and phone numbers nobody chose to collect. "
        "This page shows what is detected, what replaces it, and how well that works "
        "(`itsm.privacy`)."
    )

    settings = state.settings()
    backends = ["rules", "transformer", "presidio"]
    backend = st.radio(
        "Detector",
        backends,
        index=backends.index(settings.anonymisation_backend),
        horizontal=True,
        help=(
            "`rules`: offline patterns, triggers and a gazetteer - no download. "
            "`transformer`: a multilingual NER model from Hugging Face for names, with the same "
            "rules for formats. Needs the `hf` extra. "
            "`presidio`: Microsoft Presidio's analyser - a spaCy pipeline for names plus "
            "checksum-validated formats - running **alone**, with no rule layer underneath, so "
            "it can be benchmarked on its own recognisers. Needs the `presidio` extra and one "
            "spaCy model."
        ),
    )

    if settings.use_anonymised:
        st.success(
            "`ITSM_USE_ANONYMISED` is on: every other page of this app is reading the "
            "anonymised extract.",
            icon=":material/lock:",
        )
    else:
        st.warning(
            "`ITSM_USE_ANONYMISED` is off: the rest of the app reads the raw extract. Run "
            "`itsm privacy anonymise` and set the flag to change that.",
            icon=":material/lock_open:",
        )

    try:
        _render_live_demo(backend)
    except RuntimeError as error:
        st.error(str(error), icon=":material/error:")

    st.divider()
    _render_corpus_quality()
