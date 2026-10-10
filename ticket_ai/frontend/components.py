"""Small reusable widgets shared by the pages."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from backend.alerts import Alert

_ICONS = {"critical": ":material/error:", "warning": ":material/warning:", "info": ":material/info:"}


def render_alerts(alerts: list[Alert], empty_message: str = "No alert raised.") -> None:
    """Show alerts as coloured boxes, most severe first (the caller sorts them)."""
    if not alerts:
        st.success(empty_message, icon=":material/check_circle:")
        return
    box = {"critical": st.error, "warning": st.warning, "info": st.info}
    for alert in alerts:
        text = f"**{alert.title}**" + (f"  \n{alert.detail}" if alert.detail else "")
        box[alert.level](text, icon=_ICONS[alert.level])


def show_html(document: str, height: int = 900) -> None:
    """Display a full HTML document inline.

    Uses ``st.iframe`` where it exists and ``components.html`` on older versions.
    Only pass HTML this app generated with every dynamic value escaped (as the
    report does): an iframe runs whatever it is given.
    """
    if hasattr(st, "iframe"):
        st.iframe(document, height=height)
    else:  # pragma: no cover - older Streamlit
        components.html(document, height=height, scrolling=True)


def llm_status(mode: str) -> None:
    """One-line sidebar indicator of how ticket analysis is being done."""
    if mode == "llm":
        st.success("LLM connected", icon=":material/smart_toy:")
    else:
        st.warning(
            "Demo mode: no LLM configured, a local keyword heuristic is used. "
            "Set `LLM_BASE_URL` and `LLM_MODEL` to enable the real thing.",
            icon=":material/smart_toy:",
        )
