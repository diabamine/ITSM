"""Test data: a small extract built here (not part of the app), with a VPN surge in the last 3 weeks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from backend import data as d

END = pd.Timestamp("2026-06-30 15:00")
BASE = [  # (incident type, system, service, title)
    ("Réseau", "Wi-Fi", "Connectivité", "Pas de Wi-Fi dans le bâtiment"),
    ("Accès", "MFA", "Identité", "Double authentification impossible"),
    ("Messagerie", "Outlook", "Messagerie", "Outlook bloqué sur tentative de connexion"),
    ("Poste", "Portable", "Poste de travail", "Le portable ne démarre plus"),
]
SURGE = ("Réseau", "VPN", "Connectivité", "Déconnexions fréquentes du VPN")


def make_tickets(days: int = 300, seed: int = 1) -> pd.DataFrame:
    """~1200 tickets over ``days`` days, weekdays only, ending on ``END``."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in pd.date_range(END.normalize() - pd.Timedelta(days=days - 1), END.normalize()):
        if day.dayofweek >= 5:
            continue
        surge = day > END - pd.Timedelta(days=21)
        picks = [BASE[i] for i in rng.integers(0, len(BASE), rng.poisson(5))] + [SURGE] * (rng.poisson(8) if surge else 0)
        for incident, system, service, title in picks:
            opened = day + pd.Timedelta(hours=int(rng.integers(8, 18)), minutes=int(rng.integers(0, 60)))
            if opened > END:
                continue
            hours = float(rng.lognormal(1.5, 0.8))
            resolved = opened + pd.Timedelta(hours=hours)
            rows.append({
                d.ID: f"INC{len(rows) + 1:06d}", d.OPENED: opened, d.RESOLVED: resolved if resolved <= END else pd.NaT,
                d.TITLE: title, d.DESCRIPTION: f"{title}. Depuis ce matin, merci de traiter.", d.SERVICE: service,
                d.GROUP: "N2 Support", d.SITE: str(rng.choice(["Paris", "Brest"])), d.ENTITY: "Entité Nord",
                d.CONTACT: "Portail", d.URGENCY: str(rng.choice(d.LEVELS)), d.IMPACT: str(rng.choice(d.LEVELS)),
                d.RESOLUTION_H: round(hours, 1), d.SLA: d.SLA_BREACHED if hours > 8 else "Respecté",
                d.REOPENED: int(rng.poisson(0.1)), d.INCIDENT_TYPE: incident, d.AFFECTED_SYSTEM: system,
                d.TONE: str(rng.choice(["Neutre", "Frustré"], p=[0.7, 0.3] if surge else [0.9, 0.1])),
                d.SCOPE: "Utilisateur unique", d.ROOT_CAUSE: "Mise à jour du client VPN" if system == "VPN" else "Autre",
                d.RESOLUTION_SUMMARY: "Retour à la version précédente" if system == "VPN" else "Résolu",
            })  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tickets() -> pd.DataFrame:
    return make_tickets()


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory, tickets) -> str:
    """The extract as a French export would be: ``;`` separator, day-first dates, decimal commas."""
    path = tmp_path_factory.mktemp("data") / "extract.csv"
    tickets.to_csv(path, sep=";", decimal=",", date_format="%d/%m/%Y %H:%M:%S", index=False)
    return str(path)


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Cached embeddings from one test must not leak into the next."""
    st.cache_data.clear()
