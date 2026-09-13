"""Exploratory analysis: designation clustering and lifecycle statistics."""

from itsm.analysis.clustering import ClusteringResult, ClusterSummary, DesignationClusterer
from itsm.analysis.lifecycle import (
    aging_backlog,
    backlog_timeline,
    dwell_times,
    followed_ticket,
    kpi_overview,
    kpis_by,
    rework_profile,
    state_transitions,
    volume_timeline,
)

__all__ = [
    "ClusterSummary",
    "ClusteringResult",
    "DesignationClusterer",
    "aging_backlog",
    "backlog_timeline",
    "dwell_times",
    "followed_ticket",
    "kpi_overview",
    "kpis_by",
    "rework_profile",
    "state_transitions",
    "volume_timeline",
]
