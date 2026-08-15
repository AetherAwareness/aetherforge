"""Affinity-Guided Progressive Specialization (AGPS / DEAM)."""

from aetherforge.affinity.probe import AffinityProbe, AffinityResult
from aetherforge.affinity.ranking import rank_experts
from aetherforge.affinity.expert_selector import ExpertSelector, SelectionPlan
from aetherforge.affinity.theme_probe import attach_theme_scores, collect_theme_probes

__all__ = [
    "AffinityProbe",
    "AffinityResult",
    "rank_experts",
    "ExpertSelector",
    "SelectionPlan",
    "attach_theme_scores",
    "collect_theme_probes",
]
