"""Affinity-Guided Progressive Specialization (AGPS / DEAM)."""

from aetherforge.affinity.probe import AffinityProbe, AffinityResult
from aetherforge.affinity.ranking import rank_experts
from aetherforge.affinity.expert_selector import ExpertSelector, SelectionPlan

__all__ = [
    "AffinityProbe",
    "AffinityResult",
    "rank_experts",
    "ExpertSelector",
    "SelectionPlan",
]
