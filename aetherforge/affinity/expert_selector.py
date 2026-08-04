"""Expert selection plan for AGPS: who trains, who freezes, mitosis flags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from aetherforge.affinity.probe import AffinityResult, affinity_to_expert_refs
from aetherforge.affinity.ranking import overloaded_experts, progressive_tiers, rank_experts
from aetherforge.models.moe_utils import ExpertRef
from aetherforge.utils.config import AffinityConfig
from aetherforge.utils.logging import get_logger

log = get_logger("affinity.selector")


@dataclass
class SelectionPlan:
    """Concrete freeze / train plan for Stage 3 ESFT."""

    domain: str
    selected: list[ExpertRef]
    frozen: list[ExpertRef]
    ranked_scores: list[tuple[int, int, float]]
    tiers: list[list[ExpertRef]] = field(default_factory=list)
    mitosis_candidates: list[tuple[int, int, float]] = field(default_factory=list)
    freeze_router: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "selected": [
                {
                    "layer": e.layer_idx,
                    "expert": e.expert_idx,
                    "module": e.module_name,
                }
                for e in self.selected
            ],
            "frozen_count": len(self.frozen),
            "ranked_scores": self.ranked_scores[:128],
            "tiers": [
                [{"layer": e.layer_idx, "expert": e.expert_idx} for e in tier]
                for tier in self.tiers
            ],
            "mitosis_candidates": self.mitosis_candidates[:32],
            "freeze_router": self.freeze_router,
            "metadata": self.metadata,
        }


class ExpertSelector:
    def __init__(self, config: AffinityConfig, posture: str = "specialist"):
        self.config = config
        self.posture = (posture or "specialist").lower()

    def select(
        self,
        result: AffinityResult,
        all_experts: Optional[list[ExpertRef]] = None,
    ) -> SelectionPlan:
        cfg = self.config
        total = max(result.num_layers * result.num_experts, 1)
        all_experts = all_experts or []

        # Resolve k for ranking + progressive tiers
        if cfg.top_k_fraction is not None and cfg.top_k_fraction > 0:
            k_eff = max(1, int(total * cfg.top_k_fraction))
        elif cfg.top_k_experts and cfg.top_k_experts > 0:
            k_eff = cfg.top_k_experts
        else:
            k_eff = min(32, total)

        ranked = rank_experts(
            result,
            top_k=cfg.top_k_experts if (cfg.top_k_experts and cfg.top_k_experts > 0) else None,
            top_k_fraction=cfg.top_k_fraction,
            min_score=cfg.min_affinity_score,
        )

        # Wide: train essentially the whole lattice (still PEFT-masked elsewhere)
        if self.posture == "wide" or (
            cfg.top_k_fraction is not None and cfg.top_k_fraction >= 0.99
        ):
            if all_experts:
                selected_refs = list(all_experts)
                ranked = [
                    (e.layer_idx, e.expert_idx, 1.0) for e in all_experts
                ] or ranked
            else:
                selected_refs = affinity_to_expert_refs(result, all_experts, ranked)
        else:
            selected_refs = affinity_to_expert_refs(result, all_experts, ranked)

        selected_keys = {(e.layer_idx, e.expert_idx) for e in selected_refs}
        frozen = [
            e
            for e in all_experts
            if (e.layer_idx, e.expert_idx) not in selected_keys
        ]

        tiers_raw = []
        if cfg.progressive_unfreeze:
            if self.posture == "wide":
                tier_sizes = [
                    max(1, k_eff // 4),
                    max(1, k_eff // 2),
                    k_eff,
                    total,
                ]
            elif self.posture == "broad":
                tier_sizes = [
                    max(1, k_eff // 3),
                    k_eff,
                    min(int(k_eff * 1.5), total),
                    min(int(k_eff * 2.5), total),
                ]
            else:
                tier_sizes = [
                    max(1, k_eff // 2),
                    k_eff,
                    min(k_eff * 2, total),
                ]
            for tier in progressive_tiers(result, tier_sizes):
                tiers_raw.append(affinity_to_expert_refs(result, all_experts, tier))

        mitosis = overloaded_experts(result, threshold=cfg.mitosis_overload_threshold)

        plan = SelectionPlan(
            domain=result.domain,
            selected=selected_refs,
            frozen=frozen if cfg.freeze_low_affinity else [],
            ranked_scores=ranked,
            tiers=tiers_raw,
            mitosis_candidates=mitosis,
            freeze_router=cfg.freeze_router_initially,
            metadata={
                "top_k": k_eff,
                "top_k_fraction": cfg.top_k_fraction,
                "posture": self.posture,
                "progressive_unfreeze": cfg.progressive_unfreeze,
                "n_all_experts": len(all_experts),
                "affinity_entropy_mean": (
                    sum(result.entropy_per_layer) / len(result.entropy_per_layer)
                    if result.entropy_per_layer
                    else 0.0
                ),
            },
        )
        log.info(
            "Selection plan posture=%s domain=%s selected=%d frozen=%d "
            "mitosis=%d freeze_router=%s",
            self.posture,
            plan.domain,
            len(plan.selected),
            len(plan.frozen),
            len(plan.mitosis_candidates),
            plan.freeze_router,
        )
        return plan
