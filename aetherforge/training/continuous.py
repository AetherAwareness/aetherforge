"""Continuous / online update mode skeleton — production error → targeted ESFT."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aetherforge.utils.config import ContinuousConfig
from aetherforge.utils.logging import get_logger

log = get_logger("training.continuous")


@dataclass
class ErrorCluster:
    cluster_id: str
    examples: list[str]
    specialty_hint: Optional[str] = None
    count: int = 0


@dataclass
class ContinuousUpdatePlan:
    clusters: list[ErrorCluster] = field(default_factory=list)
    experts_to_update: list[tuple[int, int]] = field(default_factory=list)
    synthetic_budget: int = 0
    federated: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "count": c.count or len(c.examples),
                    "specialty_hint": c.specialty_hint,
                    "examples_head": c.examples[:5],
                }
                for c in self.clusters
            ],
            "experts_to_update": self.experts_to_update,
            "synthetic_budget": self.synthetic_budget,
            "federated": self.federated,
            "created_at": self.created_at,
        }


class ContinuousController:
    """
    Production logger → error clustering → generate data → targeted AGPS update.

    This module is the control-plane skeleton; wire real production logs later.
    """

    def __init__(self, config: ContinuousConfig):
        self.config = config

    def cluster_errors(self, errors: list[dict[str, Any]]) -> list[ErrorCluster]:
        """Naive keyword clustering (replace with embedding cluster in production)."""
        buckets: dict[str, list[str]] = {}
        for err in errors:
            text = str(err.get("text") or err.get("error") or err)
            key = text.strip().lower()[:48] or "misc"
            # coarse bucket by first nontrivial word
            words = [w for w in key.split() if len(w) > 3]
            bucket = words[0] if words else "misc"
            buckets.setdefault(bucket, []).append(text)

        clusters = []
        for i, (k, exs) in enumerate(buckets.items()):
            if len(exs) < self.config.error_cluster_min and len(errors) >= self.config.error_cluster_min:
                continue
            clusters.append(
                ErrorCluster(cluster_id=f"c{i}_{k}", examples=exs, specialty_hint=k, count=len(exs))
            )
        clusters.sort(key=lambda c: c.count, reverse=True)
        return clusters

    def plan_update(
        self,
        errors: list[dict[str, Any]],
        affinity_ranked: Optional[list[tuple[int, int, float]]] = None,
    ) -> ContinuousUpdatePlan:
        clusters = self.cluster_errors(errors)
        experts: list[tuple[int, int]] = []
        if affinity_ranked:
            for li, ei, _ in affinity_ranked[: self.config.max_experts_per_update]:
                experts.append((li, ei))

        plan = ContinuousUpdatePlan(
            clusters=clusters,
            experts_to_update=experts,
            synthetic_budget=sum(c.count for c in clusters) * 5,
            federated=self.config.federated,
        )
        log.info(
            "Continuous plan: %d clusters, %d experts, budget=%d",
            len(plan.clusters),
            len(plan.experts_to_update),
            plan.synthetic_budget,
        )
        return plan

    def save_plan(self, plan: ContinuousUpdatePlan, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, indent=2)
