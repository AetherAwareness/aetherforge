"""
Elastic Expert Lifecycle — rebirth, mitosis (split), optional prune.

These operations produce *plans* and metadata. Applying weight surgery
to live MoE modules is family-specific and done in adapters / export.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aetherforge.lifecycle.monitor import ExpertUtilizationMonitor, UtilizationReport
from aetherforge.utils.config import LifecycleConfig
from aetherforge.utils.logging import get_logger

log = get_logger("lifecycle.birth_death")


@dataclass
class LifecycleAction:
    action: str  # rebirth | mitosis | prune | noop
    layer: int
    expert: int
    reason: str
    rel_util: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LifecyclePlan:
    actions: list[LifecycleAction] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "actions": [
                {
                    "action": a.action,
                    "layer": a.layer,
                    "expert": a.expert,
                    "reason": a.reason,
                    "rel_util": a.rel_util,
                    "meta": a.meta,
                }
                for a in self.actions
            ],
        }


class ExpertLifecycleManager:
    def __init__(self, config: LifecycleConfig):
        self.config = config
        self.monitor = ExpertUtilizationMonitor(
            low_threshold=config.util_low_threshold,
            high_threshold=config.util_high_threshold,
        )

    def plan_from_report(self, report: UtilizationReport) -> LifecyclePlan:
        if not self.config.enabled:
            return LifecyclePlan(actions=[])

        actions: list[LifecycleAction] = []
        if self.config.allow_rebirth:
            for li, ei, rel in report.low[:16]:
                actions.append(
                    LifecycleAction(
                        action="rebirth",
                        layer=li,
                        expert=ei,
                        reason="underutilized",
                        rel_util=rel,
                        meta={"strategy": "reinit_on_new_cluster"},
                    )
                )
        if self.config.allow_mitosis:
            for li, ei, rel in report.high[:8]:
                actions.append(
                    LifecycleAction(
                        action="mitosis",
                        layer=li,
                        expert=ei,
                        reason="overloaded",
                        rel_util=rel,
                        meta={"strategy": "duplicate_and_specialize"},
                    )
                )
        if self.config.allow_prune:
            for li, ei, rel in report.low[-4:]:
                actions.append(
                    LifecycleAction(
                        action="prune",
                        layer=li,
                        expert=ei,
                        reason="chronic_underuse",
                        rel_util=rel,
                        meta={"warning": "destructive"},
                    )
                )

        plan = LifecyclePlan(actions=actions)
        log.info("Lifecycle plan: %d actions", len(actions))
        return plan

    def save(self, plan: LifecyclePlan, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, indent=2)
