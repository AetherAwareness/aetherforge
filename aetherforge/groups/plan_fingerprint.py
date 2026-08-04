"""
Immutable plan fingerprint for sector training waves.

Freezes group membership (+ train flags) so mid-wave paint cannot silently
mutate the set of experts being trained without an explicit new fingerprint.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aetherforge.groups.models import ExpertGroup, GroupPlan
from aetherforge.utils.hashing import json_hash
from aetherforge.utils.logging import get_logger

log = get_logger("groups.plan_fingerprint")


def membership_payload(plan: GroupPlan) -> dict[str, Any]:
    """Canonical serializable membership for hashing (order-stable)."""
    groups = []
    for g in sorted(plan.groups, key=lambda x: x.id):
        cells = sorted(
            [{"layer": c.layer, "expert": c.expert} for c in g.cells],
            key=lambda x: (x["layer"], x["expert"]),
        )
        groups.append(
            {
                "id": g.id,
                "name": g.name,
                "enabled": g.enabled,
                "train": g.train,
                "freeze": g.freeze,
                "domain": g.domain,
                "cells": cells,
                "n_cells": len(cells),
            }
        )
    return {
        "schema": "aetherforge.plan_membership.v1",
        "family": plan.family,
        "model_name": plan.model_name,
        "num_layers": plan.capacity.num_layers,
        "num_experts": plan.capacity.num_experts,
        "groups": groups,
    }


def plan_fingerprint(plan: GroupPlan) -> str:
    """Stable SHA-256 of membership payload."""
    return json_hash(membership_payload(plan))


@dataclass
class PlanFreeze:
    """Frozen plan snapshot at sector-wave start."""

    fingerprint: str
    frozen_at: float
    membership: dict[str, Any]
    train_group_ids: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aetherforge.plan_freeze.v1",
            "fingerprint": self.fingerprint,
            "frozen_at": self.frozen_at,
            "train_group_ids": self.train_group_ids,
            "notes": self.notes,
            "membership": self.membership,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlanFreeze":
        return cls(
            fingerprint=str(d["fingerprint"]),
            frozen_at=float(d.get("frozen_at") or 0),
            membership=d.get("membership") or {},
            train_group_ids=list(d.get("train_group_ids") or []),
            notes=str(d.get("notes") or ""),
        )


def freeze_plan(
    plan: GroupPlan,
    *,
    notes: str = "sector_wave_start",
) -> PlanFreeze:
    """Capture immutable fingerprint + membership at wave start."""
    payload = membership_payload(plan)
    fp = json_hash(payload)
    train_ids = [g.id for g in plan.enabled_train_groups()]
    freeze = PlanFreeze(
        fingerprint=fp,
        frozen_at=time.time(),
        membership=payload,
        train_group_ids=train_ids,
        notes=notes,
    )
    log.info(
        "Plan freeze fingerprint=%s train_groups=%d cells=%d",
        fp[:16],
        len(train_ids),
        sum(len(g["cells"]) for g in payload["groups"]),
    )
    return freeze


def verify_plan_fingerprint(
    plan: GroupPlan,
    expected: str,
) -> dict[str, Any]:
    """
    Compare live plan membership to expected fingerprint.

    Returns {ok, current, expected, changed_groups, message}.
    """
    current = plan_fingerprint(plan)
    ok = current == expected
    changed: list[str] = []
    if not ok:
        # find which groups differ by recomputing per-group hashes
        live = membership_payload(plan)
        # we only have expected as hash; report that mutation occurred
        for g in plan.groups:
            changed.append(g.id)
    return {
        "ok": ok,
        "current": current,
        "expected": expected,
        "mutated": not ok,
        "changed_group_ids": changed if not ok else [],
        "message": (
            "plan fingerprint matches freeze"
            if ok
            else "plan membership changed after freeze — refuse silent train"
        ),
    }


def assert_plan_frozen_or_raise(plan: GroupPlan, freeze: PlanFreeze) -> None:
    check = verify_plan_fingerprint(plan, freeze.fingerprint)
    if not check["ok"]:
        raise RuntimeError(
            f"Plan fingerprint mismatch: expected {freeze.fingerprint[:16]}… "
            f"got {check['current'][:16]}… — membership edited after wave freeze"
        )


def save_freeze(freeze: PlanFreeze, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(freeze.to_dict(), indent=2), encoding="utf-8")
    return path


def load_freeze(path: str | Path) -> PlanFreeze:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PlanFreeze.from_dict(data)


def group_cells_fingerprint(group: ExpertGroup) -> str:
    cells = sorted((c.layer, c.expert) for c in group.cells)
    return json_hash({"id": group.id, "cells": cells})
