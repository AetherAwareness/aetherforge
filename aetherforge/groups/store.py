"""Persist / load Expert Group plans (JSON)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from aetherforge.groups.capacity import recompute_all_group_capacities
from aetherforge.groups.models import ExpertGroup, GroupPlan
from aetherforge.utils.logging import get_logger

log = get_logger("groups.store")


def save_group_plan(plan: GroupPlan, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plan.updated_at = time.time()
    path.write_text(json.dumps(plan.to_dict(), indent=2, default=str), encoding="utf-8")
    log.info("Saved group plan → %s (%d groups)", path, len(plan.groups))
    return path


def load_group_plan(path: str | Path) -> GroupPlan:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    plan = GroupPlan.model_validate(raw)
    recompute_all_group_capacities(plan.groups, plan.capacity)
    return plan


def default_plan_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "expert_groups.json"


def apply_plan_patch(plan: GroupPlan, patch: dict[str, Any]) -> GroupPlan:
    """
    Apply dashboard edits.

    Supported patch shapes:
      { "target_num_groups": 8 }
      { "notes": "..." }
      { "update_group": { "id": "...", "name": "...", "enabled": true, ... } }
      { "create_group": { "name": "...", "cells": [{"layer":0,"expert":1}] } }
      { "delete_group": "id" }
      { "move_cells": { "from": "id"|null, "to": "id", "cells": [{"layer":0,"expert":1}] } }
      { "set_train_groups": ["id1", "id2"] }  # only these train=true
    """
    if "target_num_groups" in patch:
        plan.target_num_groups = int(patch["target_num_groups"])
    if "target_active_fire_ratio" in patch:
        plan.target_active_fire_ratio = float(patch["target_active_fire_ratio"])
    if "notes" in patch:
        plan.notes = str(patch["notes"])

    if "delete_group" in patch:
        gid = patch["delete_group"]
        plan.groups = [g for g in plan.groups if g.id != gid]

    if "create_group" in patch:
        body = patch["create_group"] or {}
        g = ExpertGroup(
            name=body.get("name") or f"sector_{len(plan.groups) + 1}",
            color=body.get("color") or "#5b9dff",
            description=body.get("description") or "",
            domain=body.get("domain"),
            curated_path=body.get("curated_path"),
            domain_pack=body.get("domain_pack"),
            topics=list(body.get("topics") or []),
            keywords=list(body.get("keywords") or []),
            enabled=bool(body.get("enabled", True)),
            train=bool(body.get("train", True)),
            notes=body.get("notes") or "",
        )
        for cell in body.get("cells") or []:
            g.add_cell(int(cell["layer"]), int(cell["expert"]))
        recompute_all_group_capacities([g], plan.capacity)
        plan.groups.append(g)

    if "update_group" in patch:
        body = patch["update_group"] or {}
        gid = body.get("id")
        g = plan.group_by_id(gid) if gid else None
        if g:
            for field in (
                "name",
                "color",
                "description",
                "enabled",
                "train",
                "freeze",
                "domain",
                "domain_pack",
                "curated_path",
                "data_weight",
                "notes",
            ):
                if field in body:
                    setattr(g, field, body[field])
            if "topics" in body:
                g.topics = list(body["topics"] or [])
            if "keywords" in body:
                g.keywords = list(body["keywords"] or [])
            if "tags" in body:
                g.tags = list(body["tags"] or [])
            if "cells" in body:
                g.cells = []
                for cell in body["cells"] or []:
                    g.add_cell(int(cell["layer"]), int(cell["expert"]))
            g.updated_at = time.time()
            recompute_all_group_capacities([g], plan.capacity)

    if "move_cells" in patch:
        body = patch["move_cells"] or {}
        to_id = body.get("to")
        from_id = body.get("from")
        to_g = plan.group_by_id(to_id) if to_id else None
        from_g = plan.group_by_id(from_id) if from_id else None
        cells = body.get("cells") or []
        exclusive = body.get("exclusive", True)
        # Optional: unassign (remove from all groups) when to is null and unassign=true
        if body.get("unassign") or to_id is None and body.get("remove"):
            for cell in cells:
                li, ei = int(cell["layer"]), int(cell["expert"])
                for g in plan.groups:
                    g.remove_cell(li, ei)
            recompute_all_group_capacities(plan.groups, plan.capacity)
        elif to_g:
            for cell in cells:
                li, ei = int(cell["layer"]), int(cell["expert"])
                if from_g:
                    from_g.remove_cell(li, ei)
                if exclusive:
                    for g in plan.groups:
                        if g.id != to_g.id:
                            g.remove_cell(li, ei)
                to_g.add_cell(li, ei)
            recompute_all_group_capacities(plan.groups, plan.capacity)

    if "set_train_groups" in patch:
        ids = set(patch["set_train_groups"] or [])
        for g in plan.groups:
            g.train = g.id in ids
            if g.id in ids:
                g.enabled = True

    plan.updated_at = time.time()
    return plan
