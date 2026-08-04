"""
Studio helpers: lattice views, selection → ESFT plan, analysis JSON for UI.
"""

from __future__ import annotations

from typing import Any, Optional

from aetherforge.affinity.expert_selector import SelectionPlan
from aetherforge.groups.capacity import estimate_capacity, estimate_group_capacity
from aetherforge.groups.cluster import auto_partition_groups, merge_affinity_into_plan
from aetherforge.groups.models import ExpertCell, ExpertGroup, GroupPlan, ModelCapacity
from aetherforge.models.moe_utils import ExpertRef
from aetherforge.utils.logging import get_logger

log = get_logger("groups.studio")


def create_studio_plan(
    *,
    family: str = "deepseek_v4_flash",
    model_name: str = "",
    num_groups: Optional[int] = None,
    strategy: str = "active_slots",
    affinity: Optional[dict[str, Any]] = None,
    arch_layers: Optional[int] = None,
    arch_experts: Optional[int] = None,
    arch_topk: Optional[int] = None,
    total_params_b: Optional[float] = None,
    active_params_b: Optional[float] = None,
) -> GroupPlan:
    cap = estimate_capacity(
        family=family,
        model_name=model_name,
        num_layers=arch_layers,
        num_experts=arch_experts,
        top_k=arch_topk,
        total_params_b=total_params_b,
        active_params_b=active_params_b,
    )
    matrix = None
    ranked = None
    if affinity:
        matrix = affinity.get("affinity") or affinity.get("routing_freq")
        ranked = affinity.get("ranked")
    plan = auto_partition_groups(
        cap,
        num_groups=num_groups,
        strategy=strategy,  # type: ignore[arg-type]
        affinity_matrix=matrix,
        ranked=ranked,
        model_name=model_name,
    )
    if matrix or ranked:
        merge_affinity_into_plan(plan, matrix, ranked)
    return plan


def group_plan_to_selection(
    plan: GroupPlan,
    domain: str = "general",
) -> SelectionPlan:
    """Convert enabled train groups into an ESFT SelectionPlan."""
    cells = plan.selected_cells_for_training()
    selected = [
        ExpertRef(
            layer_idx=c.layer,
            expert_idx=c.expert,
            module_name=f"model.layers.{c.layer}.mlp.experts.{c.expert}",
            family=plan.family,
        )
        for c in cells
    ]
    frozen = []
    selected_keys = {(c.layer, c.expert) for c in cells}
    for li in range(plan.capacity.num_layers):
        for ei in range(plan.capacity.num_experts):
            if (li, ei) not in selected_keys:
                frozen.append(
                    ExpertRef(
                        layer_idx=li,
                        expert_idx=ei,
                        module_name=f"model.layers.{li}.mlp.experts.{ei}",
                        family=plan.family,
                    )
                )
    ranked = [
        (c.layer, c.expert, float(c.affinity))
        for c in sorted(cells, key=lambda x: x.affinity, reverse=True)
    ]
    return SelectionPlan(
        domain=domain,
        selected=selected,
        frozen=frozen,
        ranked_scores=ranked,
        freeze_router=True,
        metadata={
            "source": "expert_group_studio",
            "n_groups_train": len(plan.enabled_train_groups()),
            "group_ids": [g.id for g in plan.enabled_train_groups()],
            "group_names": [g.name for g in plan.enabled_train_groups()],
            "n_cells": len(selected),
            "active_params_b": plan.capacity.active_params_b,
        },
    )


def selection_for_group(
    plan: GroupPlan,
    group_id: str,
    domain: str = "general",
) -> SelectionPlan:
    """
    ESFT SelectionPlan for a *single* sector — only that group's experts train;
    every other lattice cell is frozen. Used by sequential sector workflow.
    """
    g = plan.group_by_id(group_id)
    if g is None:
        return SelectionPlan(
            domain=domain,
            selected=[],
            frozen=[],
            ranked_scores=[],
            freeze_router=True,
            metadata={"source": "sector_workflow", "error": "group not found", "group_id": group_id},
        )
    cells = list(g.cells)
    selected = [
        ExpertRef(
            layer_idx=c.layer,
            expert_idx=c.expert,
            module_name=f"model.layers.{c.layer}.mlp.experts.{c.expert}",
            family=plan.family,
        )
        for c in cells
    ]
    selected_keys = {(c.layer, c.expert) for c in cells}
    frozen = []
    for li in range(plan.capacity.num_layers):
        for ei in range(plan.capacity.num_experts):
            if (li, ei) not in selected_keys:
                frozen.append(
                    ExpertRef(
                        layer_idx=li,
                        expert_idx=ei,
                        module_name=f"model.layers.{li}.mlp.experts.{ei}",
                        family=plan.family,
                    )
                )
    ranked = [
        (c.layer, c.expert, float(c.affinity))
        for c in sorted(cells, key=lambda x: x.affinity, reverse=True)
    ]
    return SelectionPlan(
        domain=g.domain or domain,
        selected=selected,
        frozen=frozen,
        ranked_scores=ranked,
        freeze_router=True,
        metadata={
            "source": "sector_workflow",
            "group_id": g.id,
            "group_name": g.name,
            "n_groups_train": 1,
            "group_ids": [g.id],
            "group_names": [g.name],
            "n_cells": len(selected),
            "active_params_b": plan.capacity.active_params_b,
            "domain_binding": g.domain,
            "topics": list(g.topics or [])[:12],
        },
    )


def lattice_view(plan: GroupPlan) -> dict[str, Any]:
    """
    Compact grid for the dashboard:
      membership[layer][expert] = group_id | null
      affinity[layer][expert] = score
      colors = {group_id: color}
    """
    layers = plan.capacity.num_layers
    experts = plan.capacity.num_experts
    membership = [[None for _ in range(experts)] for _ in range(layers)]
    affinity = [[0.0 for _ in range(experts)] for _ in range(layers)]
    colors = {g.id: g.color for g in plan.groups}
    names = {g.id: g.name for g in plan.groups}

    for g in plan.groups:
        for c in g.cells:
            if 0 <= c.layer < layers and 0 <= c.expert < experts:
                membership[c.layer][c.expert] = g.id
                affinity[c.layer][c.expert] = c.affinity

    # downsample if huge for wire size (keep step so UI can paint real cells)
    max_l, max_e = 48, 128
    step_l, step_e = 1, 1
    downsampled = False
    if layers > max_l or experts > max_e:
        step_l = max(1, (layers + max_l - 1) // max_l)
        step_e = max(1, (experts + max_e - 1) // max_e)
        membership = [row[::step_e][:max_e] for row in membership[::step_l][:max_l]]
        affinity = [row[::step_e][:max_e] for row in affinity[::step_l][:max_l]]
        downsampled = True

    return {
        "layers": layers,
        "experts": experts,
        "membership": membership,
        "affinity": affinity,
        "colors": colors,
        "names": names,
        "downsampled": downsampled,
        "step_l": step_l,
        "step_e": step_e,
        "display_rows": len(membership),
        "display_cols": len(membership[0]) if membership else 0,
        "capacity": plan.capacity.to_dict(),
        "summary": plan.summary(),
    }


def display_cell_to_real_cells(
    row: int,
    col: int,
    *,
    step_l: int = 1,
    step_e: int = 1,
    num_layers: int = 1,
    num_experts: int = 1,
) -> list[dict[str, int]]:
    """
    Map a clicked display pixel (possibly downsampled) to one or more real expert cells.
    When downsampled, paints the whole block covered by that pixel.
    """
    step_l = max(1, int(step_l))
    step_e = max(1, int(step_e))
    layer0 = int(row) * step_l
    exp0 = int(col) * step_e
    cells: list[dict[str, int]] = []
    for li in range(layer0, min(layer0 + step_l, num_layers)):
        for ei in range(exp0, min(exp0 + step_e, num_experts)):
            cells.append({"layer": li, "expert": ei})
    return cells


def analyze_group(
    plan: GroupPlan,
    group_id: str,
    *,
    affinity: Optional[dict[str, Any]] = None,
    with_forensics: bool = True,
) -> dict[str, Any]:
    g = plan.group_by_id(group_id)
    if not g:
        return {"error": "group not found"}
    estimate_group_capacity(g, plan.capacity)
    affs = [c.affinity for c in g.cells]
    layers = sorted({c.layer for c in g.cells})
    experts = sorted({c.expert for c in g.cells})
    out: dict[str, Any] = {
        "group": g.to_dict(),
        "analysis": {
            "n_cells": len(g.cells),
            "est_params_b": g.est_params_b,
            "active_fire_ratio": g.active_fire_ratio,
            "vs_model_active_b": plan.capacity.active_params_b,
            "affinity_mean": sum(affs) / len(affs) if affs else 0.0,
            "affinity_max": max(affs) if affs else 0.0,
            "affinity_min": min(affs) if affs else 0.0,
            "layer_span": [min(layers), max(layers)] if layers else None,
            "n_unique_layers": len(layers),
            "n_unique_expert_ids": len(experts),
            "top_cells": [
                {"layer": c.layer, "expert": c.expert, "affinity": c.affinity}
                for c in sorted(g.cells, key=lambda x: x.affinity, reverse=True)[:24]
            ],
            "data_binding": {
                "domain": g.domain,
                "curated_path": g.curated_path,
                "domain_pack": g.domain_pack,
                "topics": g.topics,
                "keywords": g.keywords,
                "data_weight": g.data_weight,
            },
            "train_flags": {
                "enabled": g.enabled,
                "train": g.train,
                "freeze": g.freeze,
            },
        },
    }
    if with_forensics:
        from aetherforge.groups.forensics import forensics_for_group

        out["forensics"] = forensics_for_group(
            plan, group_id, affinity=affinity
        )
    return out
