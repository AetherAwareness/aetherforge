"""
Auto-partition the expert lattice into user-facing groups.

Strategies:
  - affinity: cluster experts by domain affinity scores (needs AffinityResult)
  - round_robin: stripe experts across N groups (baseline)
  - layer_bands: contiguous layer blocks
  - active_slots: carve N groups each targeting ~one active-fire mass
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from aetherforge.groups.capacity import (
    build_full_lattice,
    estimate_group_capacity,
    recompute_all_group_capacities,
)
from aetherforge.groups.models import ExpertCell, ExpertGroup, GroupPlan, ModelCapacity
from aetherforge.utils.logging import get_logger

log = get_logger("groups.cluster")

_COLORS = [
    "#5b9dff",
    "#3dd68c",
    "#f0b429",
    "#a78bfa",
    "#ff5c6c",
    "#22d3ee",
    "#fb923c",
    "#e879f9",
    "#84cc16",
    "#38bdf8",
    "#f472b6",
    "#a3e635",
]


def auto_partition_groups(
    capacity: ModelCapacity,
    *,
    num_groups: Optional[int] = None,
    strategy: Literal["affinity", "round_robin", "layer_bands", "active_slots"] = "active_slots",
    affinity_matrix: Optional[list[list[float]]] = None,
    ranked: Optional[list[tuple[int, int, float]]] = None,
    model_name: str = "",
    target_active_fire_ratio: float = 1.0,
) -> GroupPlan:
    """
    Create a GroupPlan with N editable sectors.

    num_groups default = min(12, max_disjoint_active_groups) so Flash-class
    models surface multiple ~13B-scale sectors the user can assign data to.
    """
    # How many groupings can be "called up"
    max_disjoint = max(1, capacity.max_disjoint_active_groups)
    if num_groups is None:
        # Prefer a human-manageable count, clamped by capacity
        num_groups = max(2, min(12, max_disjoint if max_disjoint > 1 else 8))
    num_groups = max(1, min(int(num_groups), 64))

    lattice = build_full_lattice(capacity)
    # Attach affinity if provided
    aff_map: dict[tuple[int, int], float] = {}
    if affinity_matrix:
        for li, row in enumerate(affinity_matrix):
            if not isinstance(row, list):
                continue
            for ei, v in enumerate(row):
                aff_map[(li, ei)] = float(v)
    if ranked:
        for i, (li, ei, score) in enumerate(ranked):
            aff_map[(li, ei)] = float(score)
            # rank later

    for c in lattice:
        c.affinity = aff_map.get(c.as_tuple(), 0.0)

    if strategy == "affinity" and aff_map:
        buckets = _partition_by_affinity(lattice, num_groups)
    elif strategy == "layer_bands":
        buckets = _partition_layer_bands(lattice, capacity, num_groups)
    elif strategy == "round_robin":
        buckets = _partition_round_robin(lattice, num_groups)
    else:
        # active_slots: pack cells into groups near target fire mass
        buckets = _partition_active_slots(
            lattice, capacity, num_groups, target_active_fire_ratio
        )

    groups: list[ExpertGroup] = []
    for i, cells in enumerate(buckets):
        if not cells:
            continue
        g = ExpertGroup(
            name=f"sector_{i + 1}",
            color=_COLORS[i % len(_COLORS)],
            description=f"Auto-partitioned sector {i + 1}/{num_groups} ({strategy})",
            cells=cells,
            enabled=True,
            train=True,
            auto=True,
            tags=["auto", strategy],
        )
        estimate_group_capacity(g, capacity)
        groups.append(g)

    plan = GroupPlan(
        model_name=model_name or capacity.model_name,
        family=capacity.family,
        capacity=capacity,
        groups=groups,
        target_num_groups=num_groups,
        target_active_fire_ratio=target_active_fire_ratio,
        unassigned=[],
        notes=(
            f"Auto-partition strategy={strategy}. "
            f"Model active≈{capacity.active_params_b}B; "
            f"max disjoint active-scale groups≈{capacity.max_disjoint_active_groups}."
        ),
    )
    recompute_all_group_capacities(plan.groups, capacity)
    log.info(
        "Partitioned %d slots → %d groups (strategy=%s)",
        capacity.total_expert_slots,
        len(groups),
        strategy,
    )
    return plan


def _partition_round_robin(cells: list[ExpertCell], n: int) -> list[list[ExpertCell]]:
    buckets: list[list[ExpertCell]] = [[] for _ in range(n)]
    for i, c in enumerate(cells):
        buckets[i % n].append(c)
    return buckets


def _partition_layer_bands(
    cells: list[ExpertCell], capacity: ModelCapacity, n: int
) -> list[list[ExpertCell]]:
    buckets: list[list[ExpertCell]] = [[] for _ in range(n)]
    if capacity.num_layers <= 0:
        return _partition_round_robin(cells, n)
    band = max(1, (capacity.num_layers + n - 1) // n)
    for c in cells:
        bi = min(n - 1, c.layer // band)
        buckets[bi].append(c)
    return buckets


def _partition_by_affinity(cells: list[ExpertCell], n: int) -> list[list[ExpertCell]]:
    ordered = sorted(cells, key=lambda c: c.affinity, reverse=True)
    return _partition_round_robin(ordered, n)


def _partition_active_slots(
    cells: list[ExpertCell],
    capacity: ModelCapacity,
    n: int,
    target_ratio: float,
) -> list[list[ExpertCell]]:
    """
    Greedy pack: sort by affinity, fill groups until each approaches
    target_ratio * one_active_fire params.
    """
    per = capacity.params_per_expert_b or 1e-6
    fire = capacity.active_expert_params_b or capacity.active_params_b or 1.0
    target_params = max(per, fire * max(0.25, target_ratio))
    target_cells = max(1, int(round(target_params / per)))

    ordered = sorted(cells, key=lambda c: (c.affinity, -c.layer, c.expert), reverse=True)
    buckets: list[list[ExpertCell]] = [[] for _ in range(n)]
    # Primary fill target_cells each, remainder round-robin into smallest
    idx = 0
    for c in ordered:
        # find group under target
        placed = False
        for _ in range(n):
            b = buckets[idx % n]
            if len(b) < target_cells:
                b.append(c)
                placed = True
                idx += 1
                break
            idx += 1
        if not placed:
            # all full — put in smallest
            smallest = min(range(n), key=lambda i: len(buckets[i]))
            buckets[smallest].append(c)
    return buckets


def merge_affinity_into_plan(
    plan: GroupPlan,
    affinity_matrix: Optional[list[list[float]]] = None,
    ranked: Optional[list] = None,
) -> GroupPlan:
    """Update cell affinity scores inside existing groups from a probe result."""
    aff_map: dict[tuple[int, int], float] = {}
    if affinity_matrix:
        for li, row in enumerate(affinity_matrix):
            for ei, v in enumerate(row):
                aff_map[(li, ei)] = float(v)
    if ranked:
        for item in ranked:
            if len(item) >= 3:
                aff_map[(int(item[0]), int(item[1]))] = float(item[2])

    for g in plan.groups:
        for c in g.cells:
            c.affinity = aff_map.get(c.as_tuple(), c.affinity)
        if g.cells:
            # sort cells in group by affinity for UI
            g.cells.sort(key=lambda x: x.affinity, reverse=True)
    return plan
