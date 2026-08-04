"""
Expert Group data model.

Sparse MoE models (e.g. DeepSeek-V4-Flash ~284B total / ~13B active) route each
token through a small *active set* of experts (top-k per layer + shared).

AetherForge lets users carve the full expert lattice into named **groups**
(sectors). Each group is a train-able unit you can:
  - visualize
  - assign domain data to
  - enable/disable for a run
  - edit membership, color, notes, freeze flags

A "capacity group" is sized relative to the model's typical *active* budget
(~13B for Flash, ~3B for A3B) so users see how heavy each sector is vs one
routing fire.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ExpertCell(BaseModel):
    """One expert slot: layer × expert index."""

    layer: int
    expert: int
    is_shared: bool = False
    # Optional live metrics attached for studio display
    affinity: float = 0.0
    util_rel: float = 0.0
    rank: Optional[int] = None

    @property
    def key(self) -> str:
        return f"L{self.layer}/E{self.expert}"

    def as_tuple(self) -> tuple[int, int]:
        return (self.layer, self.expert)


class ModelCapacity(BaseModel):
    """
    How big is this MoE and what does one 'active fire' cost?

    Numbers are estimates unless measured from a loaded model.
    """

    family: str = "generic_moe"
    model_name: str = ""
    total_params_b: float = 0.0  # e.g. 284
    active_params_b: float = 0.0  # e.g. 13
    num_layers: int = 0
    num_experts: int = 0
    top_k: int = 0
    num_shared_experts: int = 0
    # Estimated params for one expert MLP (billions)
    params_per_expert_b: float = 0.0
    # Expert params that fire in one forward (top_k * layers * per_expert + shared)
    active_expert_params_b: float = 0.0
    # How many non-overlapping expert-sets of ~active size fit in the expert pool
    max_disjoint_active_groups: int = 1
    # Total routed expert slots (layers * experts)
    total_expert_slots: int = 0
    source: Literal["profile", "estimate", "config"] = "estimate"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ExpertGroup(BaseModel):
    """A named sector of experts + optional data assignment."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    name: str = "sector"
    color: str = "#5b9dff"
    description: str = ""
    # Membership
    cells: list[ExpertCell] = Field(default_factory=list)
    # Training assignment
    enabled: bool = True
    train: bool = True
    freeze: bool = False
    # Data binding (industry pack / corpus for THIS group only)
    domain: Optional[str] = None
    domain_pack: Optional[str] = None
    curated_path: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    data_weight: float = 1.0
    # Capacity snapshot (filled by estimator)
    n_cells: int = 0
    est_params_b: float = 0.0
    # vs one active fire: 1.0 ≈ same mass as top-k routing activation
    active_fire_ratio: float = 0.0
    # Metadata
    auto: bool = False  # created by auto-partition
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    updated_at: float = Field(default_factory=time.time)

    def refresh_counts(self) -> None:
        self.n_cells = len(self.cells)

    def cell_keys(self) -> set[tuple[int, int]]:
        return {c.as_tuple() for c in self.cells}

    def add_cell(self, layer: int, expert: int, **kw: Any) -> None:
        key = (layer, expert)
        if key in self.cell_keys():
            return
        self.cells.append(ExpertCell(layer=layer, expert=expert, **kw))
        self.refresh_counts()
        self.updated_at = time.time()

    def remove_cell(self, layer: int, expert: int) -> bool:
        before = len(self.cells)
        self.cells = [c for c in self.cells if not (c.layer == layer and c.expert == expert)]
        self.refresh_counts()
        self.updated_at = time.time()
        return len(self.cells) < before

    def to_dict(self) -> dict[str, Any]:
        self.refresh_counts()
        return self.model_dump()


class GroupPlan(BaseModel):
    """Full studio state for a model: capacity + all groups."""

    schema_version: str = "aetherforge.groups.v1"
    model_name: str = ""
    family: str = "generic_moe"
    capacity: ModelCapacity = Field(default_factory=ModelCapacity)
    groups: list[ExpertGroup] = Field(default_factory=list)
    # User preference: how many sectors to auto-carve
    target_num_groups: int = 8
    # Prefer groups sized near this fraction of one active fire (1.0 = ~13B on Flash)
    target_active_fire_ratio: float = 1.0
    unassigned: list[ExpertCell] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    notes: str = ""

    def group_by_id(self, gid: str) -> Optional[ExpertGroup]:
        for g in self.groups:
            if g.id == gid:
                return g
        return None

    def all_assigned_keys(self) -> set[tuple[int, int]]:
        keys: set[tuple[int, int]] = set()
        for g in self.groups:
            keys |= g.cell_keys()
        return keys

    def enabled_train_groups(self) -> list[ExpertGroup]:
        return [g for g in self.groups if g.enabled and g.train and not g.freeze]

    def selected_cells_for_training(self) -> list[ExpertCell]:
        cells: list[ExpertCell] = []
        seen: set[tuple[int, int]] = set()
        for g in self.enabled_train_groups():
            for c in g.cells:
                k = c.as_tuple()
                if k not in seen:
                    seen.add(k)
                    cells.append(c)
        return cells

    def summary(self) -> dict[str, Any]:
        train_g = self.enabled_train_groups()
        return {
            "model_name": self.model_name,
            "family": self.family,
            "n_groups": len(self.groups),
            "n_train_groups": len(train_g),
            "n_enabled": sum(1 for g in self.groups if g.enabled),
            "n_cells_total": self.capacity.total_expert_slots,
            "n_cells_assigned": len(self.all_assigned_keys()),
            "n_cells_train": sum(len(g.cells) for g in train_g),
            "active_params_b": self.capacity.active_params_b,
            "total_params_b": self.capacity.total_params_b,
            "max_disjoint_active_groups": self.capacity.max_disjoint_active_groups,
            "target_num_groups": self.target_num_groups,
            "groups": [
                {
                    "id": g.id,
                    "name": g.name,
                    "color": g.color,
                    "n_cells": len(g.cells),
                    "est_params_b": g.est_params_b,
                    "active_fire_ratio": g.active_fire_ratio,
                    "enabled": g.enabled,
                    "train": g.train,
                    "freeze": g.freeze,
                    "domain": g.domain,
                    "curated_path": g.curated_path,
                }
                for g in self.groups
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        self.updated_at = time.time()
        return self.model_dump()
