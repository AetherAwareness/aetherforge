"""
Capacity estimation for MoE expert pools and user-defined groups.

Goal: answer "how many ~active-sized sectors can I carve?" and
"how heavy is this group vs one routing fire (~13B on Flash)?"
"""

from __future__ import annotations

from typing import Any, Optional

from aetherforge.groups.models import ExpertCell, ExpertGroup, ModelCapacity
from aetherforge.utils.logging import get_logger

log = get_logger("groups.capacity")

# Published / conventional profiles (overridable by config or measurement)
_FAMILY_PROFILES: dict[str, dict[str, float | int | str]] = {
    "deepseek_v4_flash": {
        "total_params_b": 284.0,
        "active_params_b": 13.0,
        "num_experts": 256,
        "top_k": 6,
        "num_shared_experts": 1,
        # DeepSeek-V4-Flash-0731: 43 layers (DeepseekV4Config.num_hidden_layers)
        "num_layers": 43,
        "notes": (
            "DeepSeek-V4-Flash-0731: ~284B total / ~13B active, 43 layers, "
            "256 routed, top-k=6, fused gate_up_proj/down_proj experts."
        ),
    },
    "qwen_a3b": {
        "total_params_b": 35.0,
        "active_params_b": 3.0,
        "num_experts": 128,
        "top_k": 8,
        "num_shared_experts": 0,
        "num_layers": 48,
        "notes": "Qwen A3B-class: ~30-35B total / ~3B active.",
    },
    "generic_moe": {
        "total_params_b": 30.0,
        "active_params_b": 3.0,
        "num_experts": 64,
        "top_k": 4,
        "num_shared_experts": 0,
        "num_layers": 32,
        "notes": "Generic sparse MoE defaults — override in config.",
    },
}


def estimate_capacity(
    family: str = "auto",
    *,
    model_name: str = "",
    num_layers: Optional[int] = None,
    num_experts: Optional[int] = None,
    top_k: Optional[int] = None,
    num_shared_experts: Optional[int] = None,
    total_params_b: Optional[float] = None,
    active_params_b: Optional[float] = None,
    # Expert pool fraction of total params (rest = attn/embed/etc.)
    expert_pool_fraction: float = 0.75,
) -> ModelCapacity:
    fam = family if family in _FAMILY_PROFILES else (
        "deepseek_v4_flash" if "deepseek" in model_name.lower() or "flash" in model_name.lower()
        else "qwen_a3b" if "a3b" in model_name.lower() or "qwen" in model_name.lower()
        else "generic_moe"
    )
    if family not in ("auto",) and family in _FAMILY_PROFILES:
        fam = family

    prof = dict(_FAMILY_PROFILES.get(fam, _FAMILY_PROFILES["generic_moe"]))
    layers = int(num_layers if num_layers is not None else prof["num_layers"])
    n_exp = int(num_experts if num_experts is not None else prof["num_experts"])
    k = int(top_k if top_k is not None else prof["top_k"])
    shared = int(
        num_shared_experts if num_shared_experts is not None else prof["num_shared_experts"]
    )
    total_b = float(total_params_b if total_params_b is not None else prof["total_params_b"])
    active_b = float(active_params_b if active_params_b is not None else prof["active_params_b"])

    slots = max(layers * n_exp, 1)
    # Expert pool mass
    expert_pool_b = total_b * expert_pool_fraction
    params_per_expert_b = expert_pool_b / slots if slots else 0.0
    # One routing fire: top_k experts per MoE layer (approx) + shared
    # Conservative: top_k * num_layers * per_expert  (if every layer is MoE)
    active_expert_b = params_per_expert_b * (k * layers + shared * layers)
    # How many disjoint ~active-sized expert sets fit in the pool
    if active_expert_b > 0:
        max_groups = max(1, int(expert_pool_b / active_expert_b))
    else:
        max_groups = 1

    cap = ModelCapacity(
        family=fam,
        model_name=model_name,
        total_params_b=round(total_b, 3),
        active_params_b=round(active_b, 3),
        num_layers=layers,
        num_experts=n_exp,
        top_k=k,
        num_shared_experts=shared,
        params_per_expert_b=round(params_per_expert_b, 6),
        active_expert_params_b=round(active_expert_b, 4),
        max_disjoint_active_groups=max_groups,
        total_expert_slots=slots,
        source="estimate",
        notes=str(prof.get("notes", "")),
    )
    log.info(
        "Capacity %s: total=%.1fB active=%.1fB slots=%d per_expert=%.4fB "
        "one_fire≈%.2fB max_disjoint_groups=%d",
        fam,
        cap.total_params_b,
        cap.active_params_b,
        cap.total_expert_slots,
        cap.params_per_expert_b,
        cap.active_expert_params_b,
        cap.max_disjoint_active_groups,
    )
    return cap


def estimate_group_capacity(group: ExpertGroup, capacity: ModelCapacity) -> ExpertGroup:
    """Fill est_params_b and active_fire_ratio on a group."""
    n = len(group.cells)
    group.n_cells = n
    per = capacity.params_per_expert_b or 0.0
    group.est_params_b = round(n * per, 4)
    fire = capacity.active_expert_params_b or capacity.active_params_b or 1.0
    group.active_fire_ratio = round(group.est_params_b / fire, 3) if fire else 0.0
    return group


def recompute_all_group_capacities(groups: list[ExpertGroup], capacity: ModelCapacity) -> None:
    for g in groups:
        estimate_group_capacity(g, capacity)


def build_full_lattice(capacity: ModelCapacity) -> list[ExpertCell]:
    """All expert cells for the model lattice."""
    cells: list[ExpertCell] = []
    for li in range(capacity.num_layers):
        for ei in range(capacity.num_experts):
            cells.append(ExpertCell(layer=li, expert=ei))
    return cells


def capacity_from_arch_info(arch: Any, model_name: str = "") -> ModelCapacity:
    """Build capacity from MoEArchitectureInfo-like object."""
    family = getattr(arch, "family", "generic_moe") or "generic_moe"
    return estimate_capacity(
        family=family,
        model_name=model_name,
        num_layers=getattr(arch, "num_layers", None) or None,
        num_experts=getattr(arch, "num_experts", None) or None,
        top_k=getattr(arch, "num_experts_per_tok", None) or None,
        num_shared_experts=getattr(arch, "num_shared_experts", None),
    )
