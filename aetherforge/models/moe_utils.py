"""
MoE architecture detection and expert targeting utilities.

Abstracts DeepSeek-V4-Flash (256 routed experts, shared experts, top-k)
and Qwen A3B-style sparse MoE behind a common ExpertRef interface so
AGPS / ESFT / lifecycle do not hardcode family-specific module paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from aetherforge.utils.logging import get_logger

log = get_logger("models.moe_utils")


@dataclass(frozen=True)
class ExpertRef:
    """Canonical pointer to one expert MLP inside a MoE layer."""

    layer_idx: int
    expert_idx: int
    module_name: str
    is_shared: bool = False
    family: str = "generic_moe"


@dataclass
class MoEArchitectureInfo:
    family: str
    num_layers: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    expert_module_pattern: str
    router_module_pattern: str
    expert_param_keywords: list[str] = field(
        default_factory=lambda: ["experts", "mlp.experts", "moe"]
    )
    router_param_keywords: list[str] = field(
        default_factory=lambda: ["gate", "router", "moe_gate"]
    )
    notes: str = ""
    raw_config: dict[str, Any] = field(default_factory=dict)

    @property
    def total_routed_experts(self) -> int:
        return self.num_experts * max(self.num_layers, 1)


# Known family fingerprints (config field heuristics)
_FAMILY_HINTS = {
    "deepseek_v4_flash": {
        "model_type_substrings": ["deepseek_v4", "deepseekv4", "deepseek"],
        "default_num_experts": 256,
        "default_topk": 6,
        "default_shared": 1,
        # V4 uses fused DeepseekV4Experts (no per-expert child modules).
        # Pattern matches parent bank; list_expert_modules synthesizes slices.
        "expert_pattern": r"(?:model\.)?layers\.(\d+)\.mlp\.experts(?:\.(\d+))?",
        "router_pattern": r"(?:model\.)?layers\.(\d+)\.mlp\.gate$",
        "notes": (
            "DeepSeek-V4-Flash fused experts (gate_up_proj/down_proj 3D); "
            "CSA/HCA + mHC; PEFT target_parameters required."
        ),
        "fused_experts": True,
    },
    "qwen_a3b": {
        "model_type_substrings": ["qwen3", "qwen2_moe", "qwen3_moe", "qwen"],
        "default_num_experts": 128,
        "default_topk": 8,
        "default_shared": 0,
        "expert_pattern": r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)",
        "router_pattern": r"model\.layers\.(\d+)\.mlp\.gate",
        "notes": "Qwen sparse MoE A3B-class (~30-35B total / ~3B active).",
        "fused_experts": False,
    },
}


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if hasattr(cfg, k):
            v = getattr(cfg, k)
            if v is not None:
                return v
        if isinstance(cfg, dict) and k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def infer_family_from_name(name: str) -> str:
    n = name.lower()
    if "deepseek" in n and ("v4" in n or "flash" in n):
        return "deepseek_v4_flash"
    if "a3b" in n or "qwen3" in n or "qwen2-moe" in n or "qwen2_moe" in n:
        return "qwen_a3b"
    if "deepseek" in n:
        return "deepseek_v4_flash"
    return "generic_moe"


def detect_moe_architecture(
    model_or_config: Any,
    family_hint: str = "auto",
    model_name: Optional[str] = None,
) -> MoEArchitectureInfo:
    """
    Inspect HF config / model to produce MoEArchitectureInfo.

    Works with unloaded configs (AutoConfig) or live nn.Module.
    """
    cfg = model_or_config
    if hasattr(model_or_config, "config"):
        cfg = model_or_config.config

    raw: dict[str, Any] = {}
    if hasattr(cfg, "to_dict"):
        try:
            raw = cfg.to_dict()
        except Exception:
            raw = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}
    elif isinstance(cfg, dict):
        raw = dict(cfg)

    model_type = str(_cfg_get(cfg, "model_type", default="") or "").lower()
    name = (model_name or _cfg_get(cfg, "_name_or_path", default="") or "").lower()
    architectures = _cfg_get(cfg, "architectures", default=[]) or []
    arch_str = " ".join(str(a) for a in architectures).lower()

    family = family_hint
    if family == "auto":
        if model_type == "deepseek_v4" or "deepseekv4" in arch_str:
            family = "deepseek_v4_flash"
        else:
            family = infer_family_from_name(name or model_type)
        if family == "generic_moe":
            for fam, hints in _FAMILY_HINTS.items():
                if any(
                    s in model_type or s in name for s in hints["model_type_substrings"]
                ):
                    family = fam
                    break

    # Prefer dedicated V4 arch builder when family is Flash
    if family == "deepseek_v4_flash":
        try:
            from aetherforge.models.deepseek_v4 import arch_from_flash_config

            info = arch_from_flash_config(cfg, model_name=model_name or name)
            log.info(
                "Detected MoE family=%s experts=%d topk=%d shared=%d layers=%d",
                info.family,
                info.num_experts,
                info.num_experts_per_tok,
                info.num_shared_experts,
                info.num_layers,
            )
            return info
        except Exception as e:
            log.warning("arch_from_flash_config failed (%s); using heuristics", e)

    hints = _FAMILY_HINTS.get(family, _FAMILY_HINTS.get("qwen_a3b", {}))

    num_experts = int(
        _cfg_get(
            cfg,
            "n_routed_experts",
            "num_experts",
            "num_local_experts",
            "moe_num_experts",
            default=hints.get("default_num_experts", 8),
        )
    )
    topk = int(
        _cfg_get(
            cfg,
            "num_experts_per_tok",
            "moe_top_k",
            "num_experts_per_token",
            default=hints.get("default_topk", 2),
        )
    )
    shared = int(
        _cfg_get(
            cfg,
            "n_shared_experts",
            "num_shared_experts",
            default=hints.get("default_shared", 0),
        )
    )
    n_layers = int(
        _cfg_get(cfg, "num_hidden_layers", "n_layer", "num_layers", default=0) or 0
    )

    info = MoEArchitectureInfo(
        family=family,
        num_layers=n_layers,
        num_experts=num_experts,
        num_experts_per_tok=topk,
        num_shared_experts=shared,
        expert_module_pattern=hints.get(
            "expert_pattern", r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)"
        ),
        router_module_pattern=hints.get(
            "router_pattern", r"model\.layers\.(\d+)\.mlp\.gate"
        ),
        notes=hints.get("notes", ""),
        raw_config={
            k: raw.get(k)
            for k in (
                "model_type",
                "architectures",
                "n_routed_experts",
                "num_experts",
                "num_experts_per_tok",
                "n_shared_experts",
                "hidden_size",
                "intermediate_size",
                "moe_intermediate_size",
            )
            if k in raw or True
        },
    )
    log.info(
        "Detected MoE family=%s experts=%d topk=%d shared=%d layers=%d",
        info.family,
        info.num_experts,
        info.num_experts_per_tok,
        info.num_shared_experts,
        info.num_layers,
    )
    return info


def list_expert_modules(model: Any, arch: Optional[MoEArchitectureInfo] = None) -> list[ExpertRef]:
    """Walk named_modules and return ExpertRef list for all routed experts found."""
    if arch is None:
        arch = detect_moe_architecture(model)

    # DeepSeek-V4 fused experts: no per-expert ModuleList children
    fused = (
        arch.family == "deepseek_v4_flash"
        or bool((arch.raw_config or {}).get("fused_experts"))
    )
    if fused:
        # Prefer live discovery of DeepseekV4Experts banks, then synthesize slices
        bank_layers: list[int] = []
        for name, mod in model.named_modules():
            cls = type(mod).__name__
            if cls == "DeepseekV4Experts" or (
                name.endswith("mlp.experts") and hasattr(mod, "gate_up_proj")
            ):
                lm = re.search(r"layers\.(\d+)", name)
                if lm:
                    bank_layers.append(int(lm.group(1)))
        if bank_layers or arch.num_layers > 0:
            from aetherforge.models.deepseek_v4 import synthetic_expert_refs

            # If we found banks, align layer count with live model when possible
            if bank_layers and len(bank_layers) != arch.num_layers:
                arch.num_layers = max(bank_layers) + 1
            refs = synthetic_expert_refs(arch)
            log.info(
                "Found %d expert refs (V4 fused banks=%d)",
                len(refs),
                len(bank_layers) or arch.num_layers,
            )
            return refs

    pat = re.compile(arch.expert_module_pattern)
    refs: list[ExpertRef] = []
    seen: set[str] = set()

    for name, _mod in model.named_modules():
        m = pat.search(name)
        if not m:
            continue
        groups = m.groups()
        layer_idx = int(groups[0])
        # Pattern may be (layer) only or (layer, expert)
        if len(groups) >= 2 and groups[1] is not None and str(groups[1]).isdigit():
            expert_idx = int(groups[1])
            key = f"{layer_idx}:{expert_idx}"
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                ExpertRef(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    module_name=name,
                    is_shared=False,
                    family=arch.family,
                )
            )
        # parent-only match without expert idx: skip here (handled by fused path)

    # Fallback: ModuleList named 'experts'
    if not refs:
        for name, mod in model.named_modules():
            if name.endswith("experts") and hasattr(mod, "__len__"):
                layer_m = re.search(r"layers\.(\d+)", name)
                layer_idx = int(layer_m.group(1)) if layer_m else -1
                try:
                    n = len(mod)
                except TypeError:
                    continue
                for i in range(n):
                    child_name = f"{name}.{i}"
                    refs.append(
                        ExpertRef(
                            layer_idx=layer_idx,
                            expert_idx=i,
                            module_name=child_name,
                            family=arch.family,
                        )
                    )

    # Last resort for V4-class names even if family mis-tagged
    if not refs:
        try:
            from aetherforge.models.deepseek_v4 import (
                is_deepseek_v4_bundle,
                synthetic_expert_refs,
            )

            if is_deepseek_v4_bundle(family=arch.family, model=model):
                refs = synthetic_expert_refs(arch)
        except Exception:
            pass

    log.info("Found %d expert module refs", len(refs))
    return refs


def list_router_module_names(model: Any, arch: Optional[MoEArchitectureInfo] = None) -> list[str]:
    if arch is None:
        arch = detect_moe_architecture(model)
    pat = re.compile(arch.router_module_pattern)
    names = [n for n, _ in model.named_modules() if pat.search(n)]
    if not names:
        keywords = arch.router_param_keywords
        for n, _ in model.named_modules():
            low = n.lower()
            if any(k in low for k in keywords) and ("mlp" in low or "moe" in low):
                names.append(n)
    return sorted(set(names))


def expert_global_id(ref: ExpertRef, num_experts: int) -> int:
    """Map (layer, local_expert) → global expert id for affinity matrices."""
    if ref.layer_idx < 0:
        return ref.expert_idx
    return ref.layer_idx * num_experts + ref.expert_idx


def select_param_names_for_experts(
    model: Any,
    experts: Iterable[ExpertRef],
    include_router: bool = False,
    arch: Optional[MoEArchitectureInfo] = None,
) -> list[str]:
    """Return parameter names belonging to selected experts (for freeze/LoRA targeting)."""
    prefixes = {e.module_name for e in experts}
    selected: list[str] = []
    for pname, _ in model.named_parameters():
        if any(pname == p or pname.startswith(p + ".") for p in prefixes):
            selected.append(pname)
    if include_router:
        for rname in list_router_module_names(model, arch):
            for pname, _ in model.named_parameters():
                if pname == rname or pname.startswith(rname + "."):
                    selected.append(pname)
    return sorted(set(selected))


def freeze_all_parameters(model: Any) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_parameters_by_name(model: Any, names: Iterable[str]) -> int:
    name_set = set(names)
    n = 0
    for pname, p in model.named_parameters():
        if pname in name_set:
            p.requires_grad = True
            n += 1
    return n


def count_trainable_parameters(model: Any) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
