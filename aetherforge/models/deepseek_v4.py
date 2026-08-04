"""
DeepSeek-V4-Flash / Flash-0731 training support.

Verified against:
  - HF model: deepseek-ai/DeepSeek-V4-Flash-0731
  - transformers.models.deepseek_v4 (DeepseekV4ForCausalLM)
  - Weight map: layers.N.ffn.experts.E.w{1,2,3} on disk
  - Runtime modules: model.layers.N.mlp.{gate,experts,shared_experts}
    where experts is DeepseekV4Experts with fused 3D params:
      gate_up_proj [n_experts, 2*inter, hidden]  e.g. (256, 4096, 4096)
      down_proj    [n_experts, hidden, inter]    e.g. (256, 4096, 2048)
  - is_transposed=False on DeepseekV4Experts (PEFT ParamWrapper handles this)

This is NOT the classic Qwen/DeepSeek-V2 ModuleList of MLP experts.
Standard PEFT target_modules=['gate_proj','up_proj','down_proj'] only hits
shared_experts MLPs, NOT routed experts. We use target_parameters for the
fused tensors + optional expert-index gradient masks for ESFT.

Critical PEFT constraint: ParamWrapper forbids lora_dropout != 0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from aetherforge.models.moe_utils import ExpertRef, MoEArchitectureInfo
from aetherforge.utils.logging import get_logger

log = get_logger("models.deepseek_v4")

FLASH_0731_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
FLASH_IDS = {
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V4-Flash-Base",
    "unsloth/DeepSeek-V4-Flash-0731",
}

# Canonical architecture (from HF config.json + transformers DeepseekV4Config)
FLASH_CANONICAL = {
    "model_type": "deepseek_v4",
    "architectures": ["DeepseekV4ForCausalLM"],
    "num_hidden_layers": 43,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "n_shared_experts": 1,
    "hidden_size": 4096,
    "moe_intermediate_size": 2048,
    "total_params_b": 284.0,
    "active_params_b": 13.0,
}

# PEFT ParamWrapper requires dropout == 0 for target_parameters
V4_LORA_DROPOUT = 0.0


@dataclass
class FlashValidateReport:
    ok: bool
    model_id: str
    checks: dict[str, Any]
    blockers: list[str]
    warnings: list[str]
    train_plan: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model_id": self.model_id,
            "checks": self.checks,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "train_plan": self.train_plan,
        }


def is_deepseek_v4_name(name: str) -> bool:
    n = (name or "").lower()
    return "deepseek" in n and ("v4" in n or "flash" in n)


def is_deepseek_v4_bundle(family: str = "", model_name: str = "", model: Any = None) -> bool:
    if (family or "").lower() in ("deepseek_v4_flash", "deepseek_v4"):
        return True
    if is_deepseek_v4_name(model_name):
        return True
    if model is not None:
        cfg = getattr(model, "config", None)
        mt = str(getattr(cfg, "model_type", "") or "").lower()
        if mt == "deepseek_v4":
            return True
        archs = getattr(cfg, "architectures", None) or []
        if any("DeepseekV4" in str(a) for a in archs):
            return True
    return False


def arch_from_flash_config(cfg: Any, model_name: str = "") -> MoEArchitectureInfo:
    """Build MoEArchitectureInfo from DeepseekV4Config / dict."""

    def g(*keys, default=None):
        for k in keys:
            if hasattr(cfg, k) and getattr(cfg, k) is not None:
                return getattr(cfg, k)
            if isinstance(cfg, dict) and cfg.get(k) is not None:
                return cfg[k]
        return default

    n_layers = int(g("num_hidden_layers", default=FLASH_CANONICAL["num_hidden_layers"]))
    n_exp = int(g("n_routed_experts", "num_local_experts", "num_experts", default=256))
    topk = int(g("num_experts_per_tok", default=6))
    shared = int(g("n_shared_experts", default=1))

    return MoEArchitectureInfo(
        family="deepseek_v4_flash",
        num_layers=n_layers,
        num_experts=n_exp,
        num_experts_per_tok=topk,
        num_shared_experts=shared,
        # Fused bank — expert index is dim-0 of gate_up_proj/down_proj, not a child module
        expert_module_pattern=r"(?:model\.)?layers\.(\d+)\.mlp\.experts(?:\.(\d+))?",
        router_module_pattern=r"(?:model\.)?layers\.(\d+)\.mlp\.gate$",
        expert_param_keywords=["experts", "gate_up_proj", "down_proj"],
        router_param_keywords=["gate", "mlp.gate"],
        notes=(
            f"DeepSeek-V4-Flash fused experts; layers={n_layers} routed={n_exp} "
            f"topk={topk} shared={shared}. PEFT via target_parameters on "
            f"gate_up_proj/down_proj (lora_dropout must be 0)."
        ),
        raw_config={
            "model_type": g("model_type", default="deepseek_v4"),
            "hidden_size": g("hidden_size"),
            "moe_intermediate_size": g("moe_intermediate_size", "intermediate_size"),
            "model_name": model_name,
            "fused_experts": True,
        },
    )


def synthetic_expert_refs(arch: MoEArchitectureInfo) -> list[ExpertRef]:
    """
    V4 experts are slices of fused 3D params, not nn.Module children.
    Synthesize ExpertRef per (layer, expert_idx) pointing at the parent module.
    """
    refs: list[ExpertRef] = []
    for li in range(arch.num_layers):
        parent = f"model.layers.{li}.mlp.experts"
        for ei in range(arch.num_experts):
            refs.append(
                ExpertRef(
                    layer_idx=li,
                    expert_idx=ei,
                    module_name=f"{parent}#{ei}",  # virtual slice id
                    is_shared=False,
                    family="deepseek_v4_flash",
                )
            )
    return refs


def list_v4_runtime_modules(model: Any) -> dict[str, Any]:
    """Discover actual module paths on a loaded DeepseekV4 model."""
    expert_mods = []
    gate_mods = []
    shared_mods = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls == "DeepseekV4Experts" or (
            name.endswith("mlp.experts") and hasattr(mod, "gate_up_proj")
        ):
            expert_mods.append(name)
        if cls in ("DeepseekV4TopKRouter", "DeepseekV4HashRouter") or (
            name.endswith("mlp.gate") and "shared" not in name
        ):
            gate_mods.append(name)
        if "shared_experts" in name and name.endswith("shared_experts"):
            shared_mods.append(name)
    return {
        "expert_modules": expert_mods,
        "gate_modules": gate_mods,
        "shared_modules": shared_mods,
        "n_expert_modules": len(expert_mods),
        "n_gate_modules": len(gate_mods),
        "n_shared_modules": len(shared_mods),
    }


def apply_v4_expert_lora(
    model: Any,
    *,
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = V4_LORA_DROPOUT,
    include_shared: bool = True,
    include_router: bool = False,
    extra_modules: Optional[list[str]] = None,
) -> Any:
    """
    Attach PEFT LoRA for DeepSeek-V4 fused experts.

    Uses target_parameters for gate_up_proj / down_proj (routed experts)
    and target_modules for shared expert Linears if present.

    Forces lora_dropout=0 — PEFT ParamWrapper rejects non-zero dropout.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    if lora_dropout and lora_dropout > 0:
        log.warning(
            "DeepSeek-V4 ParamWrapper requires lora_dropout=0; "
            "got %.3f — forcing 0.0",
            lora_dropout,
        )
        lora_dropout = V4_LORA_DROPOUT

    target_parameters = ["gate_up_proj", "down_proj"]
    # Empty list = modules only via target_parameters for routed; shared via names
    target_modules: list[str] = []
    if include_shared:
        # DeepseekV4MLP on shared_experts — standard Linear LoRA
        target_modules.extend(["gate_proj", "up_proj", "down_proj"])
    if extra_modules:
        for m in extra_modules:
            if m not in target_modules:
                target_modules.append(m)
    if include_router:
        pass  # router is often nn.Parameter; leave frozen for hygiene stage

    kwargs: dict[str, Any] = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
        "target_parameters": target_parameters,
    }
    if target_modules:
        kwargs["target_modules"] = target_modules
    else:
        kwargs["target_modules"] = []

    try:
        lora_cfg = LoraConfig(**kwargs)
    except TypeError as e:
        log.warning(
            "PEFT LoraConfig rejected target_parameters (%s); falling back to "
            "shared MLP modules only. Upgrade peft>=0.15 for fused-expert LoRA.",
            e,
        )
        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["gate_proj", "up_proj", "down_proj"],
        )

    model = get_peft_model(model, lora_cfg)
    log.info(
        "Applied DeepSeek-V4 LoRA r=%d target_parameters=%s target_modules=%s dropout=%s",
        r,
        target_parameters,
        target_modules,
        lora_dropout,
    )
    return model


def _layer_from_param_name(name: str) -> Optional[int]:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def install_expert_grad_mask(
    model: Any,
    selected: Iterable[tuple[int, int]],
    arch: MoEArchitectureInfo,
) -> list[Any]:
    """
    Zero gradients on non-selected expert slices.

    Covers:
      1) Base fused params ...experts.gate_up_proj / .down_proj (full_esft)
      2) PEFT ParamWrapper multi-expert LoRA A/B weights under ...mlp.experts...
         (esft_lora) — PEFT packs experts as r*E rows/cols

    selected: iterable of (layer_idx, expert_idx)
    Returns list of hook handles (caller may remove later).
    """
    selected_set = set((int(a), int(b)) for a, b in selected)
    by_layer: dict[int, set[int]] = {}
    for li, ei in selected_set:
        by_layer.setdefault(li, set()).add(ei)

    n_exp_arch = max(int(arch.num_experts), 1)
    handles = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        li = _layer_from_param_name(name)
        if li is None:
            continue
        if ".mlp.experts" not in name and "mlp.experts" not in name:
            continue

        allowed = by_layer.get(li, set())

        # --- base fused 3D tensors (dim0 = expert) ---
        if re.search(r"experts\.(?:base_layer\.)*(?:gate_up_proj|down_proj)$", name) or (
            re.search(r"(gate_up_proj|down_proj)$", name)
            and param.dim() == 3
            and param.shape[0] == n_exp_arch
        ):
            n_exp = param.shape[0]

            def _make_base_hook(allowed_idx: set[int], n: int):
                def hook(grad):
                    if grad is None:
                        return None
                    mask = grad.new_zeros((n,) + (1,) * (grad.dim() - 1))
                    if allowed_idx:
                        mask[list(allowed_idx)] = 1
                    return grad * mask

                return hook

            handles.append(param.register_hook(_make_base_hook(allowed, n_exp)))
            log.debug("Base grad mask on %s allow=%d/%d", name, len(allowed), n_exp)
            continue

        # --- PEFT multi-expert LoRA A: (r*E, in_features) ---
        if "lora_A" in name and "experts" in name and param.dim() == 2:
            if param.shape[0] % n_exp_arch == 0:
                n_exp = n_exp_arch
                r = param.shape[0] // n_exp

                def _make_a_hook(allowed_idx: set[int], n: int, rank: int):
                    def hook(grad):
                        if grad is None:
                            return None
                        g = grad.view(n, rank, -1)
                        mask = g.new_zeros((n, 1, 1))
                        if allowed_idx:
                            mask[list(allowed_idx)] = 1
                        return (g * mask).view_as(grad)

                    return hook

                handles.append(param.register_hook(_make_a_hook(allowed, n_exp, r)))
                log.debug("LoRA-A grad mask on %s allow=%d/%d r=%d", name, len(allowed), n_exp, r)
                continue

        # --- PEFT multi-expert LoRA B: (out_features, r*E) ---
        if "lora_B" in name and "experts" in name and param.dim() == 2:
            if param.shape[1] % n_exp_arch == 0:
                n_exp = n_exp_arch
                r = param.shape[1] // n_exp

                def _make_b_hook(allowed_idx: set[int], n: int, rank: int):
                    def hook(grad):
                        if grad is None:
                            return None
                        # (out, r*E) -> (out, r, E)
                        g = grad.view(grad.shape[0], rank, n)
                        mask = g.new_zeros((1, 1, n))
                        if allowed_idx:
                            mask[..., list(allowed_idx)] = 1
                        return (g * mask).view_as(grad)

                    return hook

                handles.append(param.register_hook(_make_b_hook(allowed, n_exp, r)))
                log.debug("LoRA-B grad mask on %s allow=%d/%d r=%d", name, len(allowed), n_exp, r)
                continue

    log.info(
        "Installed expert grad masks for %d (layer,expert) pairs on %d parameters",
        len(selected_set),
        len(handles),
    )
    return handles


def freeze_nonselected_base_experts(
    model: Any,
    selected: Iterable[tuple[int, int]],
    arch: MoEArchitectureInfo,
) -> list[Any]:
    """
    For full_esft: enable grads on fused expert banks, then install slice masks.
    Shared / attention left frozen unless already trainable.
    """
    selected_set = set((int(a), int(b)) for a, b in selected)
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze fused expert params only
    n = 0
    for name, param in model.named_parameters():
        if re.search(r"layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$", name):
            param.requires_grad = True
            n += 1
    log.info("full_esft V4: unfroze %d fused expert parameters", n)
    return install_expert_grad_mask(model, selected_set, arch)


def validate_flash_training_stack(
    model_id: str = FLASH_0731_ID,
    *,
    check_weights_index: bool = True,
    try_meta_init: bool = True,
    try_peft_attach: bool = True,
) -> FlashValidateReport:
    """
    Hard validation that AetherForge can train this checkpoint — config, transformers,
    weight layout, PEFT path, VRAM notes. Does NOT download full weights.
    """
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    warnings: list[str] = []
    model_id = model_id or FLASH_0731_ID
    cfg = None
    arch = arch_from_flash_config(FLASH_CANONICAL, model_name=model_id)

    # 1) transformers support
    try:
        import transformers
        from transformers import AutoConfig

        checks["transformers_version"] = transformers.__version__
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        checks["config_class"] = type(cfg).__name__
        checks["model_type"] = getattr(cfg, "model_type", None)
        checks["num_hidden_layers"] = getattr(cfg, "num_hidden_layers", None)
        checks["n_routed_experts"] = getattr(cfg, "n_routed_experts", None) or getattr(
            cfg, "num_local_experts", None
        )
        checks["num_experts_per_tok"] = getattr(cfg, "num_experts_per_tok", None)
        checks["n_shared_experts"] = getattr(cfg, "n_shared_experts", None)
        checks["hidden_size"] = getattr(cfg, "hidden_size", None)
        checks["moe_intermediate_size"] = getattr(cfg, "moe_intermediate_size", None)
        if checks["model_type"] != "deepseek_v4":
            blockers.append(f"Unexpected model_type={checks['model_type']}")
        if int(checks["n_routed_experts"] or 0) != 256:
            warnings.append(
                f"n_routed_experts={checks['n_routed_experts']} (expected 256 for Flash)"
            )
        if int(checks["num_hidden_layers"] or 0) != 43:
            warnings.append(
                f"num_hidden_layers={checks['num_hidden_layers']} (Flash card uses 43)"
            )
        arch = arch_from_flash_config(cfg, model_name=model_id)
        checks["arch"] = {
            "family": arch.family,
            "layers": arch.num_layers,
            "experts": arch.num_experts,
            "topk": arch.num_experts_per_tok,
            "shared": arch.num_shared_experts,
            "total_routed_slots": arch.num_layers * arch.num_experts,
        }
    except Exception as e:
        blockers.append(f"transformers AutoConfig failed: {e}")

    # 2) weight index layout
    if check_weights_index:
        try:
            from huggingface_hub import hf_hub_download
            import json

            idx_path = hf_hub_download(model_id, "model.safetensors.index.json")
            wm = json.load(open(idx_path))["weight_map"]
            keys = list(wm.keys())
            checks["n_tensors"] = len(keys)
            has_ffn_experts = any("ffn.experts." in k for k in keys)
            has_mlp_experts = any("mlp.experts" in k for k in keys)
            expert_ids: set[int] = set()
            layers: set[int] = set()
            for k in keys:
                m = re.search(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.", k)
                if m:
                    layers.add(int(m.group(1)))
                    expert_ids.add(int(m.group(2)))
            checks["weight_layout"] = {
                "has_ffn_experts": has_ffn_experts,
                "has_mlp_experts": has_mlp_experts,
                "n_layers_with_experts": len(layers),
                "n_expert_ids": len(expert_ids),
                "sample": [k for k in keys if "ffn.experts.0" in k][:6],
            }
            if not has_ffn_experts and not has_mlp_experts:
                blockers.append(
                    "Weight index has no ffn.experts / mlp.experts tensors"
                )
            if expert_ids and max(expert_ids) != 255:
                warnings.append(
                    f"max expert id {max(expert_ids)} (expected 255 for 256 experts)"
                )
        except Exception as e:
            warnings.append(f"Could not fetch weight index (offline?): {e}")

    # 3) PEFT target_parameters support
    try:
        import peft
        from peft import LoraConfig

        checks["peft_version"] = peft.__version__
        try:
            LoraConfig(
                r=8,
                target_parameters=["gate_up_proj", "down_proj"],
                target_modules=[],
                task_type="CAUSAL_LM",
                lora_dropout=0.0,
            )
            checks["peft_target_parameters"] = True
        except TypeError:
            checks["peft_target_parameters"] = False
            blockers.append(
                "PEFT too old for target_parameters — need peft>=0.15 that supports "
                "LoraConfig(target_parameters=...) for fused V4 experts"
            )
    except Exception as e:
        blockers.append(f"peft import failed: {e}")

    # 4) meta init structure check (optional, no weights)
    peft_attach_ok = False
    if try_meta_init and cfg is not None:
        try:
            import torch
            from transformers import AutoModelForCausalLM

            # meta device may not work for all versions; fall back to CPU empty
            model = None
            try:
                with torch.device("meta"):
                    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
            except Exception:
                try:
                    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
                except Exception:
                    model = AutoModelForCausalLM.from_config(cfg)

            mods = list_v4_runtime_modules(model)
            checks["meta_modules"] = {
                k: (v if not isinstance(v, list) else len(v) if k.startswith("n_") else v[:3])
                for k, v in mods.items()
            }
            # keep counts only for large lists
            checks["meta_modules"] = {
                "n_expert_modules": mods["n_expert_modules"],
                "n_gate_modules": mods["n_gate_modules"],
                "n_shared_modules": mods.get("n_shared_modules", 0),
                "sample_expert": (mods["expert_modules"] or [None])[0],
                "sample_shared": (mods["shared_modules"] or [None])[0],
            }
            if mods["n_expert_modules"] == 0:
                warnings.append(
                    "meta init found 0 DeepseekV4Experts modules — verify after real load"
                )
            else:
                exp_params = []
                for n, p in model.named_parameters():
                    if "experts" in n and ("gate_up" in n or n.endswith("down_proj")):
                        exp_params.append((n, tuple(p.shape)))
                checks["expert_param_shapes"] = exp_params[:6]
                if not exp_params:
                    blockers.append(
                        "No gate_up_proj/down_proj on experts after from_config — "
                        "cannot ESFT/LoRA routed experts"
                    )
                else:
                    # sanity shapes
                    for n, shp in exp_params:
                        if "gate_up" in n and len(shp) == 3 and shp[0] != arch.num_experts:
                            warnings.append(f"{n} expert dim {shp[0]} != {arch.num_experts}")

            # 4b) PEFT attach on tiny clone is heavy; smoke with a tiny fake-like
            # structure only if not on meta (meta + peft often fails)
            if try_peft_attach and checks.get("peft_target_parameters"):
                try:
                    # Build a tiny structural twin and attach LoRA there
                    import torch.nn as nn
                    from peft import LoraConfig, get_peft_model, TaskType

                    class _TinyExperts(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.is_transposed = False
                            self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 8))
                            self.down_proj = nn.Parameter(torch.zeros(4, 8, 4))

                    class _Tiny(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.layers = nn.ModuleList(
                                [nn.Module() for _ in range(1)]
                            )
                            self.layers[0].mlp = nn.Module()
                            self.layers[0].mlp.experts = _TinyExperts()
                            self.layers[0].mlp.shared_experts = nn.Module()
                            self.layers[0].mlp.shared_experts.gate_proj = nn.Linear(8, 4)
                            self.layers[0].mlp.shared_experts.up_proj = nn.Linear(8, 4)
                            self.layers[0].mlp.shared_experts.down_proj = nn.Linear(4, 8)

                        def forward(self, input_ids=None, labels=None, **kw):
                            return type("O", (), {"loss": torch.tensor(0.0)})()

                    tiny = _Tiny()
                    lcfg = LoraConfig(
                        r=2,
                        target_parameters=["gate_up_proj", "down_proj"],
                        target_modules=["gate_proj", "up_proj", "down_proj"],
                        task_type=TaskType.FEATURE_EXTRACTION,
                        bias="none",
                        lora_dropout=0.0,
                    )
                    pt = get_peft_model(tiny, lcfg)
                    n_train = sum(p.numel() for p in pt.parameters() if p.requires_grad)
                    peft_attach_ok = n_train > 0
                    checks["peft_attach_smoke"] = {
                        "ok": peft_attach_ok,
                        "trainable_params": n_train,
                    }
                    # grad mask smoke
                    handles = install_expert_grad_mask(
                        pt,
                        selected=[(0, 0), (0, 1)],
                        arch=MoEArchitectureInfo(
                            family="deepseek_v4_flash",
                            num_layers=1,
                            num_experts=4,
                            num_experts_per_tok=2,
                            num_shared_experts=1,
                            expert_module_pattern=r"layers\.(\d+)\.mlp\.experts",
                            router_module_pattern=r"layers\.(\d+)\.mlp\.gate",
                        ),
                    )
                    checks["grad_mask_hooks"] = len(handles)
                    for h in handles:
                        h.remove()
                    del pt, tiny
                except Exception as e:
                    checks["peft_attach_smoke"] = {"ok": False, "error": str(e)}
                    blockers.append(f"PEFT attach smoke failed: {e}")

            del model
        except Exception as e:
            warnings.append(f"meta/from_config structure probe failed: {e}")

    # 5) synthetic expert refs count
    refs = synthetic_expert_refs(arch)
    checks["synthetic_expert_refs"] = len(refs)
    expected_refs = arch.num_layers * arch.num_experts
    if len(refs) != expected_refs:
        blockers.append(
            f"synthetic refs {len(refs)} != layers*experts {expected_refs}"
        )

    # 6) VRAM / train plan
    train_plan = {
        "recommended_model_id": FLASH_0731_ID,
        "method": "esft_lora",
        "backend": "peft",
        "quantization": (
            "prefer native FP4/FP8 checkpoint as released; "
            "or multi-GPU shard without 4bit if convert path unavailable"
        ),
        "lora": {
            "target_parameters": ["gate_up_proj", "down_proj"],
            "target_modules_shared": ["gate_proj", "up_proj", "down_proj"],
            "r": 64,
            "alpha": 128,
            "lora_dropout": 0.0,
            "note": "ParamWrapper forbids non-zero dropout on fused expert params",
        },
        "expert_selection": (
            "group studio / affinity → PEFT multi-expert LoRA + grad mask on expert dim"
        ),
        "hardware": {
            "min_practical": (
                "multi-GPU 4–8×80GB with PEFT, or Vast multi-GPU instance "
                "(aetherforge remote)"
            ),
            "full_bf16_impossible_on_single_consumer": True,
            "note": (
                "284B total / ~13B active; train adapters only. "
                "Do not expect single 24GB consumer GPU full train."
            ),
        },
        "config_file": "configs/deepseek_v4_flash.yaml",
        "validate_cmd": "aetherforge validate-flash",
        "train_cmd": (
            "aetherforge train -c configs/base.yaml -c configs/deepseek_v4_flash.yaml "
            "-c recipes/flagship_flash_domain.yaml --dry-run   # structure first\n"
            "aetherforge remote launch -c configs/base.yaml -c configs/deepseek_v4_flash.yaml "
            "-c recipes/flagship_flash_domain.yaml --exec   # real GPU box"
        ),
    }

    ok = len(blockers) == 0
    return FlashValidateReport(
        ok=ok,
        model_id=model_id,
        checks=checks,
        blockers=blockers,
        warnings=warnings,
        train_plan=train_plan,
    )
