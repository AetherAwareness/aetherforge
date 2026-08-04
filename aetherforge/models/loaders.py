"""
Unified MoE model loading: Unsloth (preferred) → PEFT/Transformers fallback.

Supports:
  - Qwen A3B-class on single/few GPUs (QLoRA / LoRA)
  - DeepSeek-V4-Flash-class with optional EP/TP hints for multi-node
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aetherforge.models.moe_utils import (
    MoEArchitectureInfo,
    detect_moe_architecture,
    list_expert_modules,
    ExpertRef,
)
from aetherforge.utils.config import ModelConfig, TrainingConfig
from aetherforge.utils.logging import get_logger

log = get_logger("models.loaders")


def _is_v4_family(arch: MoEArchitectureInfo, model_name: str = "", model: Any = None) -> bool:
    from aetherforge.models.deepseek_v4 import is_deepseek_v4_bundle

    return is_deepseek_v4_bundle(family=arch.family, model_name=model_name, model=model)


@dataclass
class MoEModelBundle:
    """Loaded model + tokenizer + architecture metadata."""

    model: Any
    tokenizer: Any
    arch: MoEArchitectureInfo
    backend: str  # unsloth | transformers | peft
    experts: list[ExpertRef] = field(default_factory=list)
    model_name: str = ""
    device_map: Any = "auto"
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "family": self.arch.family,
            "num_experts": self.arch.num_experts,
            "num_experts_per_tok": self.arch.num_experts_per_tok,
            "num_shared_experts": self.arch.num_shared_experts,
            "num_layers": self.arch.num_layers,
            "expert_modules_found": len(self.experts),
        }


def _resolve_source(cfg: ModelConfig) -> str:
    if cfg.local_path:
        return cfg.local_path
    return cfg.name


def _try_unsloth(
    source: str,
    cfg: ModelConfig,
    train_cfg: Optional[TrainingConfig],
) -> Optional[tuple[Any, Any, str]]:
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except Exception as e:
        log.info("Unsloth not available (%s); will fall back to transformers", e)
        return None

    max_seq = cfg.max_seq_length
    load_in_4bit = cfg.load_in_4bit
    if train_cfg and train_cfg.method == "qlora":
        load_in_4bit = True

    log.info("Loading via Unsloth: %s (4bit=%s, seq=%d)", source, load_in_4bit, max_seq)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source,
        max_seq_length=max_seq,
        dtype=None,
        load_in_4bit=load_in_4bit,
        trust_remote_code=cfg.trust_remote_code,
    )
    return model, tokenizer, "unsloth"


def _load_transformers(source: str, cfg: ModelConfig) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

    kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    if cfg.revision:
        kwargs["revision"] = cfg.revision
    if cfg.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as e:
            log.warning("4bit requested but BitsAndBytes unavailable: %s", e)
    elif cfg.load_in_8bit:
        kwargs["load_in_8bit"] = True

    log.info("Loading via transformers: %s", source)
    tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    return model, tokenizer, "transformers"


def load_moe_model(
    model_cfg: ModelConfig,
    train_cfg: Optional[TrainingConfig] = None,
    backend: str = "auto",
    for_training: bool = True,
) -> MoEModelBundle:
    """
    Load an open sparse MoE with a common wrapper interface.

    backend:
      auto     — try Unsloth, else transformers
      unsloth  — require Unsloth
      peft / transformers — HF path
    """
    source = _resolve_source(model_cfg)
    if not source:
        raise ValueError("model.name or model.local_path is required")

    backend_req = backend
    if train_cfg and backend == "auto":
        backend_req = train_cfg.backend

    model = tokenizer = None
    used_backend = "transformers"

    if backend_req in ("auto", "unsloth"):
        result = _try_unsloth(source, model_cfg, train_cfg)
        if result is not None:
            model, tokenizer, used_backend = result
        elif backend_req == "unsloth":
            raise RuntimeError("Unsloth backend requested but not importable")

    if model is None:
        model, tokenizer, used_backend = _load_transformers(source, model_cfg)

    arch = detect_moe_architecture(
        model, family_hint=model_cfg.family, model_name=source
    )
    # Allow config overrides when auto-detect is weak
    if model_cfg.num_experts:
        arch.num_experts = model_cfg.num_experts
    if model_cfg.num_experts_per_tok:
        arch.num_experts_per_tok = model_cfg.num_experts_per_tok
    if model_cfg.num_shared_experts is not None:
        arch.num_shared_experts = model_cfg.num_shared_experts

    # DeepSeek-V4: upgrade arch notes + synthetic expert refs if ModuleList walk fails
    if _is_v4_family(arch, source, model):
        from aetherforge.models.deepseek_v4 import (
            arch_from_flash_config,
            synthetic_expert_refs,
        )

        try:
            cfg_obj = getattr(model, "config", None)
            if cfg_obj is not None:
                v4_arch = arch_from_flash_config(cfg_obj, model_name=source)
                # preserve explicit overrides
                if model_cfg.num_experts:
                    v4_arch.num_experts = model_cfg.num_experts
                if model_cfg.num_experts_per_tok:
                    v4_arch.num_experts_per_tok = model_cfg.num_experts_per_tok
                if model_cfg.num_shared_experts is not None:
                    v4_arch.num_shared_experts = model_cfg.num_shared_experts
                arch = v4_arch
        except Exception as e:
            log.warning("V4 arch_from_flash_config failed: %s", e)

    experts = list_expert_modules(model, arch)
    if not experts and _is_v4_family(arch, source, model):
        from aetherforge.models.deepseek_v4 import synthetic_expert_refs

        experts = synthetic_expert_refs(arch)
        log.info(
            "V4 fused experts: synthesized %d expert refs (layers=%d × experts=%d)",
            len(experts),
            arch.num_layers,
            arch.num_experts,
        )

    if for_training and hasattr(model, "train"):
        model.train()

    bundle = MoEModelBundle(
        model=model,
        tokenizer=tokenizer,
        arch=arch,
        backend=used_backend,
        experts=experts,
        model_name=source,
    )
    log.info("Loaded MoE bundle: %s", bundle.summary())
    return bundle


def apply_expert_lora(
    bundle: MoEModelBundle,
    train_cfg: TrainingConfig,
    selected_experts: Optional[list[ExpertRef]] = None,
) -> MoEModelBundle:
    """
    Attach LoRA adapters. Prefer Unsloth FastLanguageModel.get_peft_model;
    fall back to PEFT LoraConfig on target modules.

    DeepSeek-V4-Flash uses fused 3D expert params — routes through
    apply_v4_expert_lora (target_parameters) + optional expert grad masks.
    """
    model = bundle.model
    target_modules = list(train_cfg.target_modules)
    target_parameters = list(getattr(train_cfg, "target_parameters", None) or [])

    # ── DeepSeek-V4 Flash path (fused experts) ──────────────────────
    if _is_v4_family(bundle.arch, bundle.model_name, model):
        from aetherforge.models.deepseek_v4 import (
            apply_v4_expert_lora,
            install_expert_grad_mask,
            V4_LORA_DROPOUT,
        )

        # Unsloth rarely supports V4 fused experts yet — try only if forced
        if bundle.backend == "unsloth":
            log.info(
                "DeepSeek-V4 detected: preferring PEFT target_parameters over Unsloth"
            )

        include_shared = any(
            m in ("gate_proj", "up_proj", "down_proj") for m in target_modules
        )
        # Broad/wide: also LoRA attention if requested
        if getattr(train_cfg, "include_attention", False):
            for m in train_cfg.attention_modules or []:
                if m not in target_modules:
                    target_modules.append(m)
        extra = []
        if getattr(train_cfg, "include_attention", False):
            extra = list(train_cfg.attention_modules or [])
        model = apply_v4_expert_lora(
            model,
            r=train_cfg.lora_r,
            lora_alpha=train_cfg.lora_alpha,
            lora_dropout=V4_LORA_DROPOUT,  # ParamWrapper requirement
            include_shared=include_shared if target_modules else True,
            include_router=False,
            extra_modules=extra or None,
        )
        if train_cfg.gradient_checkpointing and hasattr(
            model, "enable_input_require_grads"
        ):
            model.enable_input_require_grads()
        bundle.model = model
        bundle.backend = "peft"

        mask = getattr(train_cfg, "mask_unselected_experts", True)
        if selected_experts and mask:
            pairs = [(e.layer_idx, e.expert_idx) for e in selected_experts]
            handles = install_expert_grad_mask(model, pairs, bundle.arch)
            bundle.extras["expert_grad_mask_handles"] = handles
            bundle.extras["selected_expert_pairs"] = pairs
            log.info(
                "V4 ESFT: LoRA + grad masks on %d selected expert slots",
                len(pairs),
            )
        elif selected_experts and not mask:
            bundle.extras["selected_expert_pairs"] = [
                (e.layer_idx, e.expert_idx) for e in selected_experts
            ]
            log.info(
                "V4 wide LoRA: no expert grad masks — fused banks train fully "
                "(%d selection tags retained for audit)",
                len(selected_experts),
            )
        return bundle

    # ── Generic / Qwen A3B path ─────────────────────────────────────
    if bundle.backend == "unsloth":
        try:
            from unsloth import FastLanguageModel  # type: ignore

            model = FastLanguageModel.get_peft_model(
                model,
                r=train_cfg.lora_r,
                target_modules=target_modules,
                lora_alpha=train_cfg.lora_alpha,
                lora_dropout=train_cfg.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth"
                if train_cfg.gradient_checkpointing
                else False,
                random_state=train_cfg.seed,
            )
            bundle.model = model
            bundle.backend = "unsloth"
            log.info(
                "Applied Unsloth LoRA r=%d targets=%s", train_cfg.lora_r, target_modules
            )
            return bundle
        except Exception as e:
            log.warning("Unsloth LoRA failed (%s); falling back to PEFT", e)

    try:
        from peft import LoraConfig, get_peft_model, TaskType

        if getattr(train_cfg, "include_attention", False):
            for m in train_cfg.attention_modules or []:
                if m not in target_modules:
                    target_modules.append(m)
        kwargs: dict[str, Any] = {
            "r": train_cfg.lora_r,
            "lora_alpha": train_cfg.lora_alpha,
            "lora_dropout": train_cfg.lora_dropout,
            "target_modules": target_modules,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
        }
        if target_parameters:
            kwargs["target_parameters"] = target_parameters
        lora_cfg = LoraConfig(**kwargs)
        model = get_peft_model(model, lora_cfg)
        if train_cfg.gradient_checkpointing and hasattr(
            model, "enable_input_require_grads"
        ):
            model.enable_input_require_grads()
        bundle.model = model
        bundle.backend = "peft"
        log.info(
            "Applied PEFT LoRA r=%d targets=%s params=%s",
            train_cfg.lora_r,
            target_modules,
            target_parameters or None,
        )
    except Exception as e:
        log.error("Failed to apply LoRA: %s", e)
        raise

    return bundle


def save_adapter(bundle: MoEModelBundle, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = bundle.model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out))
    if bundle.tokenizer is not None and hasattr(bundle.tokenizer, "save_pretrained"):
        bundle.tokenizer.save_pretrained(str(out))
    log.info("Saved adapter/model to %s", out)
    return out
