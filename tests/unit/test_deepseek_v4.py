"""DeepSeek-V4-Flash-0731 training path unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from aetherforge.models.deepseek_v4 import (
    FLASH_0731_ID,
    FLASH_CANONICAL,
    apply_v4_expert_lora,
    arch_from_flash_config,
    install_expert_grad_mask,
    is_deepseek_v4_name,
    synthetic_expert_refs,
    validate_flash_training_stack,
)
from aetherforge.models.moe_utils import (
    MoEArchitectureInfo,
    detect_moe_architecture,
    infer_family_from_name,
    list_expert_modules,
)
from aetherforge.utils.config import TrainingConfig


def test_flash_name_and_family():
    assert is_deepseek_v4_name("deepseek-ai/DeepSeek-V4-Flash-0731")
    assert infer_family_from_name(FLASH_0731_ID) == "deepseek_v4_flash"


def test_arch_from_canonical():
    arch = arch_from_flash_config(FLASH_CANONICAL, model_name=FLASH_0731_ID)
    assert arch.family == "deepseek_v4_flash"
    assert arch.num_layers == 43
    assert arch.num_experts == 256
    assert arch.num_experts_per_tok == 6
    assert arch.num_shared_experts == 1
    refs = synthetic_expert_refs(arch)
    assert len(refs) == 43 * 256
    assert refs[0].module_name.endswith("#0")
    assert refs[255].expert_idx == 255


def test_detect_from_deepseek_v4_model_type():
    cfg = SimpleNamespace(
        model_type="deepseek_v4",
        n_routed_experts=256,
        num_experts_per_tok=6,
        n_shared_experts=1,
        num_hidden_layers=43,
        hidden_size=4096,
        moe_intermediate_size=2048,
        architectures=["DeepseekV4ForCausalLM"],
        _name_or_path=FLASH_0731_ID,
        to_dict=lambda: {
            "model_type": "deepseek_v4",
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "n_shared_experts": 1,
            "num_hidden_layers": 43,
        },
    )
    info = detect_moe_architecture(cfg, family_hint="auto", model_name=FLASH_0731_ID)
    assert info.family == "deepseek_v4_flash"
    assert info.num_experts == 256
    assert info.num_layers == 43


class _TinyExperts(nn.Module):
    def __init__(self, n_exp=4, hidden=8, inter=4):
        super().__init__()
        self.is_transposed = False
        # Match V4 layout: [E, 2*inter, H] and [E, H, inter]
        self.gate_up_proj = nn.Parameter(torch.randn(n_exp, 2 * inter, hidden) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(n_exp, hidden, inter) * 0.02)


class _TinyV4(nn.Module):
    """Minimal structural twin of DeepseekV4 fused experts for PEFT smoke."""

    def __init__(self, n_layers=2, n_exp=4, hidden=8, inter=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.experts = _TinyExperts(n_exp, hidden, inter)
            shared = nn.Module()
            shared.gate_proj = nn.Linear(hidden, inter, bias=False)
            shared.up_proj = nn.Linear(hidden, inter, bias=False)
            shared.down_proj = nn.Linear(inter, hidden, bias=False)
            layer.mlp.shared_experts = shared
            self.layers.append(layer)
        self.config = SimpleNamespace(
            model_type="deepseek_v4",
            n_routed_experts=n_exp,
            num_experts_per_tok=2,
            n_shared_experts=1,
            num_hidden_layers=n_layers,
            architectures=["DeepseekV4ForCausalLM"],
        )

    def prepare_inputs_for_generation(self, input_ids=None, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def forward(self, input_ids=None, labels=None, **kw):
        # dummy loss so Trainer-like loops can call model
        p = next(self.parameters())
        return SimpleNamespace(loss=p.sum() * 0.0 + torch.tensor(0.1, device=p.device))


def test_list_expert_modules_fused():
    m = _TinyV4(n_layers=2, n_exp=4)
    # Tag experts class name optional — fused path uses gate_up_proj attr
    arch = detect_moe_architecture(m, family_hint="deepseek_v4_flash", model_name=FLASH_0731_ID)
    refs = list_expert_modules(m, arch)
    assert len(refs) == 2 * 4
    assert all(r.family == "deepseek_v4_flash" for r in refs)


def test_apply_v4_lora_and_grad_mask():
    m = _TinyV4(n_layers=1, n_exp=4)
    peft_m = apply_v4_expert_lora(m, r=2, lora_alpha=4, lora_dropout=0.0, include_shared=True)
    trainable = [(n, p.shape) for n, p in peft_m.named_parameters() if p.requires_grad]
    assert trainable, "expected LoRA trainable params"
    # multi-expert LoRA A should pack r*E
    lora_a = [n for n, _ in trainable if "lora_A" in n and "experts" in n]
    assert lora_a, f"expected expert lora_A, got {[n for n,_ in trainable]}"

    arch = MoEArchitectureInfo(
        family="deepseek_v4_flash",
        num_layers=1,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        expert_module_pattern=r"layers\.(\d+)\.mlp\.experts",
        router_module_pattern=r"layers\.(\d+)\.mlp\.gate",
    )
    handles = install_expert_grad_mask(peft_m, selected=[(0, 0), (0, 1)], arch=arch)
    assert len(handles) >= 1

    # one backward step on a loss that touches LoRA weights — ensure hooks don't crash
    train_params = [p for p in peft_m.parameters() if p.requires_grad]
    assert train_params
    loss = sum(p.float().sum() for p in train_params) * 0.0 + train_params[0].float().sum() * 1e-6
    loss.backward()
    for h in handles:
        h.remove()


def test_loaders_apply_expert_lora_routes_v4():
    from aetherforge.models.loaders import MoEModelBundle, apply_expert_lora
    from aetherforge.models.moe_utils import ExpertRef

    m = _TinyV4(n_layers=1, n_exp=4)
    arch = arch_from_flash_config(
        {
            "num_hidden_layers": 1,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "model_type": "deepseek_v4",
        },
        model_name=FLASH_0731_ID,
    )
    bundle = MoEModelBundle(
        model=m,
        tokenizer=None,
        arch=arch,
        backend="transformers",
        experts=synthetic_expert_refs(arch),
        model_name=FLASH_0731_ID,
    )
    cfg = TrainingConfig(
        method="esft_lora",
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.05,  # must be forced to 0 for V4
        target_modules=["gate_proj", "up_proj", "down_proj"],
        target_parameters=["gate_up_proj", "down_proj"],
    )
    selected = [
        ExpertRef(0, 0, "model.layers.0.mlp.experts#0", family="deepseek_v4_flash"),
        ExpertRef(0, 1, "model.layers.0.mlp.experts#1", family="deepseek_v4_flash"),
    ]
    out = apply_expert_lora(bundle, cfg, selected)
    assert out.backend == "peft"
    assert out.extras.get("expert_grad_mask_handles")
    assert len(out.extras["selected_expert_pairs"]) == 2


def test_validate_flash_stack_live():
    """Live HF config + PEFT smoke (network for config/index; no full weights)."""
    report = validate_flash_training_stack(
        FLASH_0731_ID,
        check_weights_index=True,
        try_meta_init=True,
        try_peft_attach=True,
    )
    assert report.ok, f"blockers={report.blockers} warnings={report.warnings}"
    assert report.checks.get("model_type") == "deepseek_v4"
    assert int(report.checks.get("n_routed_experts") or 0) == 256
    assert int(report.checks.get("num_hidden_layers") or 0) == 43
    assert report.checks.get("peft_target_parameters") is True
    assert report.checks.get("synthetic_expert_refs") == 43 * 256
    smoke = report.checks.get("peft_attach_smoke") or {}
    assert smoke.get("ok") is True, smoke
