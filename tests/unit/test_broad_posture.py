"""Broad / wide training posture tests."""

from aetherforge.affinity.expert_selector import ExpertSelector
from aetherforge.affinity.probe import AffinityResult
from aetherforge.models.moe_utils import ExpertRef
from aetherforge.utils.config import (
    AffinityConfig,
    AetherForgeConfig,
    apply_posture_defaults,
    load_config,
)
import numpy as np


def _fake_affinity(n_layers=4, n_exp=8):
    rng = np.random.default_rng(0)
    aff = rng.random((n_layers, n_exp))
    ranked = []
    for li in range(n_layers):
        for ei in range(n_exp):
            ranked.append((li, ei, float(aff[li, ei])))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return AffinityResult(
        domain="multi",
        family="qwen_a3b",
        num_experts=n_exp,
        num_layers=n_layers,
        routing_freq=aff * 10,
        grad_contrib=np.zeros_like(aff),
        affinity=aff,
        ranked=ranked,
        entropy_per_layer=[1.0] * n_layers,
        load_balance_cv=1.0,
        probe_tokens=32,
    )


def test_posture_defaults_broad_wide():
    cfg = apply_posture_defaults(
        AetherForgeConfig.model_validate({"training": {"posture": "broad"}})
    )
    assert cfg.affinity.top_k_fraction == 0.28
    assert cfg.groups.train_scope == "top_n"
    assert cfg.training.include_attention is True

    cfgw = apply_posture_defaults(
        AetherForgeConfig.model_validate({"training": {"posture": "wide"}})
    )
    assert cfgw.affinity.top_k_fraction == 1.0
    assert cfgw.groups.train_scope == "all"
    assert cfgw.training.mask_unselected_experts is False


def test_selector_wide_selects_all():
    aff = _fake_affinity()
    experts = [
        ExpertRef(li, ei, f"L{li}/E{ei}")
        for li in range(aff.num_layers)
        for ei in range(aff.num_experts)
    ]
    cfg = AffinityConfig(top_k_fraction=1.0, freeze_low_affinity=False)
    plan = ExpertSelector(cfg, posture="wide").select(aff, experts)
    assert len(plan.selected) == len(experts)
    assert plan.metadata["posture"] == "wide"


def test_selector_broad_fraction():
    aff = _fake_affinity(n_layers=10, n_exp=10)  # 100 slots
    experts = [
        ExpertRef(li, ei, f"L{li}/E{ei}")
        for li in range(10)
        for ei in range(10)
    ]
    cfg = AffinityConfig(top_k_fraction=0.28, freeze_low_affinity=True)
    plan = ExpertSelector(cfg, posture="broad").select(aff, experts)
    # ~28 of 100
    assert 20 <= len(plan.selected) <= 35
    assert plan.metadata["posture"] == "broad"


def test_broad_recipe_loads():
    cfg = load_config(
        "configs/base.yaml",
        "configs/deepseek_v4_flash.yaml",
        "recipes/broad_flash_192gb.yaml",
    )
    assert cfg.training.posture == "broad"
    assert cfg.groups.train_scope == "top_n"
    assert cfg.groups.train_top_n >= 4
    assert cfg.data.mix_paths
