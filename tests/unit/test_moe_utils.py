from types import SimpleNamespace

from aetherforge.models.moe_utils import (
    detect_moe_architecture,
    infer_family_from_name,
    expert_global_id,
    ExpertRef,
)


def test_infer_family():
    assert infer_family_from_name("deepseek-ai/DeepSeek-V4-Flash-0731") == "deepseek_v4_flash"
    assert infer_family_from_name("Qwen/Qwen3.6-35B-A3B") == "qwen_a3b"


def test_detect_from_config_ns():
    cfg = SimpleNamespace(
        model_type="deepseek_v4",
        n_routed_experts=256,
        num_experts_per_tok=6,
        n_shared_experts=1,
        num_hidden_layers=60,
        _name_or_path="deepseek-ai/DeepSeek-V4-Flash",
    )
    info = detect_moe_architecture(cfg, family_hint="auto", model_name=cfg._name_or_path)
    assert info.family == "deepseek_v4_flash"
    assert info.num_experts == 256
    assert info.num_experts_per_tok == 6


def test_global_id():
    ref = ExpertRef(layer_idx=2, expert_idx=5, module_name="x")
    assert expert_global_id(ref, num_experts=256) == 2 * 256 + 5
