from pathlib import Path

from aetherforge.groups.capacity import estimate_capacity
from aetherforge.models.moe_utils import infer_family_from_name
from aetherforge.ux.recipes import resolve_recipe
from aetherforge.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def test_infer_qwen38_dense_not_a3b():
    assert infer_family_from_name("Qwen/Qwen3.8-27B") == "qwen38_dense"
    assert infer_family_from_name("qwen38-27b-iq4xs") == "qwen38_dense"
    assert infer_family_from_name("Qwen/Qwen3-30B-A3B") == "qwen_a3b"


def test_capacity_qwen38_dense():
    cap = estimate_capacity("qwen38_dense", model_name="Qwen/Qwen3.8-27B")
    assert cap.family == "qwen38_dense"
    assert cap.num_experts == 1
    assert cap.total_params_b == 27.0
    assert cap.active_params_b == 27.0
    assert cap.num_layers == 64


def test_recipe_qwen38_loads():
    meta = resolve_recipe("qwen38")
    assert meta["id"] == "qwen38-27b"
    cfg = load_config(*meta["config_paths"])
    assert cfg.model.family == "qwen38_dense"
    assert cfg.groups.enabled is False
    assert cfg.training.sector_mode == "joint"
    assert cfg.training.method == "qlora"
    assert cfg.model.name == "Qwen/Qwen3.8-27B"
    assert cfg.model.local_path in (None, "")


def test_qwen38_yaml_on_disk():
    assert (ROOT / "configs" / "qwen38_27b.yaml").exists()
    assert (ROOT / "recipes" / "qwen38_27b.yaml").exists()
