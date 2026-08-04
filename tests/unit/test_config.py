from pathlib import Path

from aetherforge.utils.config import AetherForgeConfig, load_config, merge_configs


ROOT = Path(__file__).resolve().parents[2]


def test_default_config_validates():
    cfg = AetherForgeConfig()
    assert cfg.data.domain == "general"
    assert cfg.affinity.top_k_experts > 0
    # no medical_* fields on thresholds
    thr = cfg.eval.scorecard_thresholds
    assert not hasattr(thr, "medical_score_min") or thr.model_dump().get("medical_score_min") is None
    assert "domain_depth_min" in thr.model_dump()


def test_load_base_yaml():
    cfg = load_config(ROOT / "configs" / "base.yaml")
    assert cfg.model.name
    assert cfg.training.method == "esft_lora"


def test_load_example_industry_yaml():
    cfg = load_config(
        ROOT / "configs" / "base.yaml",
        ROOT / "configs" / "domains" / "example_logistics.yaml",
    )
    assert cfg.data.domain == "logistics"
    assert "inventory" in cfg.data.keywords
    assert cfg.eval.scorecard_thresholds.require_human_approval is False


def test_merge_dicts():
    m = merge_configs(
        ROOT / "configs" / "base.yaml",
        overrides={"data": {"domain": "securities_law"}},
    )
    assert m["data"]["domain"] == "securities_law"
