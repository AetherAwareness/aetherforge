from aetherforge.utils.overrides import coerce_value, parse_overrides
from aetherforge.training.pipeline import resolve_stages


def test_coerce_types():
    assert coerce_value("true") is True
    assert coerce_value("false") is False
    assert coerce_value("null") is None
    assert coerce_value("42") == 42
    assert coerce_value("-3") == -3
    assert coerce_value("1.5") == 1.5
    assert coerce_value("hello") == "hello"
    assert coerce_value("[a,b]") == ["a", "b"]


def test_parse_nested():
    d = parse_overrides(
        [
            "data.domain=logistics",
            "training.max_steps=50",
            "run.dry_run=true",
            "eval.scorecard_thresholds.domain_score=0.7",
        ]
    )
    assert d["data"]["domain"] == "logistics"
    assert d["training"]["max_steps"] == 50
    assert d["run"]["dry_run"] is True
    assert d["eval"]["scorecard_thresholds"]["domain_score"] == 0.7


def test_resolve_stages_aliases():
    stages = resolve_stages(["sft", "dpo", "eval", "export"], [])
    assert stages == ["esft", "preference", "scorecard", "package"]


def test_resolve_stages_full_default():
    stages = resolve_stages(None, ["sft", "router_hygiene"])
    assert "diagnostics" in stages
    assert "lifecycle" in stages
    assert "package" in stages
