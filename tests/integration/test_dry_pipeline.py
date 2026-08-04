"""Integration: full dry-run pipeline without GPU/models — any industry."""

from pathlib import Path

from aetherforge.utils.config import load_config
from aetherforge.training.pipeline import TrainingPipeline


ROOT = Path(__file__).resolve().parents[2]


def test_dry_run_pipeline(tmp_path):
    cfg = load_config(
        ROOT / "configs" / "base.yaml",
        ROOT / "recipes" / "generic_dryrun.yaml",
        overrides={
            "run": {"dry_run": True, "name": "test-dry"},
            "training": {"output_dir": str(tmp_path / "runs")},
            "data": {"synthetic": {"num_samples": 48}},
            "eval": {
                "scorecard_thresholds": {
                    "domain_score": 0.30,
                    "routing_entropy_min": 0.3,
                    "load_balance_cv_max": 5.0,
                    "hallucination_max": 0.5,
                    "require_human_approval": False,
                    "high_stakes": False,
                }
            },
        },
    )
    assert cfg.run.dry_run is True
    assert cfg.data.domain == "demo_field"
    pipe = TrainingPipeline(cfg)
    result = pipe.run()
    assert result["run_id"]
    assert "data" in result["stages"]
    assert "affinity" in result["stages"]
    assert "lifecycle" in result["stages"]
    assert "scorecard" in result["stages"]
    assert "package" in result["stages"]
    assert (pipe.root / "aetherpackage" / "manifest.json").exists()
    assert (pipe.root / "audit.jsonl").exists()
    assert (pipe.root / "data" / "train.jsonl").exists()
    assert (pipe.root / "data" / "domain_pack.resolved.json").exists()
    assert (pipe.root / "lifecycle" / "lifecycle_plan.json").exists()
    # live dashboard telemetry
    live_path = pipe.root / "live_status.json"
    assert live_path.exists()
    import json

    live = json.loads(live_path.read_text())
    assert live["status"] == "completed"
    assert live["percent"] == 100.0
    assert "diagnostics" in live["stages"]
    assert live["stages"]["data"]["state"] == "done"
    # Expert Group Studio artifact
    assert (pipe.root / "expert_groups.json").exists()
    assert "groups" in result["stages"]
    assert result["stages"]["groups"].get("n_groups", 0) >= 1
    # Sequential sector workflow (default): forensics gate + per-sector ESFT dry-run
    assert (pipe.root / "sector_forensics.json").exists() or (
        pipe.root / "sector_readiness.json"
    ).exists()
    esft = result["stages"].get("esft") or {}
    if esft.get("schema") == "aetherforge.sector_workflow.v1" or esft.get("mode") == "sequential":
        assert (pipe.root / "sector_workflow" / "sector_workflow.json").exists()
        assert esft.get("n_trained", 0) >= 1
        # pre-train forensic cards for trained sectors
        sw = pipe.root / "sector_workflow" / "checkpoints"
        if sw.exists():
            cards = list(sw.glob("*/PRE_TRAIN_FORENSICS.md"))
            assert len(cards) >= 1
    sc = result["stages"]["scorecard"]
    assert sc["details"]["n_eval_texts"] > 0
    assert "text_proxies" in sc["details"]["mode"]
    assert "medical_score" not in sc["metrics"]
    assert result["stages"]["data"]["quality"]["diversity"] >= 0.15
    # no medical strings in resolved domain pack
    pack_txt = (pipe.root / "data" / "domain_pack.resolved.json").read_text().lower()
    assert "troponin" not in pack_txt
    assert "cardiology" not in pack_txt
    if result.get("promoted"):
        assert (pipe.root / "promoted" / "manifest.json").exists()


def test_logistics_domain_config_loads(tmp_path):
    cfg = load_config(
        ROOT / "configs" / "base.yaml",
        ROOT / "configs" / "domains" / "example_logistics.yaml",
        overrides={
            "run": {"dry_run": True, "name": "logistics-dry"},
            "training": {"output_dir": str(tmp_path / "runs"), "max_steps": 5},
            "data": {"synthetic": {"num_samples": 24}},
            "eval": {
                "scorecard_thresholds": {
                    "domain_score": 0.2,
                    "routing_entropy_min": 0.2,
                    "load_balance_cv_max": 9.0,
                    "hallucination_max": 0.9,
                }
            },
        },
    )
    pipe = TrainingPipeline(cfg)
    result = pipe.run(stages=["diagnostics", "data", "affinity", "scorecard", "package"])
    assert result["stages"]["data"]["domain"] == "logistics"
    assert "inventory" in " ".join(result["stages"]["data"].get("keywords") or [])
