"""Visual / sector-forge payload tests."""

from pathlib import Path

from aetherforge.utils.config import load_config
from aetherforge.training.pipeline import TrainingPipeline
from aetherforge.viz import run_store


ROOT = Path(__file__).resolve().parents[2]


def test_live_status_has_sector_wave(tmp_path):
    cfg = load_config(
        ROOT / "configs" / "base.yaml",
        ROOT / "recipes" / "generic_dryrun.yaml",
        overrides={
            "run": {"dry_run": True, "name": "viz-sector"},
            "training": {
                "output_dir": str(tmp_path / "runs"),
                "sector_mode": "sequential",
                "max_steps": 5,
            },
            "data": {"synthetic": {"num_samples": 32}},
            "eval": {
                "scorecard_thresholds": {
                    "domain_score": 0.2,
                    "routing_entropy_min": 0.2,
                    "load_balance_cv_max": 8.0,
                    "hallucination_max": 0.8,
                    "require_human_approval": False,
                    "high_stakes": False,
                }
            },
        },
    )
    pipe = TrainingPipeline(cfg)
    result = pipe.run()
    live_path = pipe.root / "live_status.json"
    assert live_path.exists()
    import json

    live = json.loads(live_path.read_text())
    assert live.get("schema") in ("aetherforge.live_status.v1", "aetherforge.live_status.v2")
    sec = live.get("sectors") or {}
    assert sec.get("n_trained", 0) >= 1
    assert sec.get("items")
    assert any(i.get("forensics_summary") for i in sec["items"])
    assert live.get("sector_mode") == "sequential"
    assert (live.get("visual") or {}).get("hero_label")

    bundle = run_store.load_run_bundle(pipe.root)
    assert bundle.get("sector_forge")
    assert bundle["sector_forge"].get("items")
    assert bundle["artifacts"].get("sector_workflow")
    summary = run_store.summarize_run(pipe.root)
    assert summary["has_sector_workflow"]
    assert summary["sector_mode"] == "sequential"
    assert result.get("promoted") is not None


def test_sector_forge_js_exists():
    js = ROOT / "aetherforge" / "viz" / "static" / "sector-forge.js"
    assert js.exists()
    text = js.read_text(encoding="utf-8")
    assert "renderSectorForgePanel" in text
    assert "drawSectorOrbit" in text
    dash = ROOT / "aetherforge" / "viz" / "static" / "dashboard.html"
    html = dash.read_text(encoding="utf-8")
    assert "sector-forge.js" in html
    assert 'data-theme-btn="aurora"' in html
    assert "Sector Forge" in html or "sector-forge" in html
