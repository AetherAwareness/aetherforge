import json
from pathlib import Path

from aetherforge.viz.progress import LiveProgress, StageState
from aetherforge.viz import run_store


def test_live_progress_writes(tmp_path):
    run = tmp_path / "run1"
    prog = LiveProgress(
        run,
        run_id="abc",
        run_name="test",
        domain="demo_field",
        model="m",
        dry_run=True,
        stage_list=["diagnostics", "data", "package"],
        artifacts_root=tmp_path,
    )
    prog.start_run()
    prog.start_stage("diagnostics")
    prog.end_stage("diagnostics", {"ok": True})
    prog.start_stage("data")
    prog.set_metrics({"domain_score": 0.9})
    prog.end_stage("data", {"n_train": 10})
    prog.finish(ok=True)

    status = json.loads((run / "live_status.json").read_text())
    assert status["run_id"] == "abc"
    assert status["status"] == "completed"
    assert status["stages"]["diagnostics"]["state"] == StageState.DONE.value
    assert status["metrics_snapshot"]["domain_score"] == 0.9
    assert status["percent"] == 100.0
    assert (tmp_path / "active.json").exists()


def test_run_store_list_and_control(tmp_path):
    run = tmp_path / "demo-run-xyz"
    run.mkdir()
    (run / "live_status.json").write_text(
        json.dumps(
            {
                "run_id": "xyz",
                "domain": "logistics",
                "status": "completed",
                "promoted": False,
                "percent": 100,
            }
        ),
        encoding="utf-8",
    )
    (run / "scorecard.json").write_text(
        json.dumps({"passed": True, "metrics": {"domain_score": 0.8}, "gate": {"failures": []}}),
        encoding="utf-8",
    )
    (run / "aetherpackage").mkdir()
    (run / "aetherpackage" / "manifest.json").write_text("{}", encoding="utf-8")

    runs = run_store.list_runs(tmp_path, limit=10)
    assert len(runs) == 1
    assert runs[0]["domain"] == "logistics"

    bundle = run_store.load_run_bundle(run)
    assert bundle["summary"]["run_id"] == "xyz"
    assert bundle["scorecard"]["passed"] is True

    ctrl = run_store.apply_human_approve(run, note="looks good")
    assert ctrl["human_approved"] is True
    assert (run / "operator_controls.json").exists()

    out = run_store.force_promote_package(run, note="ship it")
    assert out["ok"] is True
    assert (run / "promoted" / "manifest.json").exists()
