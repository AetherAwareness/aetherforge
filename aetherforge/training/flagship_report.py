"""Write a compact flagship_report.json for the public proof recipe."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from aetherforge.utils.logging import get_logger

log = get_logger("training.flagship_report")


def write_flagship_report(
    run_dir: str | Path,
    *,
    state: dict[str, Any],
    scorecard: Optional[dict[str, Any]] = None,
    groups_summary: Optional[dict[str, Any]] = None,
    recipe: str = "flagship_logistics_a3b",
) -> Path:
    run_dir = Path(run_dir)
    sc = scorecard or state.get("stages", {}).get("scorecard") or {}
    gs = groups_summary or state.get("stages", {}).get("groups") or {}
    data = state.get("stages", {}).get("data") or {}
    esft = state.get("stages", {}).get("esft") or {}

    report = {
        "schema": "aetherforge.flagship_report.v1",
        "recipe": recipe,
        "run_id": state.get("run_id"),
        "created_at": time.time(),
        "promoted": state.get("promoted"),
        "duration_sec": state.get("duration_sec"),
        "domain": data.get("domain") or sc.get("domain"),
        "data": {
            "n_train": data.get("n_train"),
            "n_eval": data.get("n_eval"),
            "fingerprint": data.get("fingerprint"),
            "diversity": (data.get("quality") or {}).get("diversity"),
        },
        "groups": {
            "n_groups": gs.get("n_groups"),
            "n_train_groups": gs.get("n_train_groups"),
            "n_cells_train": gs.get("n_cells_train"),
            "active_params_b": gs.get("active_params_b"),
        },
        "training": {
            "steps": esft.get("steps"),
            "final_loss": esft.get("final_loss"),
            "dry_run": esft.get("dry_run"),
            "trainable_params": esft.get("trainable_params"),
        },
        "scorecard": {
            "passed": sc.get("passed"),
            "metrics": sc.get("metrics") or {},
            "failures": (sc.get("gate") or {}).get("failures") or [],
            "mode": (sc.get("details") or {}).get("mode"),
        },
        "how_to_reproduce": [
            "bash scripts/run_flagship.sh dry-run",
            "aetherforge dashboard  # paint sectors, set train flags",
            "bash scripts/run_flagship.sh live   # requires GPU",
            "bash scripts/run_flagship.sh report",
        ],
    }
    path = run_dir / "flagship_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("Flagship report → %s", path)
    return path
