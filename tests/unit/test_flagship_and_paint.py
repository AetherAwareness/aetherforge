from pathlib import Path

from aetherforge.groups.studio import (
    create_studio_plan,
    display_cell_to_real_cells,
    lattice_view,
)
from aetherforge.groups.store import apply_plan_patch, save_group_plan, load_group_plan
from aetherforge.training.flagship_report import write_flagship_report
from aetherforge.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def test_flagship_recipe_loads():
    cfg = load_config(
        ROOT / "configs" / "base.yaml",
        ROOT / "recipes" / "flagship_logistics_a3b.yaml",
    )
    assert cfg.data.domain == "logistics"
    assert cfg.groups.enabled is True
    assert Path(cfg.data.curated_path).as_posix().endswith("logistics/train.jsonl")
    # sample files exist
    assert (ROOT / "data/samples/logistics/train.jsonl").exists()
    assert (ROOT / "data/samples/logistics/eval.jsonl").exists()


def test_display_cell_mapping_block():
    cells = display_cell_to_real_cells(
        1, 2, step_l=2, step_e=3, num_layers=10, num_experts=20
    )
    # row1 → layers 2-3, col2 → experts 6-8
    assert {"layer": 2, "expert": 6} in cells
    assert {"layer": 3, "expert": 8} in cells
    assert len(cells) == 2 * 3


def test_paint_move_cells(tmp_path):
    plan = create_studio_plan(family="qwen_a3b", num_groups=3, strategy="round_robin")
    g0, g1 = plan.groups[0], plan.groups[1]
    # take a cell from g0
    cell = g0.cells[0]
    plan = apply_plan_patch(
        plan,
        {
            "move_cells": {
                "to": g1.id,
                "exclusive": True,
                "cells": [{"layer": cell.layer, "expert": cell.expert}],
            }
        },
    )
    assert (cell.layer, cell.expert) in g1.cell_keys() or (
        cell.layer,
        cell.expert,
    ) in plan.group_by_id(g1.id).cell_keys()
    assert (cell.layer, cell.expert) not in plan.group_by_id(g0.id).cell_keys()

    lat = lattice_view(plan)
    assert "step_l" in lat and "step_e" in lat
    path = tmp_path / "expert_groups.json"
    save_group_plan(plan, path)
    assert load_group_plan(path).groups


def test_flagship_report_writer(tmp_path):
    p = write_flagship_report(
        tmp_path,
        state={
            "run_id": "x",
            "promoted": True,
            "duration_sec": 1.2,
            "stages": {
                "data": {"domain": "logistics", "n_train": 10, "n_eval": 4},
                "groups": {"n_groups": 6, "n_train_groups": 2},
                "esft": {"steps": 10, "final_loss": 1.5, "dry_run": True},
                "scorecard": {"passed": True, "metrics": {"domain_score": 0.9}},
            },
        },
        recipe="flagship-logistics-a3b",
    )
    assert p.exists()
    import json

    rep = json.loads(p.read_text())
    assert rep["schema"].startswith("aetherforge.flagship")
    assert rep["scorecard"]["passed"] is True
