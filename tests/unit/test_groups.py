from pathlib import Path

from aetherforge.groups.capacity import estimate_capacity
from aetherforge.groups.cluster import auto_partition_groups
from aetherforge.groups.studio import (
    create_studio_plan,
    group_plan_to_selection,
    lattice_view,
    analyze_group,
)
from aetherforge.groups.store import save_group_plan, load_group_plan, apply_plan_patch


def test_flash_capacity_and_group_count():
    cap = estimate_capacity(
        "deepseek_v4_flash",
        model_name="deepseek-ai/DeepSeek-V4-Flash",
        total_params_b=284,
        active_params_b=13,
    )
    assert cap.active_params_b == 13.0
    assert cap.total_params_b == 284.0
    assert cap.num_experts == 256
    assert cap.max_disjoint_active_groups >= 1
    assert cap.total_expert_slots == cap.num_layers * cap.num_experts


def test_auto_partition_num_groups():
    plan = create_studio_plan(
        family="deepseek_v4_flash",
        num_groups=12,
        strategy="active_slots",
    )
    assert len(plan.groups) == 12
    assert plan.capacity.active_params_b == 13.0
    # each group has some cells
    assert all(len(g.cells) > 0 for g in plan.groups)
    # fire ratios computed
    assert any(g.active_fire_ratio > 0 for g in plan.groups)


def test_selection_from_groups():
    plan = create_studio_plan(family="qwen_a3b", num_groups=4)
    # train only first two
    for i, g in enumerate(plan.groups):
        g.train = i < 2
        g.enabled = True
    sel = group_plan_to_selection(plan, domain="demo")
    assert len(sel.selected) > 0
    assert sel.metadata["n_groups_train"] == 2


def test_patch_and_persist(tmp_path):
    plan = create_studio_plan(family="generic_moe", num_groups=3)
    gid = plan.groups[0].id
    plan = apply_plan_patch(
        plan,
        {
            "update_group": {
                "id": gid,
                "name": "alpha_sector",
                "domain": "logistics",
                "train": True,
                "topics": ["inventory", "routing"],
            }
        },
    )
    g = plan.group_by_id(gid)
    assert g.name == "alpha_sector"
    assert g.domain == "logistics"
    path = tmp_path / "expert_groups.json"
    save_group_plan(plan, path)
    plan2 = load_group_plan(path)
    assert plan2.group_by_id(gid).name == "alpha_sector"


def test_lattice_view():
    plan = create_studio_plan(family="deepseek_v4_flash", num_groups=6)
    lat = lattice_view(plan)
    assert lat["layers"] > 0
    assert lat["experts"] == 256
    assert "membership" in lat
    assert len(lat["colors"]) == 6


def test_analyze_group():
    plan = create_studio_plan(family="qwen_a3b", num_groups=3)
    gid = plan.groups[0].id
    a = analyze_group(plan, gid)
    assert a["analysis"]["n_cells"] == len(plan.groups[0].cells)
    assert "active_fire_ratio" in a["analysis"]
