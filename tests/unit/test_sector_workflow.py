"""Sector forensics gate, per-sector datasets, and sequential workflow."""

from pathlib import Path

from aetherforge.data.sector_datasets import SectorDatasetForge, sector_match_score
from aetherforge.groups.readiness import (
    assess_sector_readiness,
    auto_bind_sector_from_forensics,
    readiness_markdown,
    run_forensics_gate,
)
from aetherforge.groups.studio import create_studio_plan, selection_for_group
from aetherforge.training.sector_workflow import SectorWorkflow
from aetherforge.utils.config import DataConfig, TrainingConfig


def test_auto_bind_and_readiness_gate():
    plan = create_studio_plan(
        family="qwen_a3b",
        model_name="Qwen/Qwen3-30B-A3B",
        num_groups=4,
        strategy="active_slots",
    )
    # leave groups unbound — gate should auto-bind from structure/themes
    report = run_forensics_gate(
        plan,
        mode="warn",
        auto_bind=True,
        global_domain="demo_field",
    )
    assert report.n_pass + report.n_warn + report.n_block == len(
        plan.enabled_train_groups()
    )
    assert report.overall in ("pass", "warn")
    # global domain applied
    assert any(g.domain == "demo_field" for g in plan.groups)
    md = readiness_markdown(report)
    assert "readiness" in md.lower() or "Forensic" in md


def test_selection_for_group_freezes_siblings():
    plan = create_studio_plan(family="qwen_a3b", num_groups=4)
    g0 = plan.groups[0]
    sel = selection_for_group(plan, g0.id, domain="demo")
    assert len(sel.selected) == len(g0.cells)
    assert sel.metadata["group_id"] == g0.id
    # frozen should cover rest of lattice
    total = plan.capacity.num_layers * plan.capacity.num_experts
    assert len(sel.selected) + len(sel.frozen) == total
    selected_keys = {(e.layer_idx, e.expert_idx) for e in sel.selected}
    for e in sel.frozen:
        assert (e.layer_idx, e.expert_idx) not in selected_keys


def test_sector_match_and_dataset_forge(tmp_path):
    plan = create_studio_plan(family="qwen_a3b", num_groups=3)
    plan.groups[0].domain = "logistics"
    plan.groups[0].keywords = ["inventory", "warehouse", "stockout", "sla"]
    plan.groups[0].topics = ["safety stock", "lead time"]
    plan.groups[1].domain = "code"
    plan.groups[1].keywords = ["python", "api", "debug", "function"]
    plan.groups[1].topics = ["unit test"]

    records = [
        {"text": "How should safety stock change when warehouse lead time doubles?", "domain": "logistics"},
        {"text": "Write a Python function that merges two sorted lists and add unit tests.", "domain": "code"},
        {"text": "Debug this API stack trace after a null pointer exception.", "domain": "code"},
        {"text": "Plan inventory routing under carrier SLA pressure and stockout risk."},
        {"text": "General chat about the weather today."},
    ]
    # match scores
    assert sector_match_score(records[0], plan.groups[0]) > sector_match_score(
        records[0], plan.groups[1]
    )
    assert sector_match_score(records[1], plan.groups[1]) > 0.2

    forge = SectorDatasetForge(
        DataConfig(domain="demo_field", seed=1),
        min_match=0.15,
        min_samples=6,
        synthesize_fill=True,
        max_synth_per_sector=12,
    )
    ds = forge.build(
        plan,
        records,
        output_dir=tmp_path / "sectors",
    )
    assert len(ds.shards) == len(plan.enabled_train_groups())
    assert (tmp_path / "sectors" / "sector_dataset_plan.json").exists()
    # logistics / code shards should have samples (synth fills thin ones)
    for sh in ds.shards:
        assert len(sh.train_records) >= 1
        assert Path(sh.paths["train"]).exists()


def test_empty_sector_blocks_readiness():
    plan = create_studio_plan(family="qwen_a3b", num_groups=2)
    g = plan.groups[0]
    g.cells = []
    g.refresh_counts()
    r = assess_sector_readiness(plan, g, mode="block")
    assert r.status == "block"
    assert r.train_eligible is False


def test_sequential_workflow_dry(tmp_path):
    plan = create_studio_plan(family="qwen_a3b", num_groups=3)
    plan.groups[0].domain = "ops"
    plan.groups[0].keywords = ["process", "failure", "recovery"]
    plan.groups[1].domain = "analysis"
    plan.groups[1].keywords = ["evidence", "hypothesis", "tradeoff"]

    records = [
        {"text": f"Process failure recovery planning case {i} with evidence tradeoffs."}
        for i in range(24)
    ] + [
        {"text": f"Hypothesis ranking under incomplete evidence sample {i}."}
        for i in range(12)
    ]

    wf = SectorWorkflow(
        plan,
        TrainingConfig(max_steps=5, sector_min_samples=4),
        DataConfig(domain="demo_field", seed=7, topics=["prioritization"], keywords=["evidence"]),
        dry_run=True,
        gate_mode="warn",
        auto_bind=True,
        min_samples=4,
    )
    result = wf.run(records, tmp_path / "wf")
    assert result.n_trained >= 1
    assert (tmp_path / "wf" / "sector_workflow.json").exists()
    assert (tmp_path / "wf" / "sector_readiness.json").exists()
    assert (tmp_path / "wf" / "sector_datasets" / "sector_dataset_plan.json").exists()
    # each trained sector has PRE_TRAIN_FORENSICS.md
    for s in result.sectors:
        if s.status in ("dry_run", "trained"):
            assert Path(s.paths["dir"]).joinpath("PRE_TRAIN_FORENSICS.md").exists()


def test_auto_bind_from_forensics_only_fills_empty():
    plan = create_studio_plan(family="deepseek_v4_flash", num_groups=2)
    g = plan.groups[0]
    g.domain = "already-set"
    g.topics = ["keep-me"]
    from aetherforge.groups.forensics import forensics_for_group

    d = forensics_for_group(plan, g.id)
    applied = auto_bind_sector_from_forensics(g, d, global_domain="other")
    assert g.domain == "already-set"
    assert "domain" not in applied
    assert g.topics == ["keep-me"]
