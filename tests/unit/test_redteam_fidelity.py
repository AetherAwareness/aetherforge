"""
Adversarial red-team tests for MoE training fidelity (no GPU / no weights).

Covers:
  - evidence tiers + non-saturating themes on unbound geometry
  - high-confidence content refused without routing/assignment
  - plan fingerprint immutability
  - data contracts on thin/all-synth shards
  - synthetic affinity labeling
  - sequential dry-run keep/rollback + interference artifacts
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aetherforge.data.contracts import DataContractSpec, evaluate_sector_contract
from aetherforge.data.sector_datasets import SectorDatasetForge
from aetherforge.groups.evidence import (
    calibrate_theme_scores,
    resolve_evidence_tier,
    score_themes_for_group,
)
from aetherforge.groups.forensics import DEFAULT_THEME_BANK, forensics_for_group, run_model_forensics
from aetherforge.groups.plan_fingerprint import (
    freeze_plan,
    plan_fingerprint,
    verify_plan_fingerprint,
)
from aetherforge.groups.studio import create_studio_plan
from aetherforge.training.sector_probe import (
    decide_keep_rollback,
    interference_summary,
    probe_sector,
    synthetic_post_boost,
)
from aetherforge.training.sector_workflow import SectorWorkflow
from aetherforge.utils.config import DataConfig, TrainingConfig, load_config
from aetherforge.training.pipeline import TrainingPipeline
from aetherforge.eval.scorecard import ReliabilityScorecard
from aetherforge.utils.config import EvalConfig
from aetherforge.affinity.probe import AffinityResult
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


# ── 1. Forensics tiers & calibration ──────────────────────────────────


def test_auto_bind_does_not_invent_multitheme_from_structure_only_zeros():
    """
    Skeptic-fixed: structure_only dossiers with zeroed themes must not paint
    code/math/science/medicine keywords just because global_domain is set.
    """
    from aetherforge.groups.readiness import (
        auto_bind_sector_from_forensics,
        run_forensics_gate,
    )

    plan = create_studio_plan(family="qwen_a3b", num_groups=4, strategy="active_slots")
    for g in plan.groups:
        g.domain = None
        g.topics = []
        g.keywords = []
        g.curated_path = None
        g.domain_pack = None
        g.description = ""
        g.notes = ""
        g.tags = []

    # structure_only dossier with zero scores (as forensics_for_group produces)
    g0 = plan.groups[0]
    dossier = forensics_for_group(plan, g0.id, affinity=None)
    assert dossier["evidence_tier"] == "structure_only"
    assert max(dossier["content"]["theme_scores"].values()) == 0.0

    applied = auto_bind_sector_from_forensics(
        g0, dossier, global_domain="demo_field"
    )
    # domain slug OK; multi-theme keyword paint forbidden
    assert applied.get("domain") == "demo_field"
    assert "topics" not in applied
    assert "keywords" not in applied
    assert not g0.topics
    assert not g0.keywords

    # Full gate path with global_domain (dry-run style)
    plan2 = create_studio_plan(family="qwen_a3b", num_groups=4)
    for g in plan2.groups:
        g.domain = None
        g.topics = []
        g.keywords = []
        g.description = ""
        g.notes = ""
        g.tags = []
    run_forensics_gate(
        plan2,
        affinity=None,
        mode="warn",
        auto_bind=True,
        global_domain="demo_field",
    )
    for g in plan2.groups:
        # may have domain from auto_bind
        if g.domain:
            assert g.domain == "demo_field"
        # must not have multi-theme keyword soup
        assert len(g.keywords or []) == 0, g.keywords
        assert len(g.topics or []) == 0, g.topics
        d2 = forensics_for_group(plan2, g.id, affinity=None)
        high = [t for t in d2["content"]["top_themes"] if t.get("score", 0) >= 0.5]
        assert len(high) <= 1, (g.name, high)


def test_unbound_geometry_structure_only_no_theme_saturation():
    plan = create_studio_plan(family="qwen_a3b", num_groups=4, strategy="active_slots")
    # ensure fully unbound
    for g in plan.groups:
        g.domain = None
        g.topics = []
        g.keywords = []
        g.curated_path = None
        g.domain_pack = None
        g.description = ""
        g.notes = ""
        g.tags = []

    g0 = plan.groups[0]
    assert resolve_evidence_tier(g0) == "structure_only"
    report = score_themes_for_group(g0, DEFAULT_THEME_BANK, affinity=None)
    assert report["evidence_tier"] == "structure_only"
    assert report["confidence"] == "low"
    # all calibrated scores zero — no multi-theme saturation
    assert max(report["theme_scores"].values()) == 0.0
    # high-confidence content refused
    assert "structure_only" in " ".join(report["notes"]).lower() or report["confidence"] == "low"

    dossier = forensics_for_group(plan, g0.id)
    assert dossier["evidence_tier"] == "structure_only"
    assert dossier["confidence"] == "low"
    scores = list(dossier["content"]["theme_scores"].values())
    assert max(scores) == 0.0
    # must not claim high multi-theme content
    assert not any(
        t.get("score", 0) >= 0.9 for t in dossier["content"]["top_themes"]
    )


def test_assignment_tier_calibrated_peak_not_all_ones():
    plan = create_studio_plan(family="qwen_a3b", num_groups=2)
    g = plan.groups[0]
    g.domain = "logistics"
    g.topics = ["inventory", "warehouse", "lead time"]
    g.keywords = ["sla", "stockout", "carrier", "otif"]
    assert resolve_evidence_tier(g) == "assignment"
    rep = score_themes_for_group(g, DEFAULT_THEME_BANK)
    assert rep["evidence_tier"] == "assignment"
    # logistics should rank high but not every theme ~1.0
    top = rep["top_themes"][0]
    assert top["id"] == "logistics_ops" or top["score"] > 0.15
    high_count = sum(1 for v in rep["theme_scores"].values() if v >= 0.9)
    assert high_count <= 1  # competitive calibration
    # assignment alone cannot claim high confidence
    assert rep["confidence"] in ("medium", "low")


def test_routing_probed_synthetic_not_high_confidence():
    plan = create_studio_plan(family="qwen_a3b", num_groups=2)
    g = plan.groups[0]
    g.domain = "code"
    g.keywords = ["python", "api", "debug"]
    n_l, n_e = plan.capacity.num_layers, plan.capacity.num_experts
    rng = np.random.default_rng(0)
    mat = rng.dirichlet(np.ones(n_e) * 0.7, size=n_l).tolist()
    aff = {
        "affinity": mat,
        "domain": "code",
        "metadata": {"synthetic": True},
        "synthetic": True,
    }
    rep = score_themes_for_group(g, DEFAULT_THEME_BANK, affinity=aff)
    assert rep["evidence_tier"] == "routing_probed"
    assert rep["affinity_synthetic"] is True
    assert rep["confidence"] != "high"  # synthetic cannot claim high


def test_calibrate_structure_only_zeros():
    raw = {tid: 1.0 for tid in list(DEFAULT_THEME_BANK)[:5]}
    cal = calibrate_theme_scores(raw, tier="structure_only")
    assert all(v == 0.0 for v in cal.values())


# ── 2. Plan fingerprint ──────────────────────────────────────────────


def test_plan_fingerprint_changes_on_membership_edit():
    plan = create_studio_plan(family="qwen_a3b", num_groups=3)
    freeze = freeze_plan(plan)
    fp1 = freeze.fingerprint
    assert verify_plan_fingerprint(plan, fp1)["ok"] is True
    # mutate membership
    g = plan.groups[0]
    if g.cells:
        g.cells.pop()
        g.refresh_counts()
    check = verify_plan_fingerprint(plan, fp1)
    assert check["ok"] is False
    assert check["mutated"] is True
    assert plan_fingerprint(plan) != fp1


def test_plan_fingerprint_stable_for_identical_membership():
    plan = create_studio_plan(family="qwen_a3b", num_groups=3, strategy="active_slots")
    a = plan_fingerprint(plan)
    b = plan_fingerprint(plan)
    assert a == b


# ── 3. Data contracts ────────────────────────────────────────────────


def test_all_synth_shard_fails_contract_block():
    records = [
        {
            "text": f"synthetic scaffold case {i} " * 5,
            "synthetic": True,
            "source": "synthetic_self_instruct",
            "meta": {"source": "sector_synth"},
        }
        for i in range(12)
    ]
    # force near-duplicate uniqueness fail too by using same text
    records = [
        {
            "text": "same synthetic blob repeated",
            "synthetic": True,
            "meta": {"source": "sector_synth"},
        }
        for _ in range(12)
    ]
    rep = evaluate_sector_contract(
        records,
        group_id="g1",
        name="thin",
        spec=DataContractSpec(
            min_samples=8,
            min_real_fraction=0.15,
            max_synth_fraction=0.85,
            min_unique_ratio=0.35,
            mode="block",
        ),
    )
    assert rep.passed is False
    assert rep.status == "fail"
    assert rep.train_eligible is False
    assert rep.violations


def test_contract_warn_allows_train_eligible():
    records = [
        {"text": f"unique real-ish example number {i}", "synthetic": True, "meta": {"source": "sector_synth"}}
        for i in range(20)
    ]
    rep = evaluate_sector_contract(
        records,
        group_id="g1",
        name="warn",
        spec=DataContractSpec(mode="warn", min_real_fraction=0.5, max_synth_fraction=0.5),
    )
    assert rep.passed is False
    assert rep.status == "warn"
    assert rep.train_eligible is True


def test_sector_dataset_forge_attaches_contract(tmp_path):
    plan = create_studio_plan(family="qwen_a3b", num_groups=2)
    plan.groups[0].domain = "ops"
    plan.groups[0].keywords = ["process", "failure"]
    forge = SectorDatasetForge(
        DataConfig(domain="demo", seed=1),
        min_samples=4,
        contract_mode="block",
        min_real_fraction=0.9,  # force fail on synth
        max_synth_fraction=0.1,
    )
    # all synth records
    recs = [
        {"text": f"process failure recovery {i}", "synthetic": True, "source": "synthetic_self_instruct"}
        for i in range(30)
    ]
    ds = forge.build(plan, recs, output_dir=tmp_path / "sec")
    assert ds.shards
    # at least one shard has contract field
    assert ds.shards[0].contract is not None
    assert "train_eligible" in ds.shards[0].to_dict()


# ── 4. Probe keep/rollback ───────────────────────────────────────────


def test_keep_rollback_regresses_then_rolls_back():
    from aetherforge.training.sector_probe import SectorProbeResult

    pre = SectorProbeResult("g", "s", "pre", mean_share=0.2, total_share=1.0, n_layers_hit=4)
    post = SectorProbeResult("g", "s", "post", mean_share=0.05, total_share=0.3, n_layers_hit=4)
    d = decide_keep_rollback(pre, post, min_delta=-0.02)
    assert d.keep is False
    assert d.decision == "rollback"


def test_keep_rollback_improve_keeps():
    from aetherforge.training.sector_probe import SectorProbeResult

    pre = SectorProbeResult("g", "s", "pre", mean_share=0.1, total_share=0.5, n_layers_hit=4)
    post = SectorProbeResult("g", "s", "post", mean_share=0.18, total_share=0.9, n_layers_hit=4)
    d = decide_keep_rollback(pre, post, min_delta=-0.02)
    assert d.keep is True
    assert d.decision == "keep"


def test_interference_summary_detects_sibling_regression():
    plan = create_studio_plan(family="qwen_a3b", num_groups=3)
    n_l, n_e = plan.capacity.num_layers, plan.capacity.num_experts
    rng = np.random.default_rng(1)
    pre = rng.dirichlet(np.ones(n_e), size=n_l).tolist()
    # post: drain mass from group 1 cells into group 0
    import copy

    post = copy.deepcopy(pre)
    g0, g1 = plan.groups[0], plan.groups[1]
    keys0 = {(c.layer, c.expert) for c in g0.cells}
    keys1 = {(c.layer, c.expert) for c in g1.cells}
    for li in range(n_l):
        for ei in range(n_e):
            if (li, ei) in keys1:
                take = post[li][ei] * 0.5
                post[li][ei] -= take
                # dump onto first g0 cell in layer if any
                targets = [e for e in range(n_e) if (li, e) in keys0]
                if targets:
                    post[li][targets[0]] += take
        s = sum(post[li]) or 1.0
        post[li] = [x / s for x in post[li]]
    inter = interference_summary(
        plan,
        {"affinity": pre},
        {"affinity": post},
        trained_group_ids=[g0.id],
    )
    assert inter["ok"] is True
    # g1 not trained and lost mass → regressed
    g1_row = next(r for r in inter["sectors"] if r["group_id"] == g1.id)
    assert g1_row["delta_mean_share"] < 0


# ── 5. Sequential dry-run workflow artifacts ──────────────────────────


def test_sequential_workflow_fingerprint_probe_interference(tmp_path):
    plan = create_studio_plan(family="qwen_a3b", num_groups=3)
    plan.groups[0].domain = "ops"
    plan.groups[0].keywords = ["process", "failure", "recovery"]
    plan.groups[1].domain = "analysis"
    plan.groups[1].keywords = ["evidence", "hypothesis"]

    n_l, n_e = plan.capacity.num_layers, plan.capacity.num_experts
    rng = np.random.default_rng(2)
    aff_mat = rng.dirichlet(np.ones(n_e) * 0.8, size=n_l).tolist()
    affinity = {
        "affinity": aff_mat,
        "domain": "demo_field",
        "metadata": {"synthetic": True, "generator": "test"},
        "synthetic": True,
        "num_layers": n_l,
        "num_experts": n_e,
    }

    records = [
        {"text": f"Process failure recovery planning with evidence {i}."}
        for i in range(40)
    ] + [
        {"text": f"Hypothesis ranking under incomplete evidence {i}."}
        for i in range(20)
    ]

    wf = SectorWorkflow(
        plan,
        TrainingConfig(
            max_steps=5,
            sector_min_samples=4,
            sector_contract_mode="warn",
            sector_probe_enabled=True,
            sector_keep_rollback=True,
        ),
        DataConfig(domain="demo_field", seed=7),
        affinity=affinity,
        dry_run=True,
        gate_mode="warn",
        auto_bind=True,
        min_samples=4,
    )
    result = wf.run(records, tmp_path / "wf")
    assert result.plan_freeze is not None
    assert result.plan_freeze["fingerprint"]
    assert (tmp_path / "wf" / "plan_freeze.json").exists()
    assert (tmp_path / "wf" / "plan_fingerprint.txt").exists()
    fp = (tmp_path / "wf" / "plan_fingerprint.txt").read_text().strip()
    assert fp == result.plan_freeze["fingerprint"]
    assert result.n_trained >= 1
    # sectors have evidence tier + probes
    kept = [s for s in result.sectors if s.status in ("dry_run", "trained")]
    assert kept
    assert any(s.evidence_tier for s in kept)
    assert any(s.pre_probe for s in kept)
    assert any(s.keep_rollback for s in kept)
    # interference artifact
    assert result.interference is not None
    assert (tmp_path / "wf" / "interference.json").exists()
    # mutate plan after freeze → verify fails
    plan.groups[0].cells.pop()
    assert verify_plan_fingerprint(plan, fp)["ok"] is False


def test_scorecard_dry_run_not_moe_ready():
    scorer = ReliabilityScorecard(EvalConfig(), domain="demo")
    # synthetic affinity
    n_l, n_e = 4, 8
    routing = np.ones((n_l, n_e))
    aff = AffinityResult(
        domain="demo",
        family="qwen_a3b",
        num_experts=n_e,
        num_layers=n_l,
        routing_freq=routing,
        grad_contrib=np.zeros_like(routing),
        affinity=routing / routing.sum(axis=1, keepdims=True),
        ranked=[(0, 0, 1.0)],
        entropy_per_layer=[1.0] * n_l,
        load_balance_cv=0.5,
        probe_tokens=10,
        metadata={"synthetic": True},
    )
    sc = scorer.evaluate(
        affinity=aff,
        eval_texts=["Assessment: tradeoff analysis with evidence and action plan."] * 8,
        dry_run=True,
        sector_workflow={"n_trained": 3, "n_blocked": 0, "n_skipped": 0, "n_rolled_back": 0},
    )
    d = sc.to_dict()
    assert d["details"]["dry_run"] is True
    assert d["details"]["scorecard_kind"] == "ci_completeness"
    assert d["details"]["moe_ready"] is False
    assert d["details"]["full_moe_promoted_readiness"] is False
    assert "DRY-RUN" in (d["details"]["promotion_label"] or "")
    assert d["details"].get("affinity_synthetic") is True


def test_full_pipeline_dry_run_fidelity_artifacts(tmp_path):
    cfg = load_config(
        ROOT / "configs" / "base.yaml",
        ROOT / "recipes" / "generic_dryrun.yaml",
        overrides={
            "run": {"dry_run": True, "name": "redteam-dry"},
            "training": {
                "output_dir": str(tmp_path / "runs"),
                "sector_mode": "sequential",
                "max_steps": 5,
                "sector_contract_mode": "warn",
            },
            "data": {"synthetic": {"num_samples": 48}},
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
    root = pipe.root
    assert (root / "sector_forensics.json").exists()
    forensics = json.loads((root / "sector_forensics.json").read_text())
    # every sector has evidence_tier
    for s in forensics.get("sectors") or []:
        assert s.get("evidence_tier") in (
            "structure_only",
            "assignment",
            "routing_probed",
        )
    assert (root / "sector_workflow" / "plan_fingerprint.txt").exists()
    assert (root / "AFFINITY_SYNTHETIC.txt").exists()
    assert (root / "PROMOTION_LABEL.txt").exists()
    label = (root / "PROMOTION_LABEL.txt").read_text()
    assert "DRY-RUN" in label or "CI" in label
    sc = json.loads((root / "scorecard.json").read_text())
    assert sc["details"]["moe_ready"] is False
    assert sc["details"]["full_moe_promoted_readiness"] is False
    if result.get("promoted"):
        assert (root / "promoted" / "DRY_RUN_NOT_MOE_READY.txt").exists()
        kind = json.loads((root / "promoted" / "PROMOTION_KIND.json").read_text())
        assert kind["promoted_kind"] == "ci_dry_run"
        assert kind["full_moe_promoted_readiness"] is False
    live = json.loads((root / "live_status.json").read_text())
    assert live.get("sectors", {}).get("items")
    wf = json.loads((root / "sector_workflow" / "sector_workflow.json").read_text())
    assert wf.get("plan_freeze", {}).get("fingerprint")
