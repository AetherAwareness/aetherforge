from aetherforge.data.quality_gates import QualityGateRunner
from aetherforge.data.trajectory_hive import TrajectoryHive
from aetherforge.eval.scorecard import ReliabilityScorecard
from aetherforge.eval.gates import apply_gates
from aetherforge.utils.config import EvalConfig, QualityGatesConfig, ScorecardThresholds


def test_quality_gates_dedupe():
    runner = QualityGateRunner(QualityGatesConfig(min_length=5, min_diversity=0.0))
    recs = [
        {"text": "This is a unique logistics case about inventory lead-time under port congestion."},
        {"text": "This is a unique logistics case about inventory lead-time under port congestion."},
        {"text": "A totally different energy note regarding nodal price spikes and imbalance."},
    ]
    kept, report = runner.filter_records(recs)
    assert report.total_out == 2
    assert report.dropped["dedupe"] == 1
    assert len(kept) == 2


def test_thd_pairs():
    hive = TrajectoryHive(specialists=["alpha", "beta"], seed=1)
    traj, pairs = hive.generate(["case one", "case two"], "demo_field")
    assert len(traj) == 2
    assert len(pairs) == 2
    assert "chosen" in pairs[0] and "rejected" in pairs[0]


def test_gates_pass_fail():
    thr = ScorecardThresholds(
        domain_score=0.8, routing_entropy_min=1.0, load_balance_cv_max=2.0
    )
    ok = apply_gates(
        {
            "domain_score": 0.9,
            "routing_entropy": 1.5,
            "load_balance_cv": 0.4,
            "hallucination_rate": 0.01,
            "safety_pass": 1.0,
        },
        thr,
        baseline={"general_score": 0.8},
    )
    assert ok.passed
    bad = apply_gates(
        {
            "domain_score": 0.1,
            "routing_entropy": 0.1,
            "load_balance_cv": 5.0,
            "hallucination_rate": 0.5,
            "safety_pass": 1.0,
        },
        thr,
        baseline={"general_score": 0.8},
    )
    assert not bad.passed
    assert bad.failures


def test_scorecard_dry_generic_domain():
    sc = ReliabilityScorecard(
        EvalConfig(),
        domain="demo_field",
        domain_keywords=["demo", "field", "decision"],
    ).evaluate(dry_run=True)
    assert "domain_score" in sc.metrics
    assert sc.to_dict()["domain"] == "demo_field"
    # no medical axes
    assert "medical_score" not in sc.metrics


def test_domain_depth_not_medical():
    thr = ScorecardThresholds(domain_score=0.1, domain_depth_min=0.5, routing_entropy_min=0.0)
    sc = ReliabilityScorecard(EvalConfig(scorecard_thresholds=thr), domain="law").evaluate(
        eval_texts=[
            "Assessment frame: contract risk.\n1. Read clause.\n2. Measure exposure.\n"
            "Risks: overclaiming. confidence may vary."
        ],
        domain_keywords=["contract", "risk", "clause"],
        dry_run=False,
    )
    assert "domain_depth" in sc.metrics
