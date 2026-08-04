from aetherforge.data.domain_pack import DomainPack
from aetherforge.data.quality_gates import QualityGateRunner
from aetherforge.data.synthetic import generate_self_instruct
from aetherforge.utils.config import QualityGatesConfig, SyntheticConfig


def test_synthetic_diversity_not_collapsed():
    pack = DomainPack(
        domain="demo_field",
        topics=[
            "prioritization under conflict",
            "root-cause of process failure",
            "confidence calibration",
            "tradeoff under constraints",
            "failure-mode recovery",
        ],
        keywords=["prioritization", "root-cause", "tradeoff", "failure"],
        actions=[
            "State assumptions.",
            "List hypotheses.",
            "Define next measurement.",
            "Document stop conditions.",
        ],
    )
    recs = generate_self_instruct(
        pack.domain,
        SyntheticConfig(enabled=True, num_samples=80, trajectory_hive=False),
        seed=7,
        pack=pack,
    )
    runner = QualityGateRunner(QualityGatesConfig(min_length=32, min_diversity=0.35))
    kept, report = runner.filter_records(recs)
    assert len(kept) >= 70
    assert report.diversity >= 0.20
    assert report.diversity_components["unique_docs"] >= 0.5
    assert report.diversity_components["field_entropy"] >= 0.0
    assert report.passed


def test_dedupe_near_duplicates():
    runner = QualityGateRunner(QualityGatesConfig(min_length=10, min_diversity=0.0))
    base = "Inventory lead-time shock requires safety-stock recalculation for critical SKUs carefully."
    recs = [
        {"text": base},
        {"text": base + "  "},
        {"text": "Completely different energy-market note about nodal congestion rents."},
    ]
    kept, report = runner.filter_records(recs)
    assert report.dropped["dedupe"] >= 1
    assert report.total_out == 2
