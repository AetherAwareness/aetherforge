from aetherforge.data.domain_pack import DomainPack, PackBenchmark, resolve_domain_pack
from aetherforge.eval.pack_eval import evaluate_pack, score_answer
from aetherforge.eval.scorecard import ReliabilityScorecard
from aetherforge.utils.config import DataConfig, EvalConfig, ScorecardThresholds


def test_score_answer_include_and_forbid():
    b = PackBenchmark(
        id="t1",
        prompt="Decide with a stop rule.",
        must_include=["assumption", "stop"],
        must_not_include=["guaranteed", "infallible"],
    )
    good = score_answer(
        b,
        "State the assumption. Stop if evidence flips. Reversible next step.",
    )
    assert good["score"] >= 0.6
    assert "assumption" in good["hits"]
    bad = score_answer(b, "This is guaranteed and infallible with zero thinking.")
    assert bad["forbidden_hits"]
    assert bad["score"] < good["score"]


def test_evaluate_pack_from_eval_texts():
    pack = DomainPack(
        domain="demo_field",
        keywords=["evidence", "tradeoff"],
        benchmarks=[
            PackBenchmark(
                id="case_a",
                prompt="Conflicting goals. State a reversible next step.",
                must_include=["reversible", "assumption"],
            )
        ],
    )
    report = evaluate_pack(
        pack,
        eval_texts=[
            "Assumption: budget is fixed. Reversible next step: trial one lane.",
            "Unrelated weather note.",
        ],
        dry_run=True,
    )
    assert report.n == 1
    assert report.n_scored == 1
    assert report.score > 0.5
    assert report.mode in ("dry_run_proxy", "eval_text_proxy")
    assert report.to_dict()["schema"] == "aetherforge.pack_eval.v1"


def test_evaluate_pack_no_benchmarks():
    pack = DomainPack(domain="empty")
    report = evaluate_pack(pack, dry_run=True)
    assert report.n == 0
    assert report.mode == "no_benchmarks"


def test_evaluate_pack_uses_pack_proxy_when_no_texts():
    pack = DomainPack(
        domain="demo_field",
        actions=["State assumptions and a reversible next step. Stop if evidence flips."],
        benchmarks=[
            PackBenchmark(
                id="case_a",
                prompt="Conflicting goals.",
                must_include=["assumption", "reversible", "stop"],
            )
        ],
    )
    report = evaluate_pack(pack, dry_run=True)
    assert report.n_scored == 1
    assert report.score > 0.5
    assert report.items[0]["source"] == "eval_text_proxy"


def test_scorecard_includes_pack_eval():
    thr = ScorecardThresholds(
        domain_score=0.1,
        routing_entropy_min=0.0,
        pack_eval_min=0.4,
    )
    sc = ReliabilityScorecard(
        EvalConfig(scorecard_thresholds=thr),
        domain="demo_field",
    ).evaluate(
        eval_texts=["Assessment: tradeoff. Action plan. Risks may vary."],
        domain_keywords=["tradeoff"],
        pack_eval={"schema": "aetherforge.pack_eval.v1", "score": 0.8, "n": 2, "mode": "dry_run_proxy"},
        dry_run=True,
    )
    assert sc.metrics["pack_eval_score"] == 0.8
    assert sc.details["pack_eval"]["n"] == 2


def test_resolve_inline_benchmarks():
    pack = resolve_domain_pack(
        DataConfig(
            domain="retail_pricing",
            topics=["promo elasticity"],
            keywords=["promo"],
            benchmarks=[
                {
                    "id": "promo",
                    "prompt": "Reprice under stockout.",
                    "must_include": ["promo", "stockout"],
                }
            ],
        )
    )
    assert len(pack.benchmarks) == 1
    assert pack.benchmarks[0].id == "promo"
