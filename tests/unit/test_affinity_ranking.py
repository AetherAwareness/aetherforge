import numpy as np

from aetherforge.affinity.probe import AffinityResult
from aetherforge.affinity.ranking import rank_experts, progressive_tiers
from aetherforge.affinity.expert_selector import ExpertSelector
from aetherforge.utils.config import AffinityConfig


def _fake_result(n_layers=2, n_exp=8) -> AffinityResult:
    rng = np.random.default_rng(0)
    routing = rng.random((n_layers, n_exp))
    aff = routing / routing.sum(axis=1, keepdims=True)
    ranked = []
    for li in range(n_layers):
        for ei in range(n_exp):
            ranked.append((li, ei, float(aff[li, ei])))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return AffinityResult(
        domain="demo_field",
        family="qwen_a3b",
        num_experts=n_exp,
        num_layers=n_layers,
        routing_freq=routing,
        grad_contrib=np.zeros_like(routing),
        affinity=aff,
        ranked=ranked,
        entropy_per_layer=[1.0, 1.1],
        load_balance_cv=0.5,
        probe_tokens=100,
    )


def test_rank_top_k():
    r = _fake_result()
    top = rank_experts(r, top_k=4)
    assert len(top) == 4
    assert top[0][2] >= top[-1][2]


def test_progressive_tiers():
    r = _fake_result()
    tiers = progressive_tiers(r, [2, 4])
    assert len(tiers) == 2
    assert len(tiers[0]) == 2
    assert len(tiers[1]) == 4


def test_selector():
    r = _fake_result()
    plan = ExpertSelector(AffinityConfig(top_k_experts=3)).select(r)
    assert len(plan.selected) == 3
    assert plan.domain == "demo_field"
