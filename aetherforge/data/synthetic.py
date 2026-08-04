"""
Industry-agnostic synthetic data generation.

All field-specific content comes from a DomainPack (config / YAML).
The trainer never hard-codes medicine, finance, or any other industry.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from aetherforge.data.domain_pack import (
    DomainPack,
    resolve_domain_pack,
    _GENERIC_ACTIONS,
    _GENERIC_ANGLES,
    _GENERIC_CONTEXTS,
    _GENERIC_HINTS,
    _GENERIC_POPULATIONS,
)
from aetherforge.utils.config import DataConfig, SyntheticConfig
from aetherforge.utils.logging import get_logger

log = get_logger("data.synthetic")

# Domain-agnostic instructional shells — {topic} filled from the pack
_TEMPLATES = [
    "Explain {topic} clearly for a specialist audience, including edge cases and {pop}.",
    "Scenario: {topic}\nContext: {context}\nProvide a structured analysis with decision criteria.",
    "What are the key decision points when dealing with {topic}? List tradeoffs for {pop}.",
    "Critique the following approach to {topic} and propose improvements: {hint}",
    "Summarize best-practice patterns related to {topic}. Focus on {angle}.",
    "A junior colleague asks about {topic}. Teach them step by step. Audience: {pop}.",
    "Identify failure modes and risks for {topic} in the setting of {context}.",
    "Compare two strategies for {topic} ({strategy_a} vs {strategy_b}) and recommend one with justification.",
    "Write a brief case involving {topic}, then give the correct next action and why.",
    "List red flags that should change management when evaluating {topic}.",
    "How would you monitor response after intervening on {topic}? Include thresholds.",
    "Translate specialist reasoning about {topic} into a checklist a generalist can use.",
]

_STRATEGIES = [
    "watchful waiting",
    "early intervention",
    "step-up approach",
    "escalate to specialist",
    "pilot then scale",
    "automate the workflow",
    "manual deep review",
    "defer and gather data",
]


def generate_self_instruct(
    domain: str,
    config: SyntheticConfig,
    seed: int = 42,
    topics: Optional[list[str]] = None,
    pack: Optional[DomainPack] = None,
    data_cfg: Optional[DataConfig] = None,
) -> list[dict[str, Any]]:
    """
    Generate synthetic instruction/response pairs for any industry.

    Prefer passing `pack` or `data_cfg` so topics/actions/keywords come from
    the domain pack rather than hard-coded field knowledge.
    """
    if pack is None:
        if data_cfg is not None:
            pack = resolve_domain_pack(data_cfg)
        else:
            pack = resolve_domain_pack(
                DataConfig(domain=domain, topics=list(topics or []))
            )

    rng = random.Random(seed)
    topic_list = list(topics or pack.topics)
    if config.num_samples > len(topic_list):
        extras = []
        for t in topic_list:
            extras.append(f"{t} (atypical case)")
            extras.append(f"recurrent or refractory: {t}")
        topic_list = topic_list + extras

    n = config.num_samples if config.enabled else 0
    actions = list(pack.actions) or list(_GENERIC_ACTIONS)
    pops = list(pack.populations) or list(_GENERIC_POPULATIONS)
    contexts = list(pack.contexts) or list(_GENERIC_CONTEXTS)
    angles = list(pack.angles) or list(_GENERIC_ANGLES)
    hints = list(pack.hints) or list(_GENERIC_HINTS)
    records: list[dict[str, Any]] = []

    def _pick_actions(k: int = 3) -> list[str]:
        pool = actions if len(actions) >= k else (actions * ((k // max(len(actions), 1)) + 1))
        return rng.sample(pool, k)

    for i in range(n):
        topic = topic_list[i % len(topic_list)]
        tmpl = _TEMPLATES[i % len(_TEMPLATES)]
        hint = rng.choice(hints)
        pop = rng.choice(pops)
        angle = rng.choice(angles)
        context = rng.choice(contexts)
        sa, sb = rng.sample(_STRATEGIES, 2)
        prompt = tmpl.format(
            topic=topic,
            hint=hint,
            pop=pop,
            angle=angle,
            context=context,
            strategy_a=sa,
            strategy_b=sb,
        )
        act1, act2, act3 = _pick_actions(3)
        conf = 0.55 + rng.random() * 0.4
        pitfall = rng.choice(
            [
                "anchoring on the first signal",
                "underweighting base rates",
                "protocol deviation under time pressure",
                "missing second-order interactions",
                "overfitting to rare edge cases",
                "confusing correlation with mechanism",
            ]
        )
        completion = (
            f"Domain: {pack.domain} | Focus: {topic}\n"
            f"Setting: {context} | Audience: {pop}\n\n"
            f"Assessment frame ({angle}):\n"
            f"- Leading consideration: {topic}\n"
            f"- Competing possibilities should be listed with discriminators.\n"
            f"- Constraint applied: {hint}.\n\n"
            f"Action plan:\n"
            f"1. {act1}\n"
            f"2. {act2}\n"
            f"3. {act3}\n\n"
            f"Risks / failure modes:\n"
            f"- Primary pitfall: {pitfall}.\n"
            f"- Escalate when red flags appear or trajectory worsens despite the plan.\n\n"
            f"Calibration: confidence ≈ {conf:.2f}; revise if new data contradicts the frame.\n"
            f"Note: synthetic scaffold — replace with teacher-model generations for production runs."
        )
        records.append(
            {
                "text": f"### Instruction\n{prompt}\n\n### Response\n{completion}",
                "prompt": prompt,
                "completion": completion,
                "domain": pack.domain,
                "topic": topic,
                "source": "synthetic_self_instruct",
                "synthetic": True,
                "meta": {
                    "context": context,
                    "population": pop,
                    "angle": angle,
                    "hint": hint,
                },
            }
        )

    log.info(
        "Generated %d synthetic samples for domain=%s (pack topics=%d)",
        len(records),
        pack.domain,
        len(pack.topics),
    )
    return records
