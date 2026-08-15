"""
Domain packs — the ONLY place industry content enters AetherForge.

A pack is YAML/JSON describing any field (logistics, law, energy, retail, …):

  domain: supply_chain
  description: Cross-border logistics and inventory risk
  topics:
    - multi-echelon inventory under lead-time shock
    - port congestion rerouting
  keywords:
    - inventory
    - lead time
    - logistics
  actions:
    - Map single points of failure in the tier-2 supplier graph.
    - Quantify stockout cost vs expedite cost.
  specialists:
    - logistics
    - procurement
  populations:           # optional framing axes (audiences / contexts)
    - enterprise ops
    - SMB
  contexts:
    - quarterly planning
    - incident response
  angles:
    - cost
    - resilience
    - compliance

If no pack is provided, the trainer uses domain slug tokens + generic scaffolds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from aetherforge.utils.config import DataConfig
from aetherforge.utils.logging import get_logger

log = get_logger("data.domain_pack")


class PackBenchmark(BaseModel):
    """One pack-owned eval case. Industry content lives here, never in trainer code."""

    id: str
    prompt: str
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    gold: Optional[str] = None
    weight: float = 1.0


class DomainPack(BaseModel):
    domain: str = "general"
    description: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    specialists: list[str] = Field(default_factory=list)
    populations: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    angles: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    high_stakes: bool = False
    benchmarks: list[PackBenchmark] = Field(default_factory=list)

    def merge_from_data_config(self, data: DataConfig) -> "DomainPack":
        """Inline DataConfig fields override pack fields when non-empty."""
        benches = list(self.benchmarks)
        extra = getattr(data, "benchmarks", None) or []
        if extra:
            benches = [
                b if isinstance(b, PackBenchmark) else PackBenchmark.model_validate(b)
                for b in extra
            ]
        return DomainPack(
            domain=data.domain or self.domain,
            description=data.description or self.description,
            topics=list(data.topics) if data.topics else list(self.topics),
            keywords=list(data.keywords) if data.keywords else list(self.keywords),
            actions=list(data.actions) if data.actions else list(self.actions),
            specialists=list(self.specialists),
            populations=list(self.populations),
            contexts=list(self.contexts),
            angles=list(self.angles),
            hints=list(self.hints),
            high_stakes=self.high_stakes,
            benchmarks=benches,
        )


_GENERIC_TOPICS = [
    "multi-step reasoning under uncertainty",
    "prioritization under conflicting goals",
    "root-cause analysis of process failure",
    "calibration of confidence and evidence",
    "tradeoff analysis with incomplete data",
    "detecting contradictions across sources",
    "structured decision logs for auditability",
    "failure-mode anticipation and recovery",
]

_GENERIC_ACTIONS = [
    "State assumptions and the decision owner explicitly.",
    "List the top competing hypotheses with discriminators.",
    "Define a reversible next experiment or measurement.",
    "Identify second-order risks if the plan succeeds.",
    "Document stop conditions and escalation triggers.",
    "Separate signal from anecdote in the evidence table.",
]

_GENERIC_POPULATIONS = [
    "operators",
    "analysts",
    "executives",
    "frontline staff",
    "cross-functional teams",
    "regulated environments",
    "resource-constrained teams",
]

_GENERIC_CONTEXTS = [
    "planning cycle",
    "incident response",
    "post-mortem",
    "onboarding a junior colleague",
    "stakeholder review",
    "time-critical decision",
    "long-horizon strategy",
]

_GENERIC_ANGLES = [
    "efficiency",
    "resilience",
    "risk",
    "quality",
    "cost",
    "compliance",
    "speed",
    "explainability",
]

_GENERIC_HINTS = [
    "prioritize reversible steps",
    "prefer evidence-backed claims",
    "include quantitative thresholds where possible",
    "note when to escalate",
    "state confidence and residual uncertainty",
    "avoid overclaiming",
    "consider resource constraints",
    "surface conflicting incentives",
]


def load_domain_pack(path: str | Path) -> DomainPack:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Domain pack not found: {path}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(raw) or {}
    else:
        import json

        data = json.loads(raw)
    # allow packs nested under `domain_pack:` or flat
    if isinstance(data, dict) and "domain_pack" in data and isinstance(data["domain_pack"], dict):
        data = data["domain_pack"]
    pack = DomainPack.model_validate(data)
    log.info(
        "Loaded domain pack domain=%s topics=%d keywords=%d from %s",
        pack.domain,
        len(pack.topics),
        len(pack.keywords),
        path,
    )
    return pack


def resolve_domain_pack(data: DataConfig) -> DomainPack:
    """
    Build the effective domain pack for a run.

    Priority: domain_pack file → inline DataConfig fields → generic defaults.
    """
    base = DomainPack(domain=data.domain)
    if data.domain_pack:
        base = load_domain_pack(data.domain_pack)
    pack = base.merge_from_data_config(data)

    # Fill gaps with industry-agnostic scaffolds (never industry-specific)
    if not pack.topics:
        slug = pack.domain.replace("_", " ").replace("-", " ").strip() or "general"
        pack.topics = [f"{slug}: {t}" for t in _GENERIC_TOPICS]
    if not pack.keywords:
        # derive from domain slug + topics
        tokens = set(pack.domain.replace("-", "_").split("_"))
        for t in pack.topics[:12]:
            for w in t.lower().split():
                if len(w) > 3:
                    tokens.add(w.strip(".,:;()"))
        pack.keywords = sorted(tokens)[:48]
    if not pack.actions:
        pack.actions = list(_GENERIC_ACTIONS)
    if not pack.populations:
        pack.populations = list(_GENERIC_POPULATIONS)
    if not pack.contexts:
        pack.contexts = list(_GENERIC_CONTEXTS)
    if not pack.angles:
        pack.angles = list(_GENERIC_ANGLES)
    if not pack.hints:
        pack.hints = list(_GENERIC_HINTS)
    if not pack.specialists:
        pack.specialists = [pack.domain, f"{pack.domain}_reviewer", "generalist"]

    return pack


def pack_to_dict(pack: DomainPack) -> dict[str, Any]:
    return pack.model_dump()
