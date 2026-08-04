"""
Sector Forensics — reverse-engineer what each MoE expert group *contains*.

Sparse MoEs (Flash ~13B active fire, A3B ~3B active fire) hide knowledge across
thousands of expert slots. Before editing a sector efficiently you need to know:

  • How heavy is this sector vs one routing fire?
  • Which layers / expert indices does it own?
  • What *kinds* of content route into it (theme probes)?
  • How distinct is it from sibling sectors?
  • What should you train / freeze / merge / rebind data to?

Works offline (structure + assignment forensics) and online (affinity matrix /
routing probe scores). Does not download weights; uses GroupPlan + optional
AffinityResult-like dicts.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from aetherforge.groups.capacity import estimate_group_capacity
from aetherforge.groups.models import ExpertGroup, GroupPlan
from aetherforge.utils.logging import get_logger

log = get_logger("groups.forensics")

# ── Theme probe bank (industry-agnostic; labels sectors, not products) ──────
# Each theme is a small bag of natural-language probes used when routing
# affinity is unavailable — we still score sectors by *assigned* domain/keywords
# and by structural heuristics. When affinity matrices exist, themes map onto
# probe-domain scores if the affinity.domain or metadata.themes match.

DEFAULT_THEME_BANK: dict[str, dict[str, Any]] = {
    "code_software": {
        "label": "Code / software engineering",
        "keywords": [
            "python", "function", "api", "debug", "compiler", "git", "algorithm",
            "typescript", "refactor", "unit test", "stack trace",
        ],
        "probes": [
            "Write a Python function that merges two sorted lists.",
            "Explain this stack trace and fix the null pointer.",
            "Design a REST API for inventory with idempotent POST.",
        ],
    },
    "math_reasoning": {
        "label": "Math / formal reasoning",
        "keywords": [
            "prove", "integral", "equation", "probability", "theorem", "derivative",
            "matrix", "combinatorics", "optimize", "constraint",
        ],
        "probes": [
            "Solve for x: 3x^2 - 12x + 9 = 0 and show steps.",
            "What is the expected value of a fair six-sided die?",
            "Prove that the sum of the first n odds is n squared.",
        ],
    },
    "science_tech": {
        "label": "Science & technical explanation",
        "keywords": [
            "physics", "chemistry", "biology", "experiment", "hypothesis",
            "quantum", "circuit", "molecule", "dataset", "measurement",
        ],
        "probes": [
            "Explain photosynthesis at a college level with equations where useful.",
            "How does a MOSFET switch a digital signal?",
            "Describe CRISPR-Cas9 editing in plain language.",
        ],
    },
    "medicine_clinical": {
        "label": "Clinical / biomedical (generic)",
        "keywords": [
            "diagnosis", "patient", "symptom", "dose", "clinical", "pathology",
            "guideline", "contraindication", "lab", "imaging",
        ],
        "probes": [
            "Differential diagnosis for acute chest pain in adults.",
            "Explain mechanism of action of beta blockers.",
            "When is imaging indicated for first seizure?",
        ],
    },
    "law_policy": {
        "label": "Law / policy / compliance",
        "keywords": [
            "statute", "contract", "liability", "regulation", "court", "privacy",
            "compliance", "jurisdiction", "clause", "gdpr",
        ],
        "probes": [
            "Summarize key elements of a valid contract.",
            "What is the difference between negligence and strict liability?",
            "Outline a data-retention policy for customer logs.",
        ],
    },
    "finance_markets": {
        "label": "Finance / markets / accounting",
        "keywords": [
            "portfolio", "valuation", "ebitda", "risk", "hedge", "balance sheet",
            "interest", "option", "liquidity", "audit",
        ],
        "probes": [
            "Explain discounted cash flow valuation step by step.",
            "How does duration measure interest-rate risk?",
            "Difference between cash flow and free cash flow.",
        ],
    },
    "logistics_ops": {
        "label": "Logistics / operations / supply chain",
        "keywords": [
            "inventory", "warehouse", "lead time", "sla", "carrier", "stockout",
            "routing", "fulfillment", "otif", "safety stock",
        ],
        "probes": [
            "How should safety stock change when lead time doubles?",
            "Plan a mode shift from ocean to air under port congestion.",
            "Tradeoffs between stockout cost and expedite freight.",
        ],
    },
    "creative_writing": {
        "label": "Creative writing / narrative",
        "keywords": [
            "story", "character", "scene", "poem", "dialogue", "plot", "tone",
            "metaphor", "chapter", "voice",
        ],
        "probes": [
            "Write a short scene about a lighthouse keeper in winter.",
            "Rewrite this paragraph in a noir detective voice.",
            "Outline a three-act structure for a heist story.",
        ],
    },
    "chat_social": {
        "label": "Conversational / social chat",
        "keywords": [
            "hello", "thanks", "how are you", "chat", "friendly", "empathy",
            "small talk", "encourage", "sorry", "help me",
        ],
        "probes": [
            "Hey, rough day — can we just talk for a minute?",
            "Thanks for the help earlier, what should I do next?",
            "I'm nervous about a presentation tomorrow.",
        ],
    },
    "tools_agents": {
        "label": "Tool use / agents / structured actions",
        "keywords": [
            "tool", "function call", "browser", "search", "json", "workflow",
            "agent", "command", "api call", "plan",
        ],
        "probes": [
            "Call a weather tool then summarize the forecast in JSON.",
            "Plan multi-step research: search, fetch, then write a brief.",
            "When a tool fails, explain the error and retry strategy.",
        ],
    },
    "multilingual": {
        "label": "Multilingual / translation",
        "keywords": [
            "translate", "spanish", "french", "chinese", "japanese", "arabic",
            "locale", "bilingual", "idioma", "翻译",
        ],
        "probes": [
            "Translate this paragraph into Spanish and keep formal tone.",
            "Explain this English idiom to a Japanese learner.",
            "Detect language and reply in the same language.",
        ],
    },
    "factual_world": {
        "label": "Factual world knowledge / encyclopedic",
        "keywords": [
            "who was", "capital of", "when did", "history", "geography",
            "biography", "timeline", "founded", "population", "war",
        ],
        "probes": [
            "When was the Treaty of Westphalia signed and why did it matter?",
            "What is the capital of Kazakhstan and recent renaming notes?",
            "Summarize the Apollo 11 mission timeline.",
        ],
    },
    "product_ops": {
        "label": "Product / SaaS ops / onboarding",
        "keywords": [
            "setup", "api key", "subscription", "onboarding", "workspace",
            "billing", "edition", "dashboard", "instance", "readiness",
        ],
        "probes": [
            "Walk a new user through connecting an API key safely.",
            "Diagnose why the brain endpoint returns 401.",
            "Explain edition differences without inventing features.",
        ],
    },
}


# Layer-depth structural roles (heuristic; MoE often specializes by depth)
def _layer_role(layer: int, n_layers: int) -> str:
    if n_layers <= 0:
        return "unknown"
    t = layer / max(n_layers - 1, 1)
    if t < 0.25:
        return "early_embed_syntax"
    if t < 0.5:
        return "mid_feature"
    if t < 0.75:
        return "mid_late_composition"
    return "late_decision_output"


def _layer_role_label(role: str) -> str:
    return {
        "early_embed_syntax": "Early layers — local syntax / token features",
        "mid_feature": "Mid layers — features & mid-level concepts",
        "mid_late_composition": "Mid-late layers — composition / multi-hop",
        "late_decision_output": "Late layers — decision / output style",
        "full_stack": "Full stack — vertical strip across early→late layers",
        "unknown": "Unknown depth band",
    }.get(role, role)


@dataclass
class SectorForensics:
    """Forensic dossier for one expert sector (group)."""

    group_id: str
    name: str
    family: str
    model_name: str
    # Mass vs active fire
    n_cells: int
    est_params_b: float
    active_fire_ratio: float
    active_params_b_model: float
    # Structure
    layer_span: Optional[list[int]]
    layer_histogram: dict[str, int]  # role -> count
    layer_role_primary: str
    expert_id_span: Optional[list[int]]
    top_cells: list[dict[str, Any]]
    # Content identity
    assigned_domain: Optional[str]
    assigned_topics: list[str]
    assigned_keywords: list[str]
    theme_scores: dict[str, float]  # theme_id -> 0..1
    top_themes: list[dict[str, Any]]
    content_summary: str
    # Distinctiveness
    uniqueness: float  # 0..1 vs siblings
    shared_with: list[dict[str, Any]]  # overlapping mass with other groups
    # Edit guidance
    edit_recommendations: list[str]
    train_flags: dict[str, bool]
    confidence: str  # high | medium | low
    evidence: list[str]
    evidence_tier: str = "structure_only"  # structure_only | assignment | routing_probed
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "family": self.family,
            "model_name": self.model_name,
            "evidence_tier": self.evidence_tier,
            "mass": {
                "n_cells": self.n_cells,
                "est_params_b": self.est_params_b,
                "active_fire_ratio": self.active_fire_ratio,
                "model_active_params_b": self.active_params_b_model,
                "fires_equivalent": round(self.active_fire_ratio, 3),
                "human": (
                    f"~{self.est_params_b:.2f}B expert mass "
                    f"({self.active_fire_ratio:.2f}× one active fire of "
                    f"~{self.active_params_b_model}B)"
                ),
            },
            "structure": {
                "layer_span": self.layer_span,
                "layer_histogram": self.layer_histogram,
                "layer_role_primary": self.layer_role_primary,
                "layer_role_label": _layer_role_label(self.layer_role_primary),
                "expert_id_span": self.expert_id_span,
                "top_cells": self.top_cells,
            },
            "content": {
                "assigned_domain": self.assigned_domain,
                "assigned_topics": self.assigned_topics,
                "assigned_keywords": self.assigned_keywords,
                "theme_scores": self.theme_scores,
                "top_themes": self.top_themes,
                "summary": self.content_summary,
                "evidence_tier": self.evidence_tier,
            },
            "distinctiveness": {
                "uniqueness": self.uniqueness,
                "shared_with": self.shared_with,
            },
            "edit_recommendations": self.edit_recommendations,
            "train_flags": self.train_flags,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "raw": self.raw,
        }


@dataclass
class ModelForensicsReport:
    """Full-lattice forensic map for a MoE checkpoint / plan."""

    model_name: str
    family: str
    capacity: dict[str, Any]
    n_groups: int
    sectors: list[SectorForensics]
    unassigned_cells: int
    inventory_table: list[dict[str, Any]]
    narrative: str
    method: str
    generated_at: float = field(default_factory=time.time)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aetherforge.forensics.v1",
            "model_name": self.model_name,
            "family": self.family,
            "capacity": self.capacity,
            "n_groups": self.n_groups,
            "unassigned_cells": self.unassigned_cells,
            "method": self.method,
            "generated_at": self.generated_at,
            "warnings": self.warnings,
            "narrative": self.narrative,
            "inventory_table": self.inventory_table,
            "sectors": [s.to_dict() for s in self.sectors],
        }


def _token_set(text: str) -> set[str]:
    from aetherforge.groups.evidence import token_set

    return token_set(text)


def _score_text_against_theme(text: str, theme: dict[str, Any]) -> float:
    """Legacy raw hit-rate helper (single theme). Prefer score_themes_for_group."""
    from aetherforge.groups.evidence import raw_theme_hits

    hits = raw_theme_hits(text, {"_t": theme})
    return float(hits.get("_t", 0.0))


def _affinity_mean_for_cells(
    cells: Iterable[tuple[int, int]],
    affinity: Optional[list[list[float]]],
) -> float:
    if not affinity:
        return 0.0
    vals = []
    for li, ei in cells:
        try:
            vals.append(float(affinity[li][ei]))
        except (IndexError, TypeError, ValueError):
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


def _theme_scores_for_group(
    group: ExpertGroup,
    *,
    theme_bank: dict[str, dict[str, Any]],
    affinity_domain: str = "",
    affinity_matrix: Optional[list[list[float]]] = None,
    domain_theme_map: Optional[dict[str, float]] = None,
    affinity: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    """
    Calibrated theme scores 0..1 (competitive; structure_only → all zero).

    Prefer full report via score_themes_for_group for tier + confidence.
    """
    from aetherforge.groups.evidence import score_themes_for_group

    aff = dict(affinity or {})
    if affinity_domain and "domain" not in aff:
        aff["domain"] = affinity_domain
    if affinity_matrix is not None and "affinity" not in aff:
        aff["affinity"] = affinity_matrix
    report = score_themes_for_group(
        group,
        theme_bank,
        affinity=aff or None,
        domain_theme_map=domain_theme_map,
    )
    return report["theme_scores"]


def _structural_profile(group: ExpertGroup, n_layers: int) -> dict[str, Any]:
    layers = [c.layer for c in group.cells]
    experts = [c.expert for c in group.cells]
    hist: Counter[str] = Counter()
    for li in layers:
        hist[_layer_role(li, n_layers)] += 1
    n_unique = len(set(layers))
    # Full vertical strips (common with active_slots) span most of the stack
    if n_layers > 0 and n_unique >= max(3, int(0.6 * n_layers)):
        primary = "full_stack"
        hist["full_stack"] = hist.get("full_stack", 0) + n_unique
    else:
        primary = hist.most_common(1)[0][0] if hist else "unknown"
    return {
        "layer_span": [min(layers), max(layers)] if layers else None,
        "expert_id_span": [min(experts), max(experts)] if experts else None,
        "layer_histogram": dict(hist),
        "layer_role_primary": primary,
        "n_unique_layers": n_unique,
        "n_unique_expert_ids": len(set(experts)),
    }


def _overlap_mass(a: ExpertGroup, b: ExpertGroup) -> int:
    return len(a.cell_keys() & b.cell_keys())


def _uniqueness(group: ExpertGroup, others: list[ExpertGroup]) -> float:
    if not group.cells:
        return 0.0
    own = group.cell_keys()
    shared = 0
    for o in others:
        if o.id == group.id:
            continue
        shared += len(own & o.cell_keys())
    # uniqueness high when little sharing (usually partitions are disjoint)
    return max(0.0, 1.0 - shared / max(len(own), 1))


def _edit_recommendations(
    group: ExpertGroup,
    *,
    mass_ratio: float,
    top_themes: list[dict[str, Any]],
    structural: dict[str, Any],
    uniqueness: float,
    family: str,
) -> list[str]:
    recs: list[str] = []
    fire = "active fire"
    if "a3b" in family or mass_ratio and mass_ratio:
        pass
    if mass_ratio > 1.4:
        recs.append(
            f"Sector is {mass_ratio:.2f}× one {fire} — consider splitting "
            "before training so updates stay localized."
        )
    elif mass_ratio < 0.35 and group.train:
        recs.append(
            f"Sector is only {mass_ratio:.2f}× one {fire} — may under-express; "
            "merge with a sibling or add cells if you want a full fire budget."
        )

    if not group.domain and not group.topics and not group.keywords:
        recs.append(
            "No domain/topics/keywords bound — bind a corpus or pack before ESFT "
            "so this sector has a clear content identity."
        )
    elif group.domain and group.train:
        recs.append(
            f"Train with data matching domain «{group.domain}» only; keep other "
            "sectors frozen to avoid cross-talk."
        )

    if top_themes:
        t0 = top_themes[0]
        if t0.get("score", 0) >= 0.35:
            recs.append(
                f"Primary content signature: {t0['label']} "
                f"(score {t0['score']:.2f}) — prioritize examples in that theme."
            )
        elif t0.get("score", 0) < 0.2:
            recs.append(
                "Weak content signature — run multi-theme affinity probes on a "
                "loaded model to learn what routes here before heavy edits."
            )

    role = structural.get("layer_role_primary")
    if role == "full_stack":
        recs.append(
            "Full-depth strip — edits touch early syntax through late style; "
            "prefer domain-bound LoRA and keep sibling sectors frozen."
        )
    elif role == "early_embed_syntax":
        recs.append(
            "Mostly early layers — edits affect local syntax/features; use lower "
            "LR and short sequences to avoid destabilizing the trunk."
        )
    elif role == "late_decision_output":
        recs.append(
            "Mostly late layers — good for style, policy, and output formatting; "
            "safer for product-behavior adapters."
        )

    if uniqueness < 0.95:
        recs.append(
            "Overlaps other sectors — resolve membership before training so "
            "gradients don't fight across groups."
        )

    if group.freeze:
        recs.append("Marked freeze — forensics only; will not receive ESFT updates.")
    elif not group.train:
        recs.append("train=false — enable train when you intend ESFT on this sector.")

    if not recs:
        recs.append("Sector looks coherent — proceed with domain-bound ESFT-LoRA.")
    return recs


def _content_summary(
    group: ExpertGroup,
    top_themes: list[dict[str, Any]],
    structural: dict[str, Any],
    mass_b: float,
    fire_ratio: float,
    model_active_b: float,
) -> str:
    theme_bits = []
    for t in top_themes[:3]:
        if t.get("score", 0) >= 0.12:
            theme_bits.append(f"{t['label']} ({t['score']:.2f})")
    theme_s = ", ".join(theme_bits) if theme_bits else "unlabeled / needs probe"
    role = _layer_role_label(structural.get("layer_role_primary", "unknown"))
    dom = group.domain or "unassigned-domain"
    return (
        f"«{group.name}» holds ~{mass_b:.2f}B expert mass "
        f"({fire_ratio:.2f}× one ~{model_active_b}B active fire). "
        f"Assigned domain: {dom}. "
        f"Structural role: {role}. "
        f"Content signature: {theme_s}."
    )


def forensics_for_group(
    plan: GroupPlan,
    group_id: str,
    *,
    affinity: Optional[dict[str, Any]] = None,
    theme_bank: Optional[dict[str, dict[str, Any]]] = None,
    domain_theme_map: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Deep forensic dossier for one sector (JSON-serializable)."""
    g = plan.group_by_id(group_id)
    if not g:
        return {"error": "group not found", "group_id": group_id}

    from aetherforge.groups.evidence import score_themes_for_group

    bank = theme_bank or DEFAULT_THEME_BANK
    estimate_group_capacity(g, plan.capacity)
    n_layers = plan.capacity.num_layers or 1
    structural = _structural_profile(g, n_layers)

    aff_matrix = None
    if affinity:
        aff_matrix = affinity.get("affinity") or affinity.get("routing_freq")

    theme_report = score_themes_for_group(
        g,
        bank,
        affinity=affinity,
        domain_theme_map=domain_theme_map,
    )
    scores = theme_report["theme_scores"]
    top_themes = theme_report["top_themes"]
    conf = theme_report["confidence"]
    evidence_tier = theme_report["evidence_tier"]

    others = [x for x in plan.groups if x.id != g.id]
    uniq = _uniqueness(g, others)
    shared = []
    for o in others:
        ov = _overlap_mass(g, o)
        if ov:
            shared.append({"group_id": o.id, "name": o.name, "shared_cells": ov})

    affs = [c.affinity for c in g.cells]
    top_cells = [
        {
            "layer": c.layer,
            "expert": c.expert,
            "affinity": round(c.affinity, 5),
            "key": c.key,
            "layer_role": _layer_role(c.layer, n_layers),
        }
        for c in sorted(g.cells, key=lambda x: x.affinity, reverse=True)[:32]
    ]

    evidence = [f"evidence_tier={evidence_tier}"]
    if g.domain:
        evidence.append(f"operator-assigned domain={g.domain}")
    if g.topics:
        evidence.append(f"{len(g.topics)} topics bound")
    if aff_matrix is not None:
        synthetic = bool(
            (affinity or {}).get("synthetic")
            or ((affinity or {}).get("metadata") or {}).get("synthetic")
        )
        evidence.append(
            "affinity/routing matrix available"
            + (" (synthetic)" if synthetic else "")
        )
    else:
        evidence.append("no routing matrix — structure + assignment only")
    if top_themes and top_themes[0]["score"] >= 0.25 and evidence_tier != "structure_only":
        evidence.append(f"calibrated theme peak: {top_themes[0]['id']}={top_themes[0]['score']:.2f}")
    evidence.extend(theme_report.get("notes") or [])

    recs = _edit_recommendations(
        g,
        mass_ratio=g.active_fire_ratio,
        top_themes=top_themes if evidence_tier != "structure_only" else [],
        structural=structural,
        uniqueness=uniq,
        family=plan.family,
    )
    if evidence_tier == "structure_only":
        recs = [
            "Evidence tier structure_only — bind domain/topics/keywords or run "
            "multi-theme affinity probes before claiming content identity."
        ] + recs
    summary = _content_summary(
        g,
        top_themes if evidence_tier != "structure_only" else [],
        structural,
        g.est_params_b,
        g.active_fire_ratio,
        plan.capacity.active_params_b,
    )
    if evidence_tier == "structure_only":
        summary = (
            f"«{g.name}» holds ~{g.est_params_b:.2f}B expert mass "
            f"({g.active_fire_ratio:.2f}× fire). "
            f"Evidence tier: structure_only — no content identity claim."
        )

    dossier = SectorForensics(
        group_id=g.id,
        name=g.name,
        family=plan.family,
        model_name=plan.model_name,
        n_cells=len(g.cells),
        est_params_b=g.est_params_b,
        active_fire_ratio=g.active_fire_ratio,
        active_params_b_model=plan.capacity.active_params_b,
        layer_span=structural["layer_span"],
        layer_histogram=structural["layer_histogram"],
        layer_role_primary=structural["layer_role_primary"],
        expert_id_span=structural["expert_id_span"],
        top_cells=top_cells,
        assigned_domain=g.domain,
        assigned_topics=list(g.topics or []),
        assigned_keywords=list(g.keywords or []),
        theme_scores={k: round(v, 4) for k, v in scores.items()},
        top_themes=top_themes[:8],
        content_summary=summary,
        uniqueness=round(uniq, 4),
        shared_with=shared,
        edit_recommendations=recs,
        train_flags={
            "enabled": g.enabled,
            "train": g.train,
            "freeze": g.freeze,
        },
        confidence=conf,
        evidence=evidence,
        evidence_tier=evidence_tier,
        raw={
            "affinity_mean": round(sum(affs) / len(affs), 5) if affs else 0.0,
            "affinity_max": round(max(affs), 5) if affs else 0.0,
            "n_unique_layers": structural["n_unique_layers"],
            "n_unique_expert_ids": structural["n_unique_expert_ids"],
            "color": g.color,
            "curated_path": g.curated_path,
            "domain_pack": g.domain_pack,
            "theme_scores_raw": theme_report.get("theme_scores_raw"),
            "affinity_synthetic": theme_report.get("affinity_synthetic"),
            "evidence_tier": evidence_tier,
        },
    )
    return dossier.to_dict()


def run_model_forensics(
    plan: GroupPlan,
    *,
    affinity: Optional[dict[str, Any]] = None,
    theme_bank: Optional[dict[str, dict[str, Any]]] = None,
    apply_labels: bool = False,
) -> ModelForensicsReport:
    """
    Full-model sector inventory: what each group contains + how to edit.

    If apply_labels=True, writes best theme label into empty group.description
    and tags (mutates plan groups in memory).
    """
    bank = theme_bank or DEFAULT_THEME_BANK
    warnings: list[str] = []
    method_parts = ["structure", "capacity", "evidence_tiers", "calibrated_themes"]
    if affinity and (affinity.get("affinity") or affinity.get("routing_freq")):
        method_parts.append("affinity")
        if (affinity.get("metadata") or {}).get("synthetic") or affinity.get("synthetic"):
            method_parts.append("synthetic_affinity")
            warnings.append(
                "Affinity matrix is SYNTHETIC — routing_probed tier is fixture-level, "
                "not weight-level forensics."
            )
    else:
        warnings.append(
            "No affinity/routing matrix — unbound sectors stay structure_only; "
            "content themes zeroed. Bind assignment or load model + probe."
        )
    method_parts.append("theme_bank")

    sectors: list[SectorForensics] = []
    for g in plan.groups:
        d = forensics_for_group(plan, g.id, affinity=affinity, theme_bank=bank)
        if d.get("error"):
            continue
        # rebuild dataclass lightly from dict for report list
        sectors.append(
            SectorForensics(
                group_id=d["group_id"],
                name=d["name"],
                family=d["family"],
                model_name=d["model_name"],
                n_cells=d["mass"]["n_cells"],
                est_params_b=d["mass"]["est_params_b"],
                active_fire_ratio=d["mass"]["active_fire_ratio"],
                active_params_b_model=d["mass"]["model_active_params_b"],
                layer_span=d["structure"]["layer_span"],
                layer_histogram=d["structure"]["layer_histogram"],
                layer_role_primary=d["structure"]["layer_role_primary"],
                expert_id_span=d["structure"]["expert_id_span"],
                top_cells=d["structure"]["top_cells"],
                assigned_domain=d["content"]["assigned_domain"],
                assigned_topics=d["content"]["assigned_topics"],
                assigned_keywords=d["content"]["assigned_keywords"],
                theme_scores=d["content"]["theme_scores"],
                top_themes=d["content"]["top_themes"],
                content_summary=d["content"]["summary"],
                uniqueness=d["distinctiveness"]["uniqueness"],
                shared_with=d["distinctiveness"]["shared_with"],
                edit_recommendations=d["edit_recommendations"],
                train_flags=d["train_flags"],
                confidence=d["confidence"],
                evidence=d["evidence"],
                evidence_tier=d.get("evidence_tier")
                or (d.get("content") or {}).get("evidence_tier")
                or "structure_only",
                raw=d.get("raw") or {},
            )
        )
        if apply_labels and d["content"]["top_themes"]:
            t0 = d["content"]["top_themes"][0]
            if t0["score"] >= 0.25:
                if not g.description:
                    g.description = (
                        f"[forensics] {t0['label']} · "
                        f"{d['mass']['est_params_b']:.2f}B "
                        f"({d['mass']['active_fire_ratio']:.2f}× fire)"
                    )
                tag = f"theme:{t0['id']}"
                if tag not in (g.tags or []):
                    g.tags = list(g.tags or []) + [tag]
                role_tag = f"role:{d['structure']['layer_role_primary']}"
                if role_tag not in g.tags:
                    g.tags.append(role_tag)

    # sort: largest mass first, then name
    sectors.sort(key=lambda s: (-s.est_params_b, s.name))

    inventory = []
    for s in sectors:
        t0 = s.top_themes[0] if s.top_themes else {}
        score = float(t0.get("score") or 0)
        inventory.append(
            {
                "id": s.group_id,
                "name": s.name,
                "params_b": s.est_params_b,
                "fire_x": s.active_fire_ratio,
                "n_cells": s.n_cells,
                "layer_span": s.layer_span,
                "primary_theme": t0.get("label") if score >= 0.12 else None,
                "theme_score": score if score >= 0.12 else 0.0,
                "layer_role": s.layer_role_primary,
                "domain": s.assigned_domain,
                "train": s.train_flags.get("train"),
                "freeze": s.train_flags.get("freeze"),
                "confidence": s.confidence,
                "evidence_tier": s.evidence_tier,
                "summary": s.content_summary,
            }
        )

    active = plan.capacity.active_params_b
    total = plan.capacity.total_params_b
    family = plan.family
    fire_name = f"~{active}B active fire"
    if family == "qwen_a3b":
        fire_blurb = (
            f"A3B-class MoE: each routing step activates ~{active}B of ~{total}B. "
            "A sector with fire_x≈1.0 is roughly 'one A3B' of capacity you can "
            "inspect and edit as a unit."
        )
    elif family == "deepseek_v4_flash":
        fire_blurb = (
            f"Flash-class MoE: each routing step activates ~{active}B of ~{total}B. "
            "A sector with fire_x≈1.0 is roughly one Flash fire (~13B) of expert mass."
        )
    else:
        fire_blurb = (
            f"MoE with ~{active}B active / ~{total}B total. "
            "fire_x is sector mass relative to one routing fire."
        )

    labeled = sum(1 for s in sectors if s.top_themes and s.top_themes[0].get("score", 0) >= 0.25)
    narrative = (
        f"Forensics for {plan.model_name or family}: {len(sectors)} sectors, "
        f"{plan.capacity.total_expert_slots} expert slots, "
        f"{len(plan.all_assigned_keys())} assigned. {fire_blurb} "
        f"{labeled}/{len(sectors)} sectors have a usable content signature. "
        "Use inventory_table to pick which sector to open before painting data."
    )

    unassigned = max(
        0,
        (plan.capacity.total_expert_slots or 0) - len(plan.all_assigned_keys()),
    )

    report = ModelForensicsReport(
        model_name=plan.model_name,
        family=plan.family,
        capacity=plan.capacity.to_dict(),
        n_groups=len(sectors),
        sectors=sectors,
        unassigned_cells=unassigned,
        inventory_table=inventory,
        narrative=narrative,
        method="+".join(method_parts),
        warnings=warnings,
    )
    log.info(
        "Forensics complete model=%s groups=%d method=%s",
        plan.model_name or plan.family,
        len(sectors),
        report.method,
    )
    return report


def probe_texts_from_theme_bank(
    theme_bank: Optional[dict[str, dict[str, Any]]] = None,
    *,
    per_theme: int = 3,
) -> list[dict[str, str]]:
    """Flatten theme bank into probe records for AffinityProbe / multi-pass forensics."""
    bank = theme_bank or DEFAULT_THEME_BANK
    out: list[dict[str, str]] = []
    for tid, theme in bank.items():
        for i, p in enumerate(theme.get("probes", [])[:per_theme]):
            out.append(
                {
                    "theme_id": tid,
                    "theme_label": theme.get("label", tid),
                    "text": p,
                    "probe_id": f"{tid}:{i}",
                }
            )
    return out


def apply_forensics_to_plan(
    plan: GroupPlan,
    report: Optional[ModelForensicsReport] = None,
    *,
    affinity: Optional[dict[str, Any]] = None,
) -> GroupPlan:
    """
    Attach forensic summaries onto groups (description/tags/notes) and return plan.
    """
    rep = report or run_model_forensics(plan, affinity=affinity, apply_labels=True)
    # also store a compact note on the plan
    plan.notes = (
        (plan.notes + "\n" if plan.notes else "")
        + f"[forensics {time.strftime('%Y-%m-%d')}] {rep.narrative}"
    ).strip()
    plan.updated_at = time.time()
    return plan


def forensics_markdown(report: ModelForensicsReport) -> str:
    """Human-readable forensic inventory."""
    lines = [
        f"# MoE Sector Forensics — {report.model_name or report.family}",
        "",
        report.narrative,
        "",
        f"- Family: `{report.family}`",
        f"- Method: `{report.method}`",
        f"- Sectors: **{report.n_groups}** · Unassigned cells: **{report.unassigned_cells}**",
        f"- Active fire: **~{report.capacity.get('active_params_b')}B** · "
        f"Total: **~{report.capacity.get('total_params_b')}B** · "
        f"Slots: **{report.capacity.get('total_expert_slots')}**",
        "",
    ]
    if report.warnings:
        lines.append("## Warnings")
        for w in report.warnings:
            lines.append(f"- ⚠ {w}")
        lines.append("")

    lines += [
        "## Inventory",
        "",
        "| Sector | Params (B) | Fire× | Cells | Layers | Primary content | Conf | Train |",
        "|--------|------------|-------|-------|--------|-----------------|------|-------|",
    ]
    for row in report.inventory_table:
        span = row.get("layer_span") or ["?", "?"]
        theme = row.get("primary_theme")
        tscore = float(row.get("theme_score") or 0)
        theme_cell = f"{theme} ({tscore:.2f})" if theme else "— (needs probe)"
        lines.append(
            f"| {row['name']} | {row['params_b']:.2f} | {row['fire_x']:.2f} | "
            f"{row['n_cells']} | {span[0]}–{span[1]} | "
            f"{theme_cell} | "
            f"{row.get('confidence')} | "
            f"{'Y' if row.get('train') else 'n'} |"
        )

    lines += ["", "## Sector dossiers", ""]
    for s in report.sectors:
        d = s.to_dict()
        lines.append(f"### {s.name} (`{s.group_id}`)")
        lines.append("")
        lines.append(d["content"]["summary"])
        lines.append("")
        lines.append(
            f"- Mass: **{s.est_params_b:.2f}B** ({s.active_fire_ratio:.2f}× fire) · "
            f"cells={s.n_cells}"
        )
        lines.append(f"- Role: {_layer_role_label(s.layer_role_primary)}")
        if s.layer_span:
            lines.append(f"- Layers: {s.layer_span[0]}–{s.layer_span[1]}")
        if s.assigned_domain:
            lines.append(f"- Domain binding: `{s.assigned_domain}`")
        lines.append("- Top themes:")
        for t in s.top_themes[:5]:
            if t["score"] <= 0:
                continue
            lines.append(f"  - {t['label']}: {t['score']:.2f}")
        lines.append("- Edit recommendations:")
        for r in s.edit_recommendations:
            lines.append(f"  - {r}")
        lines.append("")

    lines.append("---")
    lines.append("*AetherForge sector forensics — know the lattice before you paint it.*")
    return "\n".join(lines)
