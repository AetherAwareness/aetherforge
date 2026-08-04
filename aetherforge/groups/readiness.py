"""
Pre-sector forensic readiness gate.

Before any ESFT step on a sector, inventory what that sector *contains* and
decide whether it is safe / useful to train. Gates are industry-agnostic —
they check structure, content identity, and data binding, not field tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from aetherforge.groups.forensics import (
    DEFAULT_THEME_BANK,
    forensics_for_group,
    run_model_forensics,
)
from aetherforge.groups.models import ExpertGroup, GroupPlan
from aetherforge.utils.logging import get_logger

log = get_logger("groups.readiness")

GateStatus = Literal["pass", "warn", "block"]


@dataclass
class SectorReadiness:
    """Readiness verdict for one sector before training."""

    group_id: str
    name: str
    status: GateStatus
    score: float  # 0..1 readiness
    reasons: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    forensics: Optional[dict[str, Any]] = None
    binding: dict[str, Any] = field(default_factory=dict)
    train_eligible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "status": self.status,
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "fixes": self.fixes,
            "binding": self.binding,
            "train_eligible": self.train_eligible,
            "forensics": self.forensics,
        }


@dataclass
class ReadinessReport:
    """Gate report for all train-eligible sectors (and optional frozen inventory)."""

    model_name: str
    family: str
    mode: str  # block | warn | skip
    sectors: list[SectorReadiness]
    n_pass: int = 0
    n_warn: int = 0
    n_block: int = 0
    n_skipped_frozen: int = 0
    overall: GateStatus = "pass"
    narrative: str = ""
    auto_bound: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aetherforge.readiness.v1",
            "model_name": self.model_name,
            "family": self.family,
            "mode": self.mode,
            "overall": self.overall,
            "n_pass": self.n_pass,
            "n_warn": self.n_warn,
            "n_block": self.n_block,
            "n_skipped_frozen": self.n_skipped_frozen,
            "narrative": self.narrative,
            "auto_bound": self.auto_bound,
            "sectors": [s.to_dict() for s in self.sectors],
        }

    @property
    def blocked(self) -> bool:
        return self.overall == "block" or self.n_block > 0

    def train_group_ids(self) -> list[str]:
        return [s.group_id for s in self.sectors if s.train_eligible and s.status != "block"]


def _primary_theme(forensics: dict[str, Any]) -> Optional[dict[str, Any]]:
    themes = (forensics.get("content") or {}).get("top_themes") or []
    if not themes:
        return None
    t0 = themes[0]
    if float(t0.get("score") or 0) <= 0:
        return None
    return t0


def auto_bind_sector_from_forensics(
    group: ExpertGroup,
    forensics: dict[str, Any],
    *,
    global_domain: Optional[str] = None,
    theme_bank: Optional[dict[str, dict[str, Any]]] = None,
    min_theme_score: float = 0.2,
    allow_theme_content: Optional[bool] = None,
) -> dict[str, Any]:
    """
    Fill empty domain/topics/keywords on a group from forensic content signature.

    Never overwrites operator-provided bindings. Returns what was applied.

    Rules (anti-hallucination):
      - structure_only / zero-score themes MUST NOT inject topics/keywords from the
        theme bank (that reintroduced multi-theme saturation after dry-run auto-bind).
      - Only themes with score >= min_theme_score may contribute labels/keywords.
      - global_domain alone may set domain slug only (no multi-theme keyword paint).
    """
    bank = theme_bank or DEFAULT_THEME_BANK
    applied: dict[str, Any] = {}
    content = forensics.get("content") or {}
    themes = content.get("top_themes") or []
    tier = (
        forensics.get("evidence_tier")
        or content.get("evidence_tier")
        or "structure_only"
    )
    # Positive-score themes only
    peaked = [
        t
        for t in themes
        if float(t.get("score") or 0) >= min_theme_score and t.get("id")
    ]
    # Default: theme content only when we have real peaks (not structure_only zeros)
    if allow_theme_content is None:
        allow_theme_content = tier != "structure_only" and len(peaked) > 0

    if not group.domain:
        if global_domain:
            group.domain = global_domain
            applied["domain"] = global_domain
        elif peaked:
            tid = peaked[0].get("id")
            theme = bank.get(tid or "", {})
            label = theme.get("label") or peaked[0].get("label") or tid
            group.domain = (tid or "sector").replace("_", "-")
            applied["domain"] = group.domain
            applied["domain_label"] = label

    # Never invent multi-theme topics/keywords from zeroed structure_only dossiers
    if allow_theme_content and not group.topics:
        topics: list[str] = []
        for t in peaked[:3]:
            tid = t.get("id")
            theme = bank.get(tid or "", {})
            if theme.get("label"):
                topics.append(str(theme["label"]))
            elif t.get("label"):
                topics.append(str(t["label"]))
        if content.get("assigned_topics"):
            topics = list(content["assigned_topics"]) + topics
        seen: set[str] = set()
        clean = []
        for t in topics:
            k = t.lower().strip()
            if k and k not in seen:
                seen.add(k)
                clean.append(t.strip())
        if clean:
            group.topics = clean[:12]
            applied["topics"] = group.topics

    if allow_theme_content and not group.keywords:
        kws: list[str] = []
        for t in peaked[:2]:  # at most 2 peaked themes — avoid multi-theme soup
            tid = t.get("id")
            theme = bank.get(tid or "", {})
            kws.extend(list(theme.get("keywords") or [])[:6])
        if content.get("assigned_keywords"):
            kws = list(content["assigned_keywords"]) + kws
        seen_k: set[str] = set()
        clean_k = []
        for k in kws:
            kk = k.lower().strip()
            if kk and kk not in seen_k:
                seen_k.add(kk)
                clean_k.append(k.strip())
        if clean_k:
            group.keywords = clean_k[:24]
            applied["keywords"] = group.keywords

    # Tag primary theme only when peaked
    t0 = peaked[0] if peaked else None
    if allow_theme_content and t0 and t0.get("id"):
        tag = f"theme:{t0['id']}"
        tags = list(group.tags or [])
        if tag not in tags:
            tags.append(tag)
            group.tags = tags
            applied["theme_tag"] = tag

    return applied


def assess_sector_readiness(
    plan: GroupPlan,
    group: ExpertGroup,
    *,
    forensics: Optional[dict[str, Any]] = None,
    affinity: Optional[dict[str, Any]] = None,
    mode: str = "warn",
    min_theme_score: float = 0.12,
    min_cells: int = 1,
    max_fire_ratio: float = 2.5,
    min_fire_ratio: float = 0.15,
    require_content_identity: bool = True,
    has_sector_data: bool = False,
    n_sector_samples: int = 0,
    min_sector_samples: int = 4,
) -> SectorReadiness:
    """
    Forensic readiness for one sector.

    mode:
      block — hard fail on critical issues (used when groups.require_forensics_gate)
      warn  — soft; still train_eligible unless structural empty
      skip  — always pass (inventory only)
    """
    dossier = forensics or forensics_for_group(plan, group.id, affinity=affinity)
    if dossier.get("error"):
        return SectorReadiness(
            group_id=group.id,
            name=group.name,
            status="block",
            score=0.0,
            reasons=[str(dossier.get("error"))],
            fixes=["Fix group membership / plan"],
            forensics=dossier,
            train_eligible=False,
        )

    reasons: list[str] = []
    fixes: list[str] = []
    score = 1.0
    status: GateStatus = "pass"
    train_eligible = bool(group.enabled and group.train and not group.freeze)

    n_cells = int(dossier.get("mass", {}).get("n_cells") or len(group.cells) or 0)
    fire = float(dossier.get("mass", {}).get("active_fire_ratio") or group.active_fire_ratio or 0.0)
    content = dossier.get("content") or {}
    themes = content.get("top_themes") or []
    theme0_score = float(themes[0]["score"]) if themes else 0.0
    has_binding = bool(
        group.domain or group.topics or group.keywords or group.curated_path or group.domain_pack
    )
    uniqueness = float((dossier.get("distinctiveness") or {}).get("uniqueness") or 1.0)

    # Structural
    if n_cells < min_cells:
        score -= 0.5
        reasons.append(f"empty sector ({n_cells} cells)")
        fixes.append("Add expert cells or re-partition groups")
        train_eligible = False
        status = "block"

    if fire > max_fire_ratio:
        score -= 0.2
        reasons.append(f"sector mass {fire:.2f}× active fire (large)")
        fixes.append("Split sector before training so updates stay localized")
        if mode == "block" and fire > max_fire_ratio * 1.2:
            status = "block"
        elif status != "block":
            status = "warn"

    if fire > 0 and fire < min_fire_ratio and train_eligible:
        score -= 0.1
        reasons.append(f"sector mass only {fire:.2f}× active fire (thin)")
        fixes.append("Merge with sibling or add cells for a full fire budget")
        if status == "pass":
            status = "warn"

    if uniqueness < 0.9:
        score -= 0.15
        reasons.append(f"overlaps siblings (uniqueness={uniqueness:.2f})")
        fixes.append("Resolve membership so gradients do not fight across groups")
        if mode == "block" and uniqueness < 0.7:
            status = "block"
        elif status != "block":
            status = "warn"

    # Content identity (forensics of *what is in* the sector)
    if require_content_identity and train_eligible:
        if not has_binding and theme0_score < min_theme_score:
            score -= 0.35
            reasons.append("no content identity — unbound and weak theme signature")
            fixes.append(
                "Bind domain/topics/keywords or curated_path, or enable "
                "auto_bind_from_forensics / multi-theme affinity probes"
            )
            if mode == "block":
                status = "block"
            else:
                status = "warn" if status != "block" else status
        elif not has_binding and theme0_score >= min_theme_score:
            score -= 0.1
            reasons.append(
                f"unbound but theme signature present ({themes[0].get('label')}: {theme0_score:.2f})"
            )
            fixes.append("Auto-bind from forensics or set domain pack on this sector")
            if status == "pass":
                status = "warn"
        elif has_binding and theme0_score < min_theme_score * 0.5:
            score -= 0.05
            reasons.append("bound domain but weak theme confirmation")
            if status == "pass":
                status = "warn"

    # Dataset readiness (optional — caller may re-assess after shard)
    if train_eligible and min_sector_samples > 0:
        if has_sector_data and n_sector_samples < min_sector_samples:
            score -= 0.25
            reasons.append(
                f"sector dataset too small ({n_sector_samples} < {min_sector_samples})"
            )
            fixes.append("Increase synthetic samples or curated corpus for this sector")
            if mode == "block":
                status = "block"
            elif status != "block":
                status = "warn"
        elif not has_sector_data and n_sector_samples == 0:
            # not yet built — informational only unless mode=block and no global fallback
            reasons.append("sector dataset not built yet (will be forged pre-train)")

    if group.freeze or not group.train:
        train_eligible = False
        reasons.append("train disabled or frozen")
        status = "pass"  # frozen is intentional, not a failure

    score = max(0.0, min(1.0, score))
    if mode == "skip":
        status = "pass"

    # escalate: any block forces status block
    if status == "block" and mode == "warn":
        # in warn mode, structural empty still blocks train_eligible
        if n_cells >= min_cells and train_eligible:
            status = "warn"
            # keep train_eligible True for soft mode unless empty
        elif n_cells < min_cells:
            status = "block"

    evidence_tier = dossier.get("evidence_tier") or (dossier.get("content") or {}).get(
        "evidence_tier"
    ) or "structure_only"
    if evidence_tier == "structure_only" and train_eligible and require_content_identity:
        score -= 0.05
        reasons.append("evidence_tier=structure_only (no content claim)")
        if status == "pass":
            status = "warn"

    return SectorReadiness(
        group_id=group.id,
        name=group.name,
        status=status,
        score=score,
        reasons=reasons,
        fixes=fixes,
        forensics=dossier,
        binding={
            "domain": group.domain,
            "topics": list(group.topics or []),
            "keywords": list(group.keywords or [])[:16],
            "curated_path": group.curated_path,
            "domain_pack": group.domain_pack,
            "primary_theme": (themes[0] if themes else None),
            "evidence_tier": evidence_tier,
        },
        train_eligible=train_eligible and status != "block",
    )


def run_forensics_gate(
    plan: GroupPlan,
    *,
    affinity: Optional[dict[str, Any]] = None,
    mode: str = "warn",
    auto_bind: bool = True,
    global_domain: Optional[str] = None,
    only_train_groups: bool = True,
    min_sector_samples: int = 0,
    sector_sample_counts: Optional[dict[str, int]] = None,
) -> ReadinessReport:
    """
    Full pre-training forensic gate across sectors.

    Always runs forensics first. Optionally auto-binds unbound train sectors
    from content signatures, then assesses readiness.
    """
    mode = (mode or "warn").lower()
    if mode not in ("block", "warn", "skip"):
        mode = "warn"

    full = run_model_forensics(plan, affinity=affinity, apply_labels=False)
    dossiers = {s.group_id: s.to_dict() for s in full.sectors}

    auto_bound: list[str] = []
    targets = plan.enabled_train_groups() if only_train_groups else list(plan.groups)

    if auto_bind:
        for g in targets:
            d = dossiers.get(g.id) or forensics_for_group(plan, g.id, affinity=affinity)
            tier = d.get("evidence_tier") or (d.get("content") or {}).get(
                "evidence_tier"
            ) or "structure_only"
            themes = (d.get("content") or {}).get("top_themes") or []
            top_sc = float(themes[0].get("score") or 0) if themes else 0.0
            has_peak = top_sc >= 0.2
            # structure_only with zeroed themes: only domain-slug bind (no theme paint)
            # skip entirely when nothing to bind
            if tier == "structure_only" and not has_peak and not global_domain:
                continue
            applied = auto_bind_sector_from_forensics(
                g,
                d,
                global_domain=global_domain,
                # structure_only zeros must never paint multi-theme keywords
                allow_theme_content=(tier != "structure_only" and has_peak),
            )
            if applied:
                auto_bound.append(g.id)
                # refresh dossier content fields after bind
                d = forensics_for_group(plan, g.id, affinity=affinity)
                dossiers[g.id] = d

    sample_counts = sector_sample_counts or {}
    results: list[SectorReadiness] = []
    n_pass = n_warn = n_block = 0
    n_frozen = sum(
        1 for g in plan.groups if g.freeze or not g.train or not g.enabled
    )

    for g in targets:
        d = dossiers.get(g.id)
        n_samp = int(sample_counts.get(g.id, 0))
        r = assess_sector_readiness(
            plan,
            g,
            forensics=d,
            affinity=affinity,
            mode=mode,
            has_sector_data=g.id in sample_counts,
            n_sector_samples=n_samp,
            min_sector_samples=min_sector_samples if sample_counts else 0,
        )
        results.append(r)
        if r.status == "pass":
            n_pass += 1
        elif r.status == "warn":
            n_warn += 1
        else:
            n_block += 1

    overall: GateStatus = "pass"
    if mode == "block" and n_block > 0:
        overall = "block"
    elif n_block > 0 or n_warn > 0:
        overall = "warn" if mode != "block" or n_block == 0 else "block"
        if n_block > 0 and mode == "block":
            overall = "block"
        elif n_warn > 0 and n_block == 0:
            overall = "warn"
    if mode == "skip":
        overall = "pass"

    # clarify overall when block mode and blocks exist
    if any(r.status == "block" and r.train_eligible is False for r in results):
        if mode == "block" and any(r.status == "block" for r in results):
            # only overall-block if a *intended* train sector is blocked
            train_blocked = [
                r
                for r in results
                if r.status == "block"
                and not (
                    "train disabled" in " ".join(r.reasons)
                    or "frozen" in " ".join(r.reasons).lower()
                )
            ]
            if train_blocked:
                overall = "block"

    narrative = (
        f"Forensic readiness ({mode}): {n_pass} pass · {n_warn} warn · {n_block} block "
        f"across {len(results)} sector(s). "
        f"Auto-bound {len(auto_bound)} unbound sector(s) from content signatures."
        if auto_bound
        else f"Forensic readiness ({mode}): {n_pass} pass · {n_warn} warn · {n_block} block "
        f"across {len(results)} sector(s)."
    )

    report = ReadinessReport(
        model_name=plan.model_name,
        family=plan.family,
        mode=mode,
        sectors=results,
        n_pass=n_pass,
        n_warn=n_warn,
        n_block=n_block,
        n_skipped_frozen=n_frozen,
        overall=overall,
        narrative=narrative,
        auto_bound=auto_bound,
    )
    log.info(narrative)
    return report


def readiness_markdown(report: ReadinessReport) -> str:
    lines = [
        f"# Pre-sector forensic readiness — {report.model_name or report.family}",
        "",
        report.narrative,
        "",
        f"- Overall: **{report.overall}** (mode=`{report.mode}`)",
        f"- Pass / warn / block: **{report.n_pass}** / **{report.n_warn}** / **{report.n_block}**",
        "",
        "| Sector | Status | Score | Domain | Primary theme | Reasons |",
        "|--------|--------|-------|--------|---------------|---------|",
    ]
    for s in report.sectors:
        theme = (s.binding.get("primary_theme") or {}) if s.binding else {}
        tlabel = theme.get("label") or theme.get("id") or "—"
        reasons = "; ".join(s.reasons[:2]) if s.reasons else "ok"
        lines.append(
            f"| {s.name} | {s.status} | {s.score:.2f} | "
            f"{s.binding.get('domain') or '—'} | {tlabel} | {reasons} |"
        )
    lines.append("")
    for s in report.sectors:
        if s.status == "pass" and not s.fixes:
            continue
        lines.append(f"### {s.name} (`{s.group_id}`)")
        for r in s.reasons:
            lines.append(f"- ⚠ {r}")
        for f in s.fixes:
            lines.append(f"- → {f}")
        lines.append("")
    lines.append("---")
    lines.append("*AetherForge — forensically assess every sector before you train it.*")
    return "\n".join(lines)
