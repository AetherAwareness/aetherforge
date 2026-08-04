"""
Forensic evidence tiers and calibrated theme scoring.

Tiers (honest labels for what we actually know):
  structure_only  — geometry / mass / depth only; no content claim
  assignment      — operator or pack bound domain/topics/keywords
  routing_probed  — affinity/routing matrix present (may be synthetic fixture)

High-confidence content claims require assignment or routing_probed.
Geometry-only sectors must not saturate multi-theme scores to ~1.0.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal, Optional

from aetherforge.groups.models import ExpertGroup

EvidenceTier = Literal["structure_only", "assignment", "routing_probed"]

_TOKEN_RE = re.compile(r"[a-z0-9+]{3,}")


def token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def resolve_evidence_tier(
    group: ExpertGroup,
    *,
    affinity: Optional[dict[str, Any]] = None,
    domain_theme_map: Optional[dict[str, float]] = None,
) -> EvidenceTier:
    """
    Highest justified tier for this sector.

    routing_probed if a real routing matrix (or multi-theme map) is available.
    assignment if domain/topics/keywords/curated_path bound.
    else structure_only.
    """
    has_routing = False
    if affinity:
        mat = affinity.get("affinity") or affinity.get("routing_freq")
        if mat is not None:
            has_routing = True
        # multi-theme probe maps count as routing_probed evidence
        if affinity.get("domain_theme_map") or affinity.get("multi_theme"):
            has_routing = True
    if domain_theme_map:
        has_routing = True
    if has_routing:
        return "routing_probed"

    has_assign = bool(
        group.domain
        or (group.topics and len(group.topics) > 0)
        or (group.keywords and len(group.keywords) > 0)
        or group.curated_path
        or group.domain_pack
    )
    if has_assign:
        return "assignment"
    return "structure_only"


def raw_theme_hits(
    bag: str,
    theme_bank: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """
    Raw keyword hit rates 0..1 per theme (unnormalized, no saturation tricks).
    Hit rate = matched_keywords / n_keywords (not inflated by 0.35 divisor).
    """
    text = (bag or "").lower()
    toks = token_set(text)
    scores: dict[str, float] = {}
    for tid, theme in theme_bank.items():
        kws = [str(k).lower() for k in theme.get("keywords") or []]
        if not kws:
            scores[tid] = 0.0
            continue
        hits = 0
        for kw in kws:
            parts = kw.split()
            if all(p in toks or p in text for p in parts):
                hits += 1
        scores[tid] = hits / len(kws)
    return scores


def calibrate_theme_scores(
    raw: dict[str, float],
    *,
    tier: EvidenceTier,
    domain_boost: Optional[dict[str, float]] = None,
    domain_theme_map: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    Competitive calibration so unrelated themes do not all hit ~1.0.

    - structure_only: force all content scores to 0 (no content claim)
    - assignment/routing: softmax over temperature-scaled raw scores
    - optional domain_boost / domain_theme_map mixed in before softmax
    """
    if tier == "structure_only":
        return {k: 0.0 for k in raw}

    scores = {k: float(v) for k, v in raw.items()}
    if domain_boost:
        for k, v in domain_boost.items():
            if k in scores:
                scores[k] = min(1.0, scores[k] + float(v))
    if domain_theme_map:
        for k, v in domain_theme_map.items():
            if k in scores:
                scores[k] = max(scores[k], float(v))

    # Competitive softmax: only relative strength matters
    # temperature keeps mild peaks from becoming uniform 1.0
    temp = 0.35 if tier == "routing_probed" else 0.45
    # shift for numerical stability
    vals = list(scores.values())
    if not vals or max(vals) <= 0:
        return {k: 0.0 for k in scores}
    mx = max(vals)
    exps = {k: math.exp((v - mx) / temp) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    # scale so top is at most ~0.95 and sum is 1
    soft = {k: exps[k] / z for k in scores}
    # If only one theme has mass, soft is peaked; if many equal raw, soft is flat ~1/n
    # Cap absolute claim by raw evidence strength of the winner
    top_raw = max(scores.values())
    ceiling = min(0.95, 0.15 + 0.85 * top_raw) if tier == "assignment" else min(0.98, 0.2 + 0.8 * top_raw)
    peak = max(soft.values()) or 1.0
    scale = ceiling / peak if peak > 0 else 1.0
    return {k: round(min(1.0, soft[k] * scale), 6) for k in soft}


def domain_boost_map(
    group: ExpertGroup,
    theme_bank: dict[str, dict[str, Any]],
    affinity_domain: str = "",
) -> dict[str, float]:
    """Light boost when domain slug matches a theme id/label (assignment signal)."""
    boost: dict[str, float] = {}
    dom = (group.domain or affinity_domain or "").lower().replace("-", "_")
    if not dom:
        return boost
    for tid, theme in theme_bank.items():
        label = str(theme.get("label", "")).lower()
        if dom in tid or tid in dom or any(p and p in label for p in dom.split("_") if len(p) > 3):
            boost[tid] = 0.35
    return boost


def content_bag(group: ExpertGroup, affinity_domain: str = "") -> str:
    parts = [
        group.domain or "",
        group.description or "",
        group.notes or "",
        " ".join(group.topics or []),
        " ".join(group.keywords or []),
        " ".join(group.tags or []),
        affinity_domain or "",
    ]
    return " ".join(parts)


def confidence_from_tier(
    tier: EvidenceTier,
    top_theme_score: float,
    *,
    affinity_synthetic: bool = False,
) -> str:
    """
    high only with non-synthetic routing + strong peak, or strong assignment peak.
    structure_only always low for content.
    """
    if tier == "structure_only":
        return "low"
    if tier == "assignment":
        if top_theme_score >= 0.45:
            return "medium"
        if top_theme_score >= 0.25:
            return "medium"
        return "low"
    # routing_probed
    if affinity_synthetic:
        # synthetic matrices cannot claim high content confidence
        return "medium" if top_theme_score >= 0.4 else "low"
    if top_theme_score >= 0.4:
        return "high"
    if top_theme_score >= 0.2:
        return "medium"
    return "low"


def refuse_high_confidence_content(
    tier: EvidenceTier,
    confidence: str,
) -> tuple[str, list[str]]:
    """
    Downgrade illegal high-confidence content claims.
    Returns (confidence, notes).
    """
    notes: list[str] = []
    if confidence == "high" and tier == "structure_only":
        notes.append(
            "High-confidence content claim refused: evidence tier is structure_only"
        )
        return "low", notes
    if confidence == "high" and tier == "assignment":
        # assignment alone maxes at medium unless routing confirms
        notes.append(
            "High-confidence content claim reduced: assignment without routing_probed"
        )
        return "medium", notes
    return confidence, notes


def score_themes_for_group(
    group: ExpertGroup,
    theme_bank: dict[str, dict[str, Any]],
    *,
    affinity: Optional[dict[str, Any]] = None,
    domain_theme_map: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """
    Full calibrated theme report for one group.

    Returns dict with tier, raw, calibrated scores, top_themes, confidence, notes.
    """
    aff_domain = ""
    synthetic = False
    dtm = dict(domain_theme_map or {})
    if affinity:
        aff_domain = str(affinity.get("domain") or "")
        synthetic = bool(
            (affinity.get("metadata") or {}).get("synthetic")
            or affinity.get("synthetic")
        )
        if affinity.get("domain_theme_map"):
            dtm.update({k: float(v) for k, v in affinity["domain_theme_map"].items()})

    tier = resolve_evidence_tier(
        group, affinity=affinity, domain_theme_map=dtm or None
    )
    bag = content_bag(group, aff_domain)
    raw = raw_theme_hits(bag, theme_bank)
    boost = domain_boost_map(group, theme_bank, aff_domain)
    calibrated = calibrate_theme_scores(
        raw,
        tier=tier,
        domain_boost=boost if tier != "structure_only" else None,
        domain_theme_map=dtm if tier == "routing_probed" else None,
    )
    top = sorted(
        [
            {
                "id": tid,
                "label": theme_bank[tid].get("label", tid),
                "score": calibrated[tid],
                "raw": round(raw.get(tid, 0.0), 4),
            }
            for tid in calibrated
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
    top_score = float(top[0]["score"]) if top else 0.0
    conf = confidence_from_tier(tier, top_score, affinity_synthetic=synthetic)
    conf, notes = refuse_high_confidence_content(tier, conf)
    if tier == "structure_only":
        notes.append(
            "Content themes zeroed: structure_only — bind domain/topics or run routing probes"
        )
    if synthetic and tier == "routing_probed":
        notes.append(
            "Affinity matrix marked synthetic — routing_probed but not weight-level forensics"
        )

    return {
        "evidence_tier": tier,
        "theme_scores_raw": {k: round(v, 4) for k, v in raw.items()},
        "theme_scores": calibrated,
        "top_themes": top[:8],
        "confidence": conf,
        "affinity_synthetic": synthetic,
        "notes": notes,
        "bag_nonempty": bool(bag.strip()),
    }
