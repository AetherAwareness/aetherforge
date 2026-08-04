"""
Pre/post sector routing probe (fixture-capable, no weights required).

Uses affinity matrices (live or synthetic) to estimate how strongly a sector's
cells capture domain mass, and compares pre vs post to decide keep/rollback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from aetherforge.groups.models import ExpertGroup, GroupPlan
from aetherforge.utils.logging import get_logger

log = get_logger("training.sector_probe")


def _matrix_from_affinity(affinity: Optional[dict[str, Any]]) -> Optional[list[list[float]]]:
    if not affinity:
        return None
    mat = affinity.get("affinity") or affinity.get("routing_freq")
    if mat is None:
        return None
    # normalize rows if needed
    out: list[list[float]] = []
    for row in mat:
        row = [float(x) for x in row]
        s = sum(row) or 1.0
        # if looks like counts, normalize
        if s > 1.5:
            row = [x / s for x in row]
        out.append(row)
    return out


def sector_mass_on_matrix(
    group: ExpertGroup,
    matrix: list[list[float]],
) -> dict[str, float]:
    """
    Fraction of per-layer affinity mass landing on this sector's experts.
    """
    if not group.cells or not matrix:
        return {"mean_share": 0.0, "total_share": 0.0, "n_layers_hit": 0.0}
    by_layer: dict[int, set[int]] = {}
    for c in group.cells:
        by_layer.setdefault(c.layer, set()).add(c.expert)

    shares: list[float] = []
    for li, experts in by_layer.items():
        if li < 0 or li >= len(matrix):
            continue
        row = matrix[li]
        layer_sum = 0.0
        for ei in experts:
            if 0 <= ei < len(row):
                layer_sum += float(row[ei])
        shares.append(layer_sum)
    if not shares:
        return {"mean_share": 0.0, "total_share": 0.0, "n_layers_hit": 0.0}
    return {
        "mean_share": float(sum(shares) / len(shares)),
        "total_share": float(sum(shares)),
        "n_layers_hit": float(len(shares)),
    }


def synthetic_post_boost(
    pre_matrix: list[list[float]],
    group: ExpertGroup,
    *,
    boost: float = 0.08,
) -> list[list[float]]:
    """
    Dry-run / fixture: simulate a mild post-train affinity lift on sector cells
    (redistribute from non-sector cells on the same layer). No weights required.
    """
    import copy

    mat = copy.deepcopy(pre_matrix)
    keys = {(c.layer, c.expert) for c in group.cells}
    for li, row in enumerate(mat):
        sector_idx = [ei for ei in range(len(row)) if (li, ei) in keys]
        other_idx = [ei for ei in range(len(row)) if (li, ei) not in keys]
        if not sector_idx or not other_idx:
            continue
        # take mass from others, add to sector
        take = 0.0
        for ei in other_idx:
            delta = min(row[ei] * 0.15, boost / max(len(other_idx), 1))
            row[ei] = max(0.0, row[ei] - delta)
            take += delta
        add = take / len(sector_idx)
        for ei in sector_idx:
            row[ei] += add
        s = sum(row) or 1.0
        mat[li] = [x / s for x in row]
    return mat


@dataclass
class SectorProbeResult:
    group_id: str
    name: str
    phase: str  # pre | post
    mean_share: float
    total_share: float
    n_layers_hit: float
    synthetic_affinity: bool = False
    matrix_source: str = "affinity"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "phase": self.phase,
            "mean_share": round(self.mean_share, 6),
            "total_share": round(self.total_share, 6),
            "n_layers_hit": self.n_layers_hit,
            "synthetic_affinity": self.synthetic_affinity,
            "matrix_source": self.matrix_source,
            "details": self.details,
        }


@dataclass
class SectorProbeDelta:
    group_id: str
    name: str
    pre: SectorProbeResult
    post: SectorProbeResult
    delta_mean_share: float
    keep: bool
    decision: str  # keep | rollback
    reason: str
    min_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "pre": self.pre.to_dict(),
            "post": self.post.to_dict(),
            "delta_mean_share": round(self.delta_mean_share, 6),
            "keep": self.keep,
            "decision": self.decision,
            "reason": self.reason,
            "min_delta": self.min_delta,
        }


def probe_sector(
    plan: GroupPlan,
    group: ExpertGroup,
    affinity: Optional[dict[str, Any]],
    *,
    phase: str = "pre",
    matrix_override: Optional[list[list[float]]] = None,
) -> SectorProbeResult:
    synthetic = bool(
        (affinity or {}).get("synthetic")
        or ((affinity or {}).get("metadata") or {}).get("synthetic")
    )
    matrix = matrix_override or _matrix_from_affinity(affinity)
    if matrix is None:
        # structure-only: zero routing mass (honest)
        return SectorProbeResult(
            group_id=group.id,
            name=group.name,
            phase=phase,
            mean_share=0.0,
            total_share=0.0,
            n_layers_hit=0.0,
            synthetic_affinity=False,
            matrix_source="none",
            details={"warning": "no affinity matrix"},
        )
    mass = sector_mass_on_matrix(group, matrix)
    return SectorProbeResult(
        group_id=group.id,
        name=group.name,
        phase=phase,
        mean_share=mass["mean_share"],
        total_share=mass["total_share"],
        n_layers_hit=mass["n_layers_hit"],
        synthetic_affinity=synthetic,
        matrix_source="override" if matrix_override is not None else "affinity",
        details={},
    )


def decide_keep_rollback(
    pre: SectorProbeResult,
    post: SectorProbeResult,
    *,
    min_delta: float = -0.02,
    require_non_negative: bool = True,
) -> SectorProbeDelta:
    """
    Keep sector adapter if post mean_share did not regress beyond min_delta.

    Default min_delta=-0.02 allows tiny noise; large negative → rollback.
    """
    delta = post.mean_share - pre.mean_share
    keep = True
    reason = "routing share stable or improved"
    if require_non_negative and delta < min_delta:
        keep = False
        reason = (
            f"routing share regressed (delta={delta:.4f} < min_delta={min_delta})"
        )
    elif delta >= 0:
        reason = f"routing share improved by {delta:.4f}"
    return SectorProbeDelta(
        group_id=pre.group_id,
        name=pre.name,
        pre=pre,
        post=post,
        delta_mean_share=delta,
        keep=keep,
        decision="keep" if keep else "rollback",
        reason=reason,
        min_delta=min_delta,
    )


def interference_summary(
    plan: GroupPlan,
    affinity_pre: Optional[dict[str, Any]],
    affinity_post: Optional[dict[str, Any]],
    *,
    trained_group_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Cross-sector mass shift summary: which sectors gained/lost routing share.
    """
    pre_m = _matrix_from_affinity(affinity_pre)
    post_m = _matrix_from_affinity(affinity_post)
    rows = []
    if pre_m is None or post_m is None:
        return {
            "ok": False,
            "warning": "missing pre/post affinity for interference matrix",
            "sectors": [],
        }
    trained = set(trained_group_ids or [g.id for g in plan.enabled_train_groups()])
    for g in plan.groups:
        pre = sector_mass_on_matrix(g, pre_m)
        post = sector_mass_on_matrix(g, post_m)
        delta = post["mean_share"] - pre["mean_share"]
        rows.append(
            {
                "group_id": g.id,
                "name": g.name,
                "trained": g.id in trained,
                "pre_mean_share": round(pre["mean_share"], 6),
                "post_mean_share": round(post["mean_share"], 6),
                "delta_mean_share": round(delta, 6),
                "regressed": delta < -0.02 and g.id not in trained,
            }
        )
    regressions = [r for r in rows if r["regressed"]]
    return {
        "ok": True,
        "schema": "aetherforge.interference.v1",
        "n_sectors": len(rows),
        "n_regressions": len(regressions),
        "sectors": rows,
        "regressions": regressions,
        "narrative": (
            f"Interference: {len(regressions)} non-trained sector(s) regressed "
            f"routing share beyond -0.02 among {len(rows)} sectors."
            if regressions
            else f"Interference: no sibling regressions beyond threshold ({len(rows)} sectors)."
        ),
    }
