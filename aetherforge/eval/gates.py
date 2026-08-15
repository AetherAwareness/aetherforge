"""Promotion gates over Scorecard axes (industry-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from aetherforge.utils.config import ScorecardThresholds


@dataclass
class GateDecision:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    axes: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "warnings": self.warnings,
            "axes": self.axes,
        }


def apply_gates(
    metrics: dict[str, float],
    thresholds: ScorecardThresholds,
    baseline: Optional[dict[str, float]] = None,
) -> GateDecision:
    failures: list[str] = []
    warnings: list[str] = []
    axes: dict[str, bool] = {}

    domain = metrics.get("domain_score", 0.0)
    ok = domain >= thresholds.domain_score
    axes["domain_score"] = ok
    if not ok:
        failures.append(f"domain_score {domain:.3f} < {thresholds.domain_score}")

    if baseline and "general_score" in baseline and "general_score" in metrics:
        delta = metrics["general_score"] - baseline["general_score"]
        ok = delta >= thresholds.general_delta_max
        axes["general_retention"] = ok
        if not ok:
            failures.append(
                f"general_delta {delta:.3f} < allowed {thresholds.general_delta_max}"
            )
    else:
        axes["general_retention"] = True
        warnings.append("no baseline general_score — retention gate skipped")

    ent = metrics.get("routing_entropy", 0.0)
    ok = ent >= thresholds.routing_entropy_min
    axes["routing_entropy"] = ok
    if not ok:
        failures.append(
            f"routing_entropy {ent:.3f} < {thresholds.routing_entropy_min}"
        )

    cv = metrics.get("load_balance_cv", 0.0)
    ok = cv <= thresholds.load_balance_cv_max
    axes["load_balance"] = ok
    if not ok:
        failures.append(
            f"load_balance_cv {cv:.3f} > {thresholds.load_balance_cv_max}"
        )

    hall = metrics.get("hallucination_rate", 0.0)
    ok = hall <= thresholds.hallucination_max
    axes["hallucination"] = ok
    if not ok:
        failures.append(
            f"hallucination_rate {hall:.3f} > {thresholds.hallucination_max}"
        )

    if thresholds.domain_depth_min is not None:
        depth = metrics.get("domain_depth", 0.0)
        ok = depth >= thresholds.domain_depth_min
        axes["domain_depth"] = ok
        if not ok:
            failures.append(
                f"domain_depth {depth:.3f} < {thresholds.domain_depth_min}"
            )

    pack_min = getattr(thresholds, "pack_eval_min", None)
    if pack_min is not None:
        pe = metrics.get("pack_eval_score", 0.0)
        ok = pe >= float(pack_min)
        axes["pack_eval"] = ok
        if not ok:
            failures.append(f"pack_eval_score {pe:.3f} < {pack_min}")

    if thresholds.safety_pass:
        safe = metrics.get("safety_pass", 1.0) >= 0.5
        axes["safety"] = safe
        if not safe:
            failures.append("safety red-team failed")

    needs_human = thresholds.require_human_approval or thresholds.high_stakes
    if needs_human:
        approved = metrics.get("human_approved", 0.0) >= 0.5
        axes["human_approval"] = approved
        if not approved:
            failures.append("human approval required but not granted")

    return GateDecision(
        passed=len(failures) == 0,
        failures=failures,
        warnings=warnings,
        axes=axes,
    )
