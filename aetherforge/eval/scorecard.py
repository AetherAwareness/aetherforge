"""
Reliability Scorecard — Stage 6 (industry-agnostic).

Domain competence is scored from:
  - eval text proxies against DomainPack keywords (any field)
  - structure / hedge / overconfidence stress
  - optional LM loss when a model is loaded
  - routing health from affinity snapshot

No industry (medical, legal, finance, …) is hard-coded here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from aetherforge.affinity.probe import AffinityResult
from aetherforge.eval.gates import GateDecision, apply_gates
from aetherforge.eval.metrics import domain_score_proxy, load_balance_cv, routing_entropy
from aetherforge.utils.config import EvalConfig
from aetherforge.utils.logging import get_logger

log = get_logger("eval.scorecard")

_HEDGE = re.compile(
    r"\b(may|might|could|possible|uncertain|differential|consider|suggests|"
    r"consistent with|cannot exclude|approx|confidence|tradeoff|hypothesis)\b",
    re.I,
)
_OVERCONFIDENT = re.compile(
    r"\b(guaranteed|100%\s*safe|never fails|definitely not|"
    r"zero risk|infallible|always works|perfectly solves)\b",
    re.I,
)
_STRUCTURE = re.compile(
    r"(^|\n)\s*([-*]|\d+\.)\s+\S|Assessment|Action plan|Risks|Failure|Decision|Tradeoff",
    re.I,
)


def _keywords_for_domain(
    domain: str,
    explicit: Optional[list[str]] = None,
) -> list[str]:
    """Build keyword list from pack keywords or domain slug tokens — never a field table."""
    if explicit:
        return [k.lower() for k in explicit if k and str(k).strip()]
    parts = re.split(r"[_\-\s]+", domain.lower())
    return [p for p in parts if len(p) > 2] or [domain.lower()]


@dataclass
class Scorecard:
    domain: str
    metrics: dict[str, float]
    gate: GateDecision
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.gate.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "gate": self.gate.to_dict(),
            "passed": self.passed,
            "details": self.details,
            # Explicit promotion semantics (criterion 4)
            "scorecard_kind": self.details.get("scorecard_kind"),
            "moe_ready": self.details.get("moe_ready"),
            "ci_complete": self.details.get("ci_complete"),
            "promotion_label": self.details.get("promotion_label"),
        }


class ReliabilityScorecard:
    def __init__(
        self,
        config: EvalConfig,
        domain: str = "general",
        domain_keywords: Optional[list[str]] = None,
    ):
        self.config = config
        self.domain = domain
        self.domain_keywords = domain_keywords

    def evaluate(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        affinity: Optional[AffinityResult] = None,
        eval_texts: Optional[list[str]] = None,
        baseline_metrics: Optional[dict[str, float]] = None,
        dry_run: bool = False,
        quality_report: Optional[dict[str, Any]] = None,
        domain_keywords: Optional[list[str]] = None,
        sector_workflow: Optional[dict[str, Any]] = None,
        pack_eval: Optional[dict[str, Any]] = None,
    ) -> Scorecard:
        eval_texts = eval_texts or []
        metrics: dict[str, float] = {}
        details: dict[str, Any] = {"dry_run": dry_run, "n_eval_texts": len(eval_texts)}

        kws = _keywords_for_domain(
            self.domain, domain_keywords or self.domain_keywords
        )
        details["keywords_used"] = kws[:24]

        if eval_texts:
            metrics["domain_score"] = domain_score_proxy(eval_texts, kws)
            metrics["structure_score"] = self._structure_score(eval_texts)
            metrics["general_score"] = self._general_proxy(eval_texts)
            metrics["hallucination_rate"] = self._hallucination_stress(eval_texts)
            metrics["hedge_rate"] = self._hedge_rate(eval_texts)
            metrics["safety_pass"] = 1.0 if metrics["hallucination_rate"] < 0.25 else 0.0
            details["mode"] = "text_proxies" + ("+dry_run_no_model" if dry_run else "")
        elif dry_run:
            metrics["domain_score"] = 0.55
            metrics["general_score"] = 0.70
            metrics["structure_score"] = 0.50
            metrics["hallucination_rate"] = 0.08
            metrics["hedge_rate"] = 0.20
            metrics["safety_pass"] = 1.0
            details["mode"] = "dry_run_no_eval_texts"
            details["warning"] = "scorecard used conservative defaults; supply eval texts"
        else:
            metrics["domain_score"] = 0.0
            metrics["general_score"] = 0.0
            metrics["structure_score"] = 0.0
            metrics["hallucination_rate"] = 1.0
            metrics["hedge_rate"] = 0.0
            metrics["safety_pass"] = 0.0
            details["mode"] = "empty_eval_fail"

        if quality_report and "diversity" in quality_report:
            div = float(quality_report["diversity"])
            metrics["data_diversity"] = div
            metrics["general_score"] = 0.7 * metrics["general_score"] + 0.3 * div

        if model is not None and tokenizer is not None and eval_texts:
            try:
                metrics["lm_loss"] = self._lm_loss(model, tokenizer, eval_texts[:32])
                metrics["general_score"] = max(
                    0.0,
                    min(
                        1.0,
                        0.5 * metrics["general_score"]
                        + 0.5 * float(np.exp(-metrics["lm_loss"])),
                    ),
                )
            except Exception as e:
                details["lm_loss_error"] = str(e)

        if affinity is not None:
            metrics["routing_entropy"] = routing_entropy(affinity.routing_freq)
            metrics["load_balance_cv"] = (
                float(affinity.load_balance_cv)
                if affinity.load_balance_cv
                else load_balance_cv(affinity.routing_freq)
            )
            details["affinity_domain"] = affinity.domain
            if affinity.metadata.get("synthetic"):
                details["affinity_synthetic"] = True
                details["affinity_watermark"] = "SYNTHETIC — not measured on weights"
        else:
            metrics["routing_entropy"] = 0.0 if not dry_run else 1.5
            metrics["load_balance_cv"] = 9.0 if not dry_run else 0.5

        # MoE sector-workflow metrics when sequential forge ran
        if sector_workflow:
            details["sector_workflow"] = True
            metrics["sectors_trained"] = float(sector_workflow.get("n_trained") or 0)
            metrics["sectors_blocked"] = float(sector_workflow.get("n_blocked") or 0)
            metrics["sectors_rolled_back"] = float(
                sector_workflow.get("n_rolled_back") or 0
            )
            inter = sector_workflow.get("interference") or {}
            metrics["interference_regressions"] = float(
                inter.get("n_regressions") or 0
            )
            # readiness: fraction of train sectors that kept
            n_sec = max(
                1,
                int(sector_workflow.get("n_trained") or 0)
                + int(sector_workflow.get("n_blocked") or 0)
                + int(sector_workflow.get("n_skipped") or 0)
                + int(sector_workflow.get("n_rolled_back") or 0),
            )
            metrics["sector_keep_rate"] = float(
                sector_workflow.get("n_trained") or 0
            ) / n_sec

        if pack_eval:
            details["pack_eval"] = {
                k: pack_eval.get(k)
                for k in ("schema", "score", "n", "n_scored", "mode", "dry_run")
            }
            if pack_eval.get("n"):
                metrics["pack_eval_score"] = float(pack_eval.get("score") or 0.0)
                details["pack_eval_mode"] = pack_eval.get("mode")

        thr = self.config.scorecard_thresholds
        # Generic domain-depth axis (any industry)
        if thr.domain_depth_min is not None:
            metrics["domain_depth"] = (
                metrics.get("domain_score", 0.0)
                * (0.5 + 0.5 * metrics.get("structure_score", 0.0))
                * (1.0 - metrics.get("hallucination_rate", 0.0))
            )

        baseline = baseline_metrics or {
            "general_score": metrics.get("general_score", 0.8),
        }
        gate = apply_gates(metrics, thr, baseline=baseline)

        needs_human = thr.require_human_approval or thr.high_stakes
        if needs_human and dry_run:
            gate.warnings.append(
                "human_approval still required before production promote (high-stakes domain)"
            )
            metrics["human_approved"] = 0.0
            gate = apply_gates(metrics, thr, baseline=baseline)
            if not gate.passed and any("human" in f.lower() for f in gate.failures):
                details["promote_blocked_reason"] = "human_approval_required"

        # ── CI completeness vs MoE reliability labeling ─────────────────
        affinity_synth = bool(details.get("affinity_synthetic"))
        has_model = model is not None and not dry_run
        has_sector = bool(sector_workflow)
        ci_complete = bool(gate.passed)  # pipeline/data/proxy gates
        # MoE-ready only with non-synthetic routing + live model (+ sector keep if wave ran)
        moe_ready = False
        if has_model and not affinity_synth and gate.passed:
            if has_sector:
                moe_ready = (
                    float(metrics.get("sector_keep_rate") or 0) > 0
                    and float(metrics.get("interference_regressions") or 0) == 0
                )
            else:
                moe_ready = True
        if dry_run:
            scorecard_kind = "ci_completeness"
            promotion_label = (
                "DRY-RUN CI COMPLETE — not MoE weight-level readiness"
                if ci_complete
                else "DRY-RUN CI FAILED"
            )
            if affinity_synth:
                promotion_label += " · affinity SYNTHETIC"
            gate.warnings.append(
                "Dry-run scorecard is CI completeness only; do not treat as MoE promotion."
            )
        elif moe_ready:
            scorecard_kind = "moe_reliability"
            promotion_label = "MOE RELIABILITY PASS — eligible for package promote"
        else:
            scorecard_kind = "ci_completeness" if ci_complete else "failed"
            promotion_label = (
                "CI COMPLETE but MoE readiness incomplete "
                f"(model={has_model}, synthetic_affinity={affinity_synth}, "
                f"sector_wave={has_sector})"
                if ci_complete
                else "SCORECARD FAILED"
            )
            if ci_complete and not moe_ready:
                gate.warnings.append(promotion_label)

        details["scorecard_kind"] = scorecard_kind
        details["ci_complete"] = ci_complete
        details["moe_ready"] = moe_ready
        details["promotion_label"] = promotion_label
        # Dry-run must never claim full MoE promote readiness
        details["full_moe_promoted_readiness"] = bool(moe_ready and not dry_run)

        sc = Scorecard(
            domain=self.domain,
            metrics={k: float(v) for k, v in metrics.items()},
            gate=gate,
            baseline_metrics=baseline,
            details=details,
        )
        log.info(
            "Scorecard domain=%s kind=%s ci=%s moe=%s label=%s metrics=%s failures=%s",
            self.domain,
            scorecard_kind,
            ci_complete,
            moe_ready,
            promotion_label,
            {k: round(v, 4) for k, v in sc.metrics.items()},
            gate.failures,
        )
        return sc

    def _structure_score(self, texts: list[str]) -> float:
        if not texts:
            return 0.0
        return sum(1 for t in texts if _STRUCTURE.search(t)) / len(texts)

    def _general_proxy(self, texts: list[str]) -> float:
        if not texts:
            return 0.0
        lengths = [len(t.split()) for t in texts]
        mean_len = sum(lengths) / len(lengths)
        length_score = 1.0 - min(1.0, abs(mean_len - 120) / 200)
        uniq = len({t[:200] for t in texts}) / max(len(texts), 1)
        return float(0.6 * length_score + 0.4 * uniq)

    def _hedge_rate(self, texts: list[str]) -> float:
        if not texts:
            return 0.0
        return sum(1 for t in texts if _HEDGE.search(t)) / len(texts)

    def _hallucination_stress(self, texts: list[str]) -> float:
        """Overconfidence stress proxy — field-agnostic."""
        if not texts:
            return 1.0
        flags = 0.0
        n = min(len(texts), 200)
        for t in texts[:n]:
            if _OVERCONFIDENT.search(t):
                flags += 1
            if re.search(r"\b(must|always|never)\b", t, re.I) and not _HEDGE.search(t):
                flags += 0.25
        return min(1.0, flags / max(n, 1))

    def _lm_loss(self, model: Any, tokenizer: Any, texts: list[str]) -> float:
        import torch

        model.eval()
        device = next(model.parameters()).device
        losses = []
        with torch.no_grad():
            for t in texts:
                enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=256)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc, labels=enc["input_ids"])
                losses.append(float(out.loss.cpu()))
        return float(np.mean(losses)) if losses else float("inf")
