"""
Sequential per-sector training workflow.

For each train-enabled MoE sector:
  0. Freeze plan fingerprint (immutable membership for the wave)
  1. Forensic inventory + evidence tier
  2. Readiness gate
  3. Sector dataset + data contract
  4. Pre-probe routing mass
  5. ESFT (or dry-run) — only that sector's experts
  6. Post-probe → keep/rollback
  7. Interference summary across sectors
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aetherforge.affinity.expert_selector import SelectionPlan
from aetherforge.data.sector_datasets import SectorDatasetForge, SectorDatasetPlan
from aetherforge.groups.models import ExpertGroup, GroupPlan
from aetherforge.groups.plan_fingerprint import (
    freeze_plan,
    plan_fingerprint,
    save_freeze,
    verify_plan_fingerprint,
)
from aetherforge.groups.readiness import readiness_markdown, run_forensics_gate
from aetherforge.groups.store import save_group_plan
from aetherforge.groups.studio import selection_for_group
from aetherforge.models.loaders import MoEModelBundle
from aetherforge.training.esft_trainer import ESFTResult, ESFTTrainer
from aetherforge.training.sector_probe import (
    decide_keep_rollback,
    interference_summary,
    probe_sector,
    synthetic_post_boost,
    _matrix_from_affinity,
)
from aetherforge.utils.audit import AuditLog
from aetherforge.utils.config import DataConfig, TrainingConfig
from aetherforge.utils.logging import get_logger
from aetherforge.viz.progress import LiveProgress

log = get_logger("training.sector_workflow")


@dataclass
class SectorTrainResult:
    group_id: str
    name: str
    status: str  # trained | skipped | blocked | dry_run | error | rolled_back
    readiness_status: str = "pass"
    n_train: int = 0
    n_experts: int = 0
    forensics_summary: str = ""
    evidence_tier: str = "structure_only"
    esft: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    paths: dict[str, str] = field(default_factory=dict)
    duration_sec: float = 0.0
    plan_fingerprint: str = ""
    cells_fingerprint: str = ""
    pre_probe: Optional[dict[str, Any]] = None
    post_probe: Optional[dict[str, Any]] = None
    keep_rollback: Optional[dict[str, Any]] = None
    data_contract: Optional[dict[str, Any]] = None
    decision: str = "keep"  # keep | rollback | skip | block

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "status": self.status,
            "readiness_status": self.readiness_status,
            "n_train": self.n_train,
            "n_experts": self.n_experts,
            "forensics_summary": self.forensics_summary,
            "evidence_tier": self.evidence_tier,
            "esft": self.esft,
            "error": self.error,
            "paths": self.paths,
            "duration_sec": self.duration_sec,
            "plan_fingerprint": self.plan_fingerprint,
            "cells_fingerprint": self.cells_fingerprint,
            "pre_probe": self.pre_probe,
            "post_probe": self.post_probe,
            "keep_rollback": self.keep_rollback,
            "data_contract": self.data_contract,
            "decision": self.decision,
        }


@dataclass
class SectorWorkflowResult:
    mode: str
    readiness: dict[str, Any]
    datasets: dict[str, Any]
    sectors: list[SectorTrainResult]
    n_trained: int = 0
    n_skipped: int = 0
    n_blocked: int = 0
    n_rolled_back: int = 0
    duration_sec: float = 0.0
    paths: dict[str, str] = field(default_factory=dict)
    plan_freeze: Optional[dict[str, Any]] = None
    interference: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aetherforge.sector_workflow.v2",
            "mode": self.mode,
            "n_trained": self.n_trained,
            "n_skipped": self.n_skipped,
            "n_blocked": self.n_blocked,
            "n_rolled_back": self.n_rolled_back,
            "duration_sec": self.duration_sec,
            "paths": self.paths,
            "plan_freeze": self.plan_freeze,
            "interference": self.interference,
            "readiness": self.readiness,
            "datasets": self.datasets,
            "sectors": [s.to_dict() for s in self.sectors],
        }


class SectorWorkflow:
    """Orchestrate forensics → datasets → sequential ESFT with probe/keep-rollback."""

    def __init__(
        self,
        group_plan: GroupPlan,
        train_cfg: TrainingConfig,
        data_cfg: DataConfig,
        *,
        bundle: Optional[MoEModelBundle] = None,
        affinity: Optional[dict[str, Any]] = None,
        audit: Optional[AuditLog] = None,
        dry_run: bool = False,
        gate_mode: str = "warn",
        auto_bind: bool = True,
        min_samples: int = 8,
        shared_fraction: float = 0.15,
        continue_on_block: bool = True,
        progress: Optional[LiveProgress] = None,
    ):
        self.group_plan = group_plan
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg
        self.bundle = bundle
        self.affinity = affinity
        self.audit = audit
        self.dry_run = dry_run
        self.gate_mode = gate_mode
        self.auto_bind = auto_bind
        self.min_samples = min_samples
        self.shared_fraction = shared_fraction
        self.continue_on_block = continue_on_block
        self.progress = progress

    def run(
        self,
        train_records: list[dict[str, Any]],
        output_dir: str | Path,
        *,
        eval_texts: Optional[list[str]] = None,
        only_group_ids: Optional[list[str]] = None,
    ) -> SectorWorkflowResult:
        t0 = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plan = self.group_plan
        tcfg = self.train_cfg

        # ── 0. Freeze plan fingerprint ─────────────────────────────────
        freeze = freeze_plan(plan, notes="sector_wave_start")
        save_freeze(freeze, output_dir / "plan_freeze.json")
        (output_dir / "plan_fingerprint.txt").write_text(
            freeze.fingerprint + "\n", encoding="utf-8"
        )
        if self.audit:
            self.audit.record(
                "sector_workflow",
                "plan_freeze",
                {"fingerprint": freeze.fingerprint, "n_train": len(freeze.train_group_ids)},
            )

        # ── 1. Forensic readiness ──────────────────────────────────────
        log.info("Sector workflow: forensic readiness (mode=%s)", self.gate_mode)
        readiness = run_forensics_gate(
            plan,
            affinity=self.affinity,
            mode=self.gate_mode,
            auto_bind=self.auto_bind,
            global_domain=self.data_cfg.domain,
            only_train_groups=True,
        )
        save_group_plan(plan, output_dir / "expert_groups.bound.json")
        with open(output_dir / "sector_readiness.json", "w", encoding="utf-8") as f:
            json.dump(readiness.to_dict(), f, indent=2, default=str)
        (output_dir / "sector_readiness.md").write_text(
            readiness_markdown(readiness), encoding="utf-8"
        )

        # Re-freeze after auto-bind (bindings change description, not cells —
        # membership payload includes domain; re-freeze so train uses post-bind fp)
        freeze = freeze_plan(plan, notes="sector_wave_post_bind")
        save_freeze(freeze, output_dir / "plan_freeze.json")
        (output_dir / "plan_fingerprint.txt").write_text(
            freeze.fingerprint + "\n", encoding="utf-8"
        )

        if readiness.overall == "block" and not self.continue_on_block:
            result = SectorWorkflowResult(
                mode="sequential",
                readiness=readiness.to_dict(),
                datasets={},
                sectors=[],
                n_blocked=readiness.n_block,
                duration_sec=time.time() - t0,
                plan_freeze=freeze.to_dict(),
                paths={"readiness": str(output_dir / "sector_readiness.json")},
            )
            with open(output_dir / "sector_workflow.json", "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            raise RuntimeError(
                f"Forensic readiness blocked training: {readiness.narrative}"
            )

        ready_by_id = {s.group_id: s for s in readiness.sectors}
        forensics_by_id = {
            s.group_id: s.forensics for s in readiness.sectors if s.forensics
        }

        # ── 2. Per-sector datasets + contracts ─────────────────────────
        log.info("Sector workflow: building per-sector datasets + contracts")
        forge = SectorDatasetForge(
            self.data_cfg,
            min_match=float(getattr(self.data_cfg, "sector_min_match", 0.18) or 0.18),
            shared_fraction=self.shared_fraction,
            min_samples=self.min_samples,
            synthesize_fill=True,
            contract_mode=str(getattr(tcfg, "sector_contract_mode", "warn") or "warn"),
            min_real_fraction=float(getattr(tcfg, "sector_min_real_fraction", 0.15) or 0.15),
            max_synth_fraction=float(getattr(tcfg, "sector_max_synth_fraction", 0.85) or 0.85),
            min_unique_ratio=float(getattr(tcfg, "sector_min_unique_ratio", 0.35) or 0.35),
        )
        targets = plan.enabled_train_groups()
        if only_group_ids:
            allow = set(only_group_ids)
            targets = [g for g in targets if g.id in allow]

        ds_plan = forge.build(
            plan,
            train_records,
            output_dir=output_dir / "sector_datasets",
            forensics_by_id=forensics_by_id,
            eval_texts=eval_texts,
            groups=targets,
        )

        readiness2 = run_forensics_gate(
            plan,
            affinity=self.affinity,
            mode=self.gate_mode,
            auto_bind=False,
            global_domain=self.data_cfg.domain,
            only_train_groups=True,
            min_sector_samples=self.min_samples,
            sector_sample_counts=ds_plan.sample_counts(),
        )
        with open(
            output_dir / "sector_readiness_post_data.json", "w", encoding="utf-8"
        ) as f:
            json.dump(readiness2.to_dict(), f, indent=2, default=str)
        ready_by_id = {s.group_id: s for s in readiness2.sectors}

        # ── 3. Sequential ESFT + probe ─────────────────────────────────
        sector_results: list[SectorTrainResult] = []
        n_trained = n_skipped = n_blocked = n_rolled_back = 0
        ckpt_root = output_dir / "checkpoints"
        ckpt_root.mkdir(parents=True, exist_ok=True)

        # Track post matrices for interference (start from pre affinity)
        pre_matrix = _matrix_from_affinity(self.affinity)
        post_matrix = None
        if pre_matrix is not None:
            import copy

            post_matrix = copy.deepcopy(pre_matrix)

        if self.progress:
            self.progress.begin_sector_wave(n_total=len(targets), mode="sequential")

        for idx, g in enumerate(targets):
            # Immutable membership check before each sector
            check = verify_plan_fingerprint(plan, freeze.fingerprint)
            if not check["ok"]:
                log.error(check["message"])
                if not self.continue_on_block:
                    raise RuntimeError(check["message"])
                sr = SectorTrainResult(
                    group_id=g.id,
                    name=g.name,
                    status="blocked",
                    readiness_status="fail",
                    error=check["message"],
                    plan_fingerprint=freeze.fingerprint,
                    decision="block",
                )
                sector_results.append(sr)
                n_blocked += 1
                continue

            rdy = ready_by_id.get(g.id)
            shard = ds_plan.shard_for(g.id)
            fre = forensics_by_id.get(g.id)
            summary = ""
            tier = "structure_only"
            if fre:
                summary = (fre.get("content") or {}).get("summary") or ""
                tier = fre.get("evidence_tier") or (
                    fre.get("content") or {}
                ).get("evidence_tier") or "structure_only"
            n_samp = len(shard.train_records) if shard else 0

            if self.progress:
                self.progress.sector_start(
                    group_id=g.id,
                    name=g.name,
                    index=idx,
                    n_total=len(targets),
                    forensics_summary=summary,
                    readiness=rdy.status if rdy else "pass",
                    n_experts=len(g.cells),
                    n_samples=n_samp,
                    color=g.color,
                    domain=g.domain,
                )

            sr = self._train_one_sector(
                g,
                readiness=rdy,
                shard=shard,
                forensics=fre,
                ckpt_root=ckpt_root,
                plan_fp=freeze.fingerprint,
                pre_matrix=pre_matrix,
            )

            # Cumulative post matrix for interference (kept sectors only)
            if (
                post_matrix is not None
                and sr.decision == "keep"
                and sr.status in ("trained", "dry_run")
            ):
                boost = float(getattr(tcfg, "sector_probe_dry_boost", 0.08) or 0.08)
                post_matrix = synthetic_post_boost(post_matrix, g, boost=boost)

            sector_results.append(sr)
            if self.progress:
                self.progress.sector_end(
                    group_id=g.id,
                    name=g.name,
                    status=sr.status,
                    readiness=sr.readiness_status,
                    n_experts=sr.n_experts,
                    n_samples=sr.n_train,
                    forensics_summary=sr.forensics_summary,
                    error=sr.error,
                    duration_sec=sr.duration_sec,
                    color=g.color,
                )

            if sr.status in ("trained", "dry_run") and sr.decision == "keep":
                n_trained += 1
            elif sr.status == "rolled_back" or sr.decision == "rollback":
                n_rolled_back += 1
            elif sr.status == "blocked":
                n_blocked += 1
            else:
                n_skipped += 1

        if self.progress:
            self.progress.end_sector_wave(overall="done")

        # Interference summary
        inter = None
        if pre_matrix is not None and post_matrix is not None:
            aff_pre = dict(self.affinity or {})
            aff_pre["affinity"] = pre_matrix
            aff_post = dict(self.affinity or {})
            aff_post["affinity"] = post_matrix
            aff_post["metadata"] = {
                **(aff_post.get("metadata") or {}),
                "post_sector_wave": True,
            }
            inter = interference_summary(
                plan,
                aff_pre,
                aff_post,
                trained_group_ids=[
                    s.group_id
                    for s in sector_results
                    if s.decision == "keep" and s.status in ("trained", "dry_run")
                ],
            )
            with open(output_dir / "interference.json", "w", encoding="utf-8") as f:
                json.dump(inter, f, indent=2)
            (output_dir / "interference.md").write_text(
                (inter.get("narrative") or "")
                + "\n\n"
                + "\n".join(
                    f"- {r['name']}: Δ={r['delta_mean_share']:+.4f}"
                    f"{' ⚠ regress' if r.get('regressed') else ''}"
                    for r in inter.get("sectors") or []
                ),
                encoding="utf-8",
            )

        combined = self._combined_selection(targets, ready_by_id, sector_results)
        with open(output_dir / "selection_plan.sectors.json", "w", encoding="utf-8") as f:
            json.dump(combined.to_dict(), f, indent=2, default=str)

        result = SectorWorkflowResult(
            mode="sequential",
            readiness=readiness2.to_dict(),
            datasets=ds_plan.to_dict(),
            sectors=sector_results,
            n_trained=n_trained,
            n_skipped=n_skipped,
            n_blocked=n_blocked,
            n_rolled_back=n_rolled_back,
            duration_sec=time.time() - t0,
            plan_freeze=freeze.to_dict(),
            interference=inter,
            paths={
                "root": str(output_dir),
                "readiness": str(output_dir / "sector_readiness.json"),
                "readiness_post_data": str(
                    output_dir / "sector_readiness_post_data.json"
                ),
                "datasets": str(
                    output_dir / "sector_datasets" / "sector_dataset_plan.json"
                ),
                "selection": str(output_dir / "selection_plan.sectors.json"),
                "workflow": str(output_dir / "sector_workflow.json"),
                "plan_freeze": str(output_dir / "plan_freeze.json"),
                "plan_fingerprint": str(output_dir / "plan_fingerprint.txt"),
                "interference": str(output_dir / "interference.json")
                if inter
                else "",
            },
        )
        with open(output_dir / "sector_workflow.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        (output_dir / "sector_workflow.md").write_text(
            self._markdown(result), encoding="utf-8"
        )
        result.paths["markdown"] = str(output_dir / "sector_workflow.md")

        if self.audit:
            self.audit.record(
                "sector_workflow",
                "complete",
                {
                    "n_trained": n_trained,
                    "n_skipped": n_skipped,
                    "n_blocked": n_blocked,
                    "n_rolled_back": n_rolled_back,
                    "fingerprint": freeze.fingerprint,
                    "duration_sec": result.duration_sec,
                },
            )
        log.info(
            "Sector workflow done: trained=%d skipped=%d blocked=%d rolled_back=%d fp=%s",
            n_trained,
            n_skipped,
            n_blocked,
            n_rolled_back,
            freeze.fingerprint[:12],
        )
        return result

    def _train_one_sector(
        self,
        group: ExpertGroup,
        *,
        readiness: Any,
        shard: Any,
        forensics: Optional[dict[str, Any]],
        ckpt_root: Path,
        plan_fp: str,
        pre_matrix: Optional[list[list[float]]],
    ) -> SectorTrainResult:
        from aetherforge.groups.plan_fingerprint import group_cells_fingerprint

        t0 = time.time()
        tcfg = self.train_cfg
        sector_dir = ckpt_root / group.id
        sector_dir.mkdir(parents=True, exist_ok=True)

        r_status = readiness.status if readiness else "pass"
        summary = ""
        tier = "structure_only"
        if forensics:
            summary = (forensics.get("content") or {}).get("summary") or ""
            tier = forensics.get("evidence_tier") or (
                forensics.get("content") or {}
            ).get("evidence_tier") or "structure_only"
        elif readiness and readiness.forensics:
            summary = (readiness.forensics.get("content") or {}).get("summary") or ""
            tier = readiness.forensics.get("evidence_tier") or "structure_only"

        contract = (shard.contract if shard else None) or {}
        train_eligible_contract = True
        if shard is not None:
            train_eligible_contract = bool(getattr(shard, "train_eligible", True))
            contract = shard.contract or contract

        # Gate: readiness block or contract block
        if readiness and readiness.status == "block":
            res = SectorTrainResult(
                group_id=group.id,
                name=group.name,
                status="blocked",
                readiness_status=r_status,
                forensics_summary=summary,
                evidence_tier=tier,
                plan_fingerprint=plan_fp,
                cells_fingerprint=group_cells_fingerprint(group),
                data_contract=contract,
                decision="block",
                paths={"dir": str(sector_dir)},
                duration_sec=time.time() - t0,
            )
            with open(sector_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(res.to_dict(), f, indent=2)
            return res

        if not train_eligible_contract:
            res = SectorTrainResult(
                group_id=group.id,
                name=group.name,
                status="blocked",
                readiness_status=r_status,
                forensics_summary=summary,
                evidence_tier=tier,
                plan_fingerprint=plan_fp,
                cells_fingerprint=group_cells_fingerprint(group),
                data_contract=contract,
                decision="block",
                error="data_contract_failed: " + "; ".join(
                    (contract or {}).get("violations") or ["contract fail"]
                ),
                paths={"dir": str(sector_dir)},
                duration_sec=time.time() - t0,
            )
            with open(sector_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(res.to_dict(), f, indent=2)
            return res

        if not group.train or group.freeze or not group.enabled:
            res = SectorTrainResult(
                group_id=group.id,
                name=group.name,
                status="skipped",
                readiness_status=r_status,
                forensics_summary=summary,
                evidence_tier=tier,
                plan_fingerprint=plan_fp,
                decision="skip",
                paths={"dir": str(sector_dir)},
                duration_sec=time.time() - t0,
            )
            with open(sector_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(res.to_dict(), f, indent=2)
            return res

        records = list(shard.train_records) if shard else []
        n_train = len(records)
        sel = selection_for_group(
            self.group_plan,
            group.id,
            domain=group.domain or self.data_cfg.domain,
        )
        with open(sector_dir / "selection_plan.json", "w", encoding="utf-8") as f:
            json.dump(sel.to_dict(), f, indent=2)
        if forensics:
            with open(sector_dir / "forensics.json", "w", encoding="utf-8") as f:
                json.dump(forensics, f, indent=2, default=str)
        if readiness:
            with open(sector_dir / "readiness.json", "w", encoding="utf-8") as f:
                json.dump(readiness.to_dict(), f, indent=2, default=str)
        if contract:
            with open(sector_dir / "data_contract.json", "w", encoding="utf-8") as f:
                json.dump(contract, f, indent=2, default=str)

        # Pre-probe
        pre_probe_d = None
        if getattr(tcfg, "sector_probe_enabled", True):
            pre = probe_sector(
                self.group_plan,
                group,
                self.affinity,
                phase="pre",
                matrix_override=pre_matrix,
            )
            pre_probe_d = pre.to_dict()
            with open(sector_dir / "probe_pre.json", "w", encoding="utf-8") as f:
                json.dump(pre_probe_d, f, indent=2)

        card = self._sector_card(
            group, summary, readiness, n_train, len(sel.selected), tier, plan_fp
        )
        (sector_dir / "PRE_TRAIN_FORENSICS.md").write_text(card, encoding="utf-8")
        log.info(
            "=== PRE-TRAIN FORENSICS [%s]: %s ===\n%s",
            tier,
            group.name,
            summary or "(structure only)",
        )

        if self.dry_run or self.bundle is None:
            dry = {
                "dry_run": True,
                "group_id": group.id,
                "name": group.name,
                "n_experts": len(sel.selected),
                "n_train": n_train,
                "readiness": r_status,
                "evidence_tier": tier,
                "forensics_summary": summary,
                "plan_fingerprint": plan_fp,
            }
            esft_dir = sector_dir / "esft"
            esft_dir.mkdir(parents=True, exist_ok=True)
            with open(esft_dir / "esft_result.json", "w", encoding="utf-8") as f:
                json.dump(dry, f, indent=2)
            # post probe with synthetic boost for dry-run
            post_probe_d = None
            kr = None
            decision = "keep"
            status = "dry_run"
            if pre_probe_d and getattr(tcfg, "sector_probe_enabled", True):
                mat = pre_matrix
                if mat is not None:
                    boosted = synthetic_post_boost(
                        mat,
                        group,
                        boost=float(
                            getattr(tcfg, "sector_probe_dry_boost", 0.08) or 0.08
                        ),
                    )
                    post = probe_sector(
                        self.group_plan,
                        group,
                        self.affinity,
                        phase="post",
                        matrix_override=boosted,
                    )
                    post_probe_d = post.to_dict()
                    with open(sector_dir / "probe_post.json", "w", encoding="utf-8") as f:
                        json.dump(post_probe_d, f, indent=2)
                    from aetherforge.training.sector_probe import SectorProbeResult

                    pre_obj = SectorProbeResult(
                        group_id=group.id,
                        name=group.name,
                        phase="pre",
                        mean_share=float(pre_probe_d.get("mean_share") or 0),
                        total_share=float(pre_probe_d.get("total_share") or 0),
                        n_layers_hit=float(pre_probe_d.get("n_layers_hit") or 0),
                        synthetic_affinity=bool(pre_probe_d.get("synthetic_affinity")),
                    )
                    delta = decide_keep_rollback(
                        pre_obj,
                        post,
                        min_delta=float(
                            getattr(tcfg, "sector_probe_min_delta", -0.02) or -0.02
                        ),
                    )
                    kr = delta.to_dict()
                    with open(
                        sector_dir / "keep_rollback.json", "w", encoding="utf-8"
                    ) as f:
                        json.dump(kr, f, indent=2)
                    if getattr(tcfg, "sector_keep_rollback", True) and not delta.keep:
                        decision = "rollback"
                        status = "rolled_back"

            res = SectorTrainResult(
                group_id=group.id,
                name=group.name,
                status=status,
                readiness_status=r_status,
                n_train=n_train,
                n_experts=len(sel.selected),
                forensics_summary=summary,
                evidence_tier=tier,
                esft=dry,
                plan_fingerprint=plan_fp,
                cells_fingerprint=group_cells_fingerprint(group),
                pre_probe=pre_probe_d,
                post_probe=post_probe_d,
                keep_rollback=kr,
                data_contract=contract,
                decision=decision,
                paths={
                    "dir": str(sector_dir),
                    "selection": str(sector_dir / "selection_plan.json"),
                    "pre_train": str(sector_dir / "PRE_TRAIN_FORENSICS.md"),
                    "esft": str(esft_dir),
                },
                duration_sec=time.time() - t0,
            )
            with open(sector_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(res.to_dict(), f, indent=2)
            if self.audit:
                self.audit.record("sector_esft", status, res.to_dict())
            return res

        try:
            trainer = ESFTTrainer(
                self.bundle, self.train_cfg, sel, audit=self.audit
            )
            esft: ESFTResult = trainer.train(
                records,
                sector_dir / "esft",
                eval_texts=shard.eval_texts if shard else None,
            )
            # post probe (live path uses same affinity unless caller updates)
            post_probe_d = None
            kr = None
            decision = "keep"
            status = "trained"
            if pre_probe_d and getattr(tcfg, "sector_probe_enabled", True):
                post = probe_sector(
                    self.group_plan, group, self.affinity, phase="post"
                )
                post_probe_d = post.to_dict()
                from aetherforge.training.sector_probe import SectorProbeResult

                pre_obj = SectorProbeResult(
                    group_id=group.id,
                    name=group.name,
                    phase="pre",
                    mean_share=float(pre_probe_d.get("mean_share") or 0),
                    total_share=float(pre_probe_d.get("total_share") or 0),
                    n_layers_hit=float(pre_probe_d.get("n_layers_hit") or 0),
                )
                delta = decide_keep_rollback(
                    pre_obj,
                    post,
                    min_delta=float(
                        getattr(tcfg, "sector_probe_min_delta", -0.02) or -0.02
                    ),
                )
                kr = delta.to_dict()
                if getattr(tcfg, "sector_keep_rollback", True) and not delta.keep:
                    decision = "rollback"
                    status = "rolled_back"
            res = SectorTrainResult(
                group_id=group.id,
                name=group.name,
                status=status,
                readiness_status=r_status,
                n_train=n_train,
                n_experts=len(sel.selected),
                forensics_summary=summary,
                evidence_tier=tier,
                esft=esft.to_dict(),
                plan_fingerprint=plan_fp,
                cells_fingerprint=group_cells_fingerprint(group),
                pre_probe=pre_probe_d,
                post_probe=post_probe_d,
                keep_rollback=kr,
                data_contract=contract,
                decision=decision,
                paths={
                    "dir": str(sector_dir),
                    "esft": str(sector_dir / "esft"),
                    "pre_train": str(sector_dir / "PRE_TRAIN_FORENSICS.md"),
                },
                duration_sec=time.time() - t0,
            )
        except Exception as e:
            log.exception("Sector %s training failed: %s", group.name, e)
            res = SectorTrainResult(
                group_id=group.id,
                name=group.name,
                status="error",
                readiness_status=r_status,
                n_train=n_train,
                n_experts=len(sel.selected),
                forensics_summary=summary,
                evidence_tier=tier,
                error=str(e),
                plan_fingerprint=plan_fp,
                decision="skip",
                paths={"dir": str(sector_dir)},
                duration_sec=time.time() - t0,
            )

        with open(sector_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, indent=2, default=str)
        if self.audit:
            self.audit.record("sector_esft", res.status, res.to_dict())
        return res

    def _combined_selection(
        self,
        targets: list[ExpertGroup],
        ready_by_id: dict[str, Any],
        sector_results: list[SectorTrainResult],
    ) -> SelectionPlan:
        from aetherforge.models.moe_utils import ExpertRef

        kept = {
            s.group_id
            for s in sector_results
            if s.decision == "keep" and s.status in ("trained", "dry_run")
        }
        selected: list[ExpertRef] = []
        seen: set[tuple[int, int]] = set()
        group_ids = []
        for g in targets:
            if g.id not in kept:
                continue
            group_ids.append(g.id)
            for c in g.cells:
                k = (c.layer, c.expert)
                if k in seen:
                    continue
                seen.add(k)
                selected.append(
                    ExpertRef(
                        layer_idx=c.layer,
                        expert_idx=c.expert,
                        module_name=f"model.layers.{c.layer}.mlp.experts.{c.expert}",
                        family=self.group_plan.family,
                    )
                )
        frozen = []
        cap = self.group_plan.capacity
        for li in range(cap.num_layers):
            for ei in range(cap.num_experts):
                if (li, ei) not in seen:
                    frozen.append(
                        ExpertRef(
                            layer_idx=li,
                            expert_idx=ei,
                            module_name=f"model.layers.{li}.mlp.experts.{ei}",
                            family=self.group_plan.family,
                        )
                    )
        return SelectionPlan(
            domain=self.data_cfg.domain,
            selected=selected,
            frozen=frozen,
            ranked_scores=[(e.layer_idx, e.expert_idx, 1.0) for e in selected],
            freeze_router=True,
            metadata={
                "source": "sector_workflow",
                "mode": "sequential",
                "group_ids": group_ids,
                "n_cells": len(selected),
                "kept_only": True,
            },
        )

    def _sector_card(
        self,
        group: ExpertGroup,
        summary: str,
        readiness: Any,
        n_train: int,
        n_experts: int,
        tier: str,
        plan_fp: str,
    ) -> str:
        lines = [
            f"# Pre-train forensics — {group.name}",
            "",
            f"**Group id:** `{group.id}`",
            f"**Evidence tier:** `{tier}`",
            f"**Plan fingerprint:** `{plan_fp[:24]}…`",
            f"**Domain:** `{group.domain or '—'}`",
            f"**Experts to train:** {n_experts}",
            f"**Dataset size:** {n_train}",
            f"**Readiness:** {readiness.status if readiness else 'n/a'} "
            f"(score {readiness.score if readiness else '—'})",
            "",
            "## What this sector contains",
            "",
            summary or "_structure_only — no content identity claim._",
            "",
        ]
        if group.topics:
            lines.append("**Topics:** " + ", ".join(group.topics[:12]))
        if group.keywords:
            lines.append("**Keywords:** " + ", ".join(group.keywords[:16]))
        if readiness and readiness.reasons:
            lines += ["", "## Gate notes", ""]
            for r in readiness.reasons:
                lines.append(f"- {r}")
        lines += [
            "",
            "## Training posture",
            "",
            "- Only this sector's experts receive gradients; siblings frozen.",
            "- Plan membership frozen at wave start (fingerprint checked).",
            "- Pre/post probe decides keep vs rollback.",
            "",
            "---",
            f"*AetherForge sector workflow · {time.strftime('%Y-%m-%d %H:%M')}*",
        ]
        return "\n".join(lines)

    def _markdown(self, result: SectorWorkflowResult) -> str:
        fp = (result.plan_freeze or {}).get("fingerprint", "")[:24]
        lines = [
            "# Sector training workflow",
            "",
            f"- Mode: **{result.mode}**",
            f"- Plan fingerprint: `{fp}…`",
            f"- Trained: **{result.n_trained}** · Skipped: **{result.n_skipped}** · "
            f"Blocked: **{result.n_blocked}** · Rolled back: **{result.n_rolled_back}**",
            f"- Duration: {result.duration_sec:.1f}s",
            "",
            "| Sector | Status | Decision | Tier | Experts | Samples | Readiness |",
            "|--------|--------|----------|------|---------|---------|-----------|",
        ]
        for s in result.sectors:
            lines.append(
                f"| {s.name} | {s.status} | {s.decision} | {s.evidence_tier} | "
                f"{s.n_experts} | {s.n_train} | {s.readiness_status} |"
            )
        if result.interference:
            lines += ["", "## Interference", "", result.interference.get("narrative") or ""]
        lines += ["", "## Per-sector forensics (pre-train)", ""]
        for s in result.sectors:
            lines.append(f"### {s.name}")
            lines.append(f"Tier: `{s.evidence_tier}`")
            lines.append(s.forensics_summary or "_n/a_")
            lines.append("")
        return "\n".join(lines)
