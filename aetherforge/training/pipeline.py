"""
End-to-end AetherForge training pipeline orchestrator.

Stages 0→7 with gates, audit, promotion directory, and lifecycle plans.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from aetherforge.affinity.expert_selector import ExpertSelector
from aetherforge.affinity.probe import AffinityProbe
from aetherforge.data.forge import DataForge
from aetherforge.eval.scorecard import ReliabilityScorecard
from aetherforge.lifecycle.birth_death import ExpertLifecycleManager
from aetherforge.lifecycle.monitor import ExpertUtilizationMonitor
from aetherforge.models.loaders import load_moe_model
from aetherforge.packaging.aetherpackage import AetherPackageBuilder
from aetherforge.training.esft_trainer import ESFTTrainer
from aetherforge.training.preference import PreferenceAligner, PreferencePair
from aetherforge.training.router_hygiene import RouterHygieneTrainer
from aetherforge.utils.audit import AuditLog
from aetherforge.utils.config import AetherForgeConfig, dump_config
from aetherforge.utils.logging import get_logger, setup_logging
from aetherforge.utils.seed import set_global_seed
from aetherforge.viz.progress import LiveProgress

log = get_logger("training.pipeline")

# Canonical stage order
STAGE_ORDER = [
    "diagnostics",
    "data",
    "affinity",
    "groups",
    "esft",
    "router_hygiene",
    "preference",
    "lifecycle",
    "scorecard",
    "package",
]

STAGE_ALIASES = {
    "sft": "esft",
    "dpo": "preference",
    "thd": "preference",
    "router": "router_hygiene",
    "hygiene": "router_hygiene",
    "eval": "scorecard",
    "gate": "scorecard",
    "export": "package",
    "probe": "affinity",
    "sectors": "groups",
    "expert_groups": "groups",
    "studio": "groups",
}


def resolve_stages(requested: Optional[list[str]], cfg_stages: list[str]) -> list[str]:
    """
    Expand user/config stage names into ordered canonical stages.
    If requested is None, run full default pipeline.
    """
    if not requested:
        # full pipeline; ensure training.stages intents are covered
        stages = list(STAGE_ORDER)
        return stages

    resolved: list[str] = []
    for s in requested:
        s = s.strip().lower()
        s = STAGE_ALIASES.get(s, s)
        if s not in resolved:
            resolved.append(s)
    # stable order by STAGE_ORDER when possible
    order_idx = {name: i for i, name in enumerate(STAGE_ORDER)}
    resolved.sort(key=lambda x: order_idx.get(x, 100 + hash(x) % 50))
    return resolved


class TrainingPipeline:
    def __init__(self, config: AetherForgeConfig):
        self.config = config
        self.run_id = config.run.run_id or uuid.uuid4().hex[:12]
        self.root = Path(config.training.output_dir) / f"{config.run.name}-{self.run_id}"
        self.root.mkdir(parents=True, exist_ok=True)
        setup_logging(
            level="INFO",
            log_file=self.root / "aetherforge.log",
            run_id=self.run_id,
        )
        set_global_seed(config.training.seed or config.data.seed)
        self.audit = AuditLog(self.root / "audit.jsonl", run_id=self.run_id)
        dump_config(config, self.root / "config.resolved.yaml")
        self.bundle = None
        self.affinity = None
        self.plan = None
        self.baseline_scorecard = None
        self.final_scorecard = None
        self._data_bundle = None
        self.group_plan = None
        self.progress: Optional[LiveProgress] = None
        self.state: dict[str, Any] = {
            "run_id": self.run_id,
            "stages": {},
            "promoted": False,
        }

    def run(self, stages: Optional[list[str]] = None) -> dict[str, Any]:
        t0 = time.time()
        stages = resolve_stages(stages, self.config.training.stages)
        self.state["stage_list"] = stages

        self.progress = LiveProgress(
            self.root,
            run_id=self.run_id,
            run_name=self.config.run.name,
            domain=self.config.data.domain,
            model=self.config.model.name,
            dry_run=self.config.run.dry_run,
            stage_list=stages,
            artifacts_root=Path(self.config.training.output_dir),
        )
        self.progress.start_run()

        self.audit.record("pipeline", "start", {"stages": stages, "run_id": self.run_id})
        log.info(
            "AetherForge pipeline start run_id=%s dry_run=%s stages=%s root=%s",
            self.run_id,
            self.config.run.dry_run,
            stages,
            self.root,
        )
        log.info(
            "Live status → %s  |  dashboard: aetherforge dashboard",
            self.root / "live_status.json",
        )

        handlers = {
            "diagnostics": self._stage_diagnostics,
            "data": self._stage_data,
            "affinity": self._stage_affinity,
            "groups": self._stage_groups,
            "esft": self._stage_esft,
            "router_hygiene": self._stage_router_hygiene,
            "preference": self._stage_preference,
            "lifecycle": self._stage_lifecycle,
            "scorecard": self._stage_scorecard,
            "package": self._stage_package,
        }

        ok = True
        err_msg: Optional[str] = None
        try:
            for name in stages:
                fn = handlers.get(name)
                if fn is None:
                    log.warning("Unknown stage skipped: %s", name)
                    if self.progress:
                        self.progress.skip_stage(name, "unknown stage")
                    continue
                self.progress.start_stage(name)
                try:
                    fn()
                    summary = self.state.get("stages", {}).get(name)
                    if isinstance(summary, dict):
                        self.progress.end_stage(name, summary)
                    else:
                        self.progress.end_stage(name, {"ok": True})
                except Exception as stage_err:
                    self.progress.end_stage(
                        name, failed=True, error=str(stage_err)
                    )
                    raise
            self._finalize_promotion()
        except Exception as e:
            ok = False
            err_msg = str(e)
            self.audit.record("pipeline", "error", {"error": str(e)})
            log.exception("Pipeline failed: %s", e)
            raise
        finally:
            self.state["duration_sec"] = time.time() - t0
            with open(self.root / "pipeline_result.json", "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, default=str)
            self.audit.record(
                "pipeline",
                "end",
                {
                    "duration_sec": self.state["duration_sec"],
                    "promoted": self.state.get("promoted"),
                },
            )
            if self.progress:
                if self.state.get("promoted"):
                    self.progress.set_promotion(
                        promoted=True, path=self.state.get("promoted_path")
                    )
                elif self.state.get("promote_blocked"):
                    self.progress.set_promotion(
                        promoted=False, blocked=self.state.get("promote_blocked")
                    )
                self.progress.finish(ok=ok, error=err_msg)
            self.audit.close()

        log.info(
            "Pipeline complete in %.1fs promoted=%s → %s",
            self.state["duration_sec"],
            self.state.get("promoted"),
            self.root,
        )
        return self.state

    def _finalize_promotion(self) -> None:
        """Copy AetherPackage to promoted/ only on scorecard pass (+ human gate if high-stakes)."""
        sc = self.final_scorecard
        if sc is None:
            self.state["promoted"] = False
            return

        # Operator controls from dashboard (approve / reject / force)
        controls: dict[str, Any] = {}
        if self.progress:
            controls = self.progress.merge_controls_from_disk()
        else:
            ctrl_path = self.root / "operator_controls.json"
            if ctrl_path.exists():
                try:
                    controls = json.loads(ctrl_path.read_text(encoding="utf-8"))
                except Exception:
                    controls = {}

        if controls.get("rejected"):
            self.state["promoted"] = False
            self.state["promote_blocked"] = "operator_rejected"
            log.warning("Promotion blocked: operator rejected this run")
            return

        thr = self.config.eval.scorecard_thresholds
        needs_human = thr.require_human_approval or thr.high_stakes
        human_ok = bool(controls.get("human_approved"))

        if needs_human and not human_ok:
            # dry-run or live: wait for dashboard Approve
            self.state["promoted"] = False
            self.state["promote_blocked"] = "human_approval_required"
            log.warning(
                "Promotion blocked: high-stakes domain — approve in Training Console "
                "(aetherforge dashboard) or set operator_controls.json"
            )
            return

        if not sc.passed and not controls.get("force_promote"):
            self.state["promoted"] = False
            if self.config.eval.auto_rollback:
                log.warning(
                    "Scorecard FAILED — checkpoint not promoted "
                    "(use Force promote in dashboard to override)"
                )
            return

        # Dry-run: allow CI package export to promoted/ but stamp clearly
        moe_ready = bool((sc.details or {}).get("moe_ready"))
        dry = bool(self.config.run.dry_run)
        self.state["promoted"] = True
        self.state["promoted_kind"] = (
            "ci_dry_run" if dry else ("moe" if moe_ready else "ci_only")
        )
        pkg = self.root / "aetherpackage"
        if pkg.exists():
            dest = self.root / "promoted"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(pkg, dest)
            self.state["promoted_path"] = str(dest)
            # Truth stamp so operators never confuse dry CI with MoE readiness
            stamp = {
                "promoted_kind": self.state["promoted_kind"],
                "dry_run": dry,
                "moe_ready": moe_ready,
                "ci_complete": bool((sc.details or {}).get("ci_complete")),
                "promotion_label": (sc.details or {}).get("promotion_label"),
                "full_moe_promoted_readiness": bool(
                    (sc.details or {}).get("full_moe_promoted_readiness")
                ),
            }
            (dest / "PROMOTION_KIND.json").write_text(
                json.dumps(stamp, indent=2), encoding="utf-8"
            )
            if dry:
                (dest / "DRY_RUN_NOT_MOE_READY.txt").write_text(
                    "This package was produced by a DRY-RUN. "
                    "It is CI completeness only — NOT MoE weight-level readiness.\n",
                    encoding="utf-8",
                )
            self.audit.record(
                "pipeline",
                "promoted",
                {
                    "path": str(dest),
                    "force": bool(controls.get("force_promote")),
                    "human_approved": human_ok,
                    **stamp,
                },
            )
            log.info(
                "Promoted AetherPackage → %s (kind=%s)",
                dest,
                self.state["promoted_kind"],
            )

    # ── stages ──────────────────────────────────────────────────────────

    def _stage_diagnostics(self) -> None:
        log.info("Stage 0 — Diagnostics & load")
        if self.config.run.dry_run:
            summary = {
                "dry_run": True,
                "model": self.config.model.name,
                "family": self.config.model.family,
                "seed": self.config.training.seed,
            }
            self.state["stages"]["diagnostics"] = summary
            with open(self.root / "model_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            self.audit.record("diagnostics", "dry_run_skip_load", summary)
            return

        self.bundle = load_moe_model(
            self.config.model,
            self.config.training,
            backend=self.config.training.backend,
            for_training=True,
        )
        summary = self.bundle.summary()
        self.state["stages"]["diagnostics"] = summary
        with open(self.root / "model_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self.audit.record("diagnostics", "model_loaded", summary)

    def _stage_data(self) -> None:
        log.info("Stage 1 — DataForge")
        forge = DataForge(self.config.data, audit=self.audit)
        bundle = forge.build(output_dir=self.root / "data")
        self.state["stages"]["data"] = bundle.to_dict()
        self._data_bundle = bundle

    def _stage_affinity(self) -> None:
        log.info("Stage 2 — Affinity probe & selection")
        data = self._data_bundle
        probe_texts = (
            data.probe_texts
            if data
            else [
                f"Domain knowledge sample for {self.config.data.domain} case {i}."
                for i in range(min(64, self.config.affinity.probe_size))
            ]
        )

        if self.config.run.dry_run or self.bundle is None:
            import numpy as np
            from aetherforge.affinity.probe import AffinityResult

            # Prefer real architecture hints so Group Studio / ESFT dry-runs match Flash-0731 etc.
            name_l = (self.config.model.name or "").lower()
            if (
                self.config.model.family == "deepseek_v4_flash"
                or "flash" in name_l
                or "deepseek-v4" in name_l
            ):
                n_layers = 43  # DeepseekV4Config.num_hidden_layers for Flash-0731
                n_exp = int(self.config.model.num_experts or 256)
            else:
                n_layers = 4
                n_exp = max(self.config.model.num_experts or 8, 8)
            rng = np.random.default_rng(self.config.data.seed)
            # Mildly peaked distribution (more realistic than pure uniform random)
            routing = rng.dirichlet(alpha=np.ones(n_exp) * 0.7, size=n_layers)
            routing = routing * 100  # scale as pseudo-counts
            affinity = routing / (routing.sum(axis=1, keepdims=True) + 1e-12)
            ranked = []
            for li in range(n_layers):
                for ei in range(n_exp):
                    ranked.append((li, ei, float(affinity[li, ei])))
            ranked.sort(key=lambda x: x[2], reverse=True)
            from aetherforge.eval.metrics import load_balance_cv, routing_entropy

            self.affinity = AffinityResult(
                domain=self.config.data.domain,
                family=self.config.model.family,
                num_experts=n_exp,
                num_layers=n_layers,
                routing_freq=routing,
                grad_contrib=np.zeros_like(routing),
                affinity=affinity,
                ranked=ranked,
                entropy_per_layer=[
                    routing_entropy(routing[li]) for li in range(n_layers)
                ],
                load_balance_cv=load_balance_cv(routing),
                probe_tokens=len(probe_texts),
                metadata={
                    "synthetic": True,
                    "generator": "dirichlet",
                    "watermark": "SYNTHETIC_AFFINITY — not measured on weights",
                },
            )
        else:
            probe = AffinityProbe(
                self.bundle, self.config.affinity, domain=self.config.data.domain
            )
            self.affinity = probe.run(probe_texts)

        with open(self.root / "affinity.json", "w", encoding="utf-8") as f:
            json.dump(self.affinity.to_dict(), f, indent=2)
        # Truth watermark for synthetic dry-run affinity
        if (self.affinity.metadata or {}).get("synthetic"):
            (self.root / "AFFINITY_SYNTHETIC.txt").write_text(
                "SYNTHETIC_AFFINITY — routing matrix was generated (dirichlet/fixture), "
                "not measured from a loaded model. Do not treat as weight-level forensics.\n",
                encoding="utf-8",
            )
            if self.progress:
                self.progress.event(
                    "affinity",
                    "synthetic_watermark",
                    {"watermark": "SYNTHETIC_AFFINITY"},
                )

        selector = ExpertSelector(
            self.config.affinity,
            posture=getattr(self.config.training, "posture", "specialist"),
        )
        experts = self.bundle.experts if self.bundle else []
        # Dry-run: synthesize enough expert refs for broad/wide selection
        if not experts and (
            self.config.run.dry_run
            or getattr(self.config.training, "posture", "specialist") != "specialist"
        ):
            from aetherforge.models.moe_utils import ExpertRef

            n_l = self.affinity.num_layers
            n_e = self.affinity.num_experts
            experts = [
                ExpertRef(li, ei, f"model.layers.{li}.mlp.experts#{ei}", family=self.affinity.family)
                for li in range(n_l)
                for ei in range(n_e)
            ]
        self.plan = selector.select(self.affinity, experts)
        with open(self.root / "selection_plan.json", "w", encoding="utf-8") as f:
            json.dump(self.plan.to_dict(), f, indent=2)

        self.state["stages"]["affinity"] = {
            "top": self.affinity.ranked[:10],
            "selected": len(self.plan.selected),
            "load_balance_cv": self.affinity.load_balance_cv,
            "entropy_mean": (
                sum(self.affinity.entropy_per_layer) / len(self.affinity.entropy_per_layer)
                if self.affinity.entropy_per_layer
                else 0.0
            ),
        }
        self.audit.record("affinity", "complete", self.state["stages"]["affinity"])

    def _stage_groups(self) -> None:
        """Stage — Expert Group Studio: carve MoE into train-able sectors."""
        log.info("Stage — Expert Group Studio")
        from aetherforge.groups.studio import (
            create_studio_plan,
            group_plan_to_selection,
        )
        from aetherforge.groups.store import load_group_plan, save_group_plan
        from aetherforge.groups.cluster import merge_affinity_into_plan

        gcfg = self.config.groups
        out = self.root / "expert_groups.json"

        if not gcfg.enabled:
            self.state["stages"]["groups"] = {"skipped": True, "reason": "groups.disabled"}
            if self.progress:
                self.progress.skip_stage("groups", "disabled")
            return

        if gcfg.plan_path and Path(gcfg.plan_path).exists():
            plan = load_group_plan(gcfg.plan_path)
            log.info("Loaded group plan from %s", gcfg.plan_path)
        else:
            aff_dict = None
            if self.affinity is not None:
                aff_dict = self.affinity.to_dict()
            arch = self.bundle.arch if self.bundle else None
            plan = create_studio_plan(
                family=self.config.model.family
                if self.config.model.family != "auto"
                else (arch.family if arch else "generic_moe"),
                model_name=self.config.model.name,
                num_groups=gcfg.target_num_groups,
                strategy=gcfg.strategy,
                affinity=aff_dict,
                arch_layers=arch.num_layers if arch else None,
                arch_experts=self.config.model.num_experts
                or (arch.num_experts if arch else None),
                arch_topk=self.config.model.num_experts_per_tok
                or (arch.num_experts_per_tok if arch else None),
                total_params_b=gcfg.total_params_b,
                active_params_b=gcfg.active_params_b,
            )
            # Fix layers from affinity if available
            if self.affinity and plan.capacity.num_layers != self.affinity.num_layers:
                from aetherforge.groups.capacity import estimate_capacity
                from aetherforge.groups.cluster import auto_partition_groups

                cap = estimate_capacity(
                    family=plan.capacity.family,
                    model_name=self.config.model.name,
                    num_layers=self.affinity.num_layers,
                    num_experts=self.affinity.num_experts,
                    top_k=plan.capacity.top_k,
                    total_params_b=gcfg.total_params_b,
                    active_params_b=gcfg.active_params_b,
                )
                plan = auto_partition_groups(
                    cap,
                    num_groups=gcfg.target_num_groups,
                    strategy=gcfg.strategy,
                    affinity_matrix=self.affinity.affinity.tolist()
                    if hasattr(self.affinity.affinity, "tolist")
                    else None,
                    ranked=self.affinity.ranked,
                    model_name=self.config.model.name,
                    target_active_fire_ratio=gcfg.target_active_fire_ratio,
                )

        if self.affinity is not None:
            merge_affinity_into_plan(
                plan,
                affinity_matrix=self.affinity.affinity.tolist()
                if hasattr(self.affinity.affinity, "tolist")
                else None,
                ranked=self.affinity.ranked,
            )

        # Merge operator edits if dashboard saved a plan mid-run
        live_plan = self.root / "expert_groups.json"
        if live_plan.exists() and gcfg.plan_path is None:
            try:
                edited = load_group_plan(live_plan)
                # Prefer edited membership if same model family
                if edited.groups:
                    plan = edited
                    log.info("Using on-disk expert_groups.json edits (%d groups)", len(plan.groups))
            except Exception as e:
                log.debug("Could not reload live group plan: %s", e)

        save_group_plan(plan, out)
        self.group_plan = plan

        # Apply train_scope (specialist selected | broad top_n | wide all)
        scope = getattr(gcfg, "train_scope", "selected") or "selected"
        if scope == "all":
            for g in plan.groups:
                g.enabled = True
                g.train = True
                g.freeze = False
        elif scope == "all_enabled":
            for g in plan.groups:
                if g.enabled:
                    g.train = True
                    g.freeze = False
        elif scope == "top_n":
            n = max(1, int(getattr(gcfg, "train_top_n", 4) or 4))
            ranked_g = sorted(
                plan.groups,
                key=lambda gg: (
                    sum(c.affinity for c in gg.cells) / max(len(gg.cells), 1)
                ),
                reverse=True,
            )
            train_ids = {g.id for g in ranked_g[:n]}
            for g in plan.groups:
                if g.id in train_ids:
                    g.enabled = True
                    g.train = True
                    g.freeze = False
                else:
                    g.train = False
                    g.freeze = True
            log.info(
                "train_scope=top_n: training %d/%d sectors",
                len(train_ids),
                len(plan.groups),
            )
        save_group_plan(plan, out)

        # Drive ESFT selection from enabled train groups
        if gcfg.use_for_training and plan.enabled_train_groups():
            self.plan = group_plan_to_selection(plan, domain=self.config.data.domain)
            # Stamp posture into selection metadata
            self.plan.metadata["posture"] = getattr(
                self.config.training, "posture", "specialist"
            )
            self.plan.metadata["train_scope"] = scope
            with open(self.root / "selection_plan.json", "w", encoding="utf-8") as f:
                json.dump(self.plan.to_dict(), f, indent=2)

        # Sector forensics — inventory what each sector contains (edit guide)
        aff_dict = self.affinity.to_dict() if self.affinity is not None else None
        try:
            from aetherforge.groups.forensics import (
                run_model_forensics,
                forensics_markdown,
            )

            freport = run_model_forensics(plan, affinity=aff_dict)
            with open(self.root / "sector_forensics.json", "w", encoding="utf-8") as f:
                json.dump(freport.to_dict(), f, indent=2, default=str)
            (self.root / "sector_forensics.md").write_text(
                forensics_markdown(freport), encoding="utf-8"
            )
            self.state.setdefault("stages", {})
            log.info(
                "Sector forensics: %d dossiers → sector_forensics.json",
                freport.n_groups,
            )
        except Exception as e:
            log.warning("Sector forensics skipped: %s", e)

        # Pre-sector forensic readiness gate (auto-bind unbound sectors)
        if gcfg.require_forensics_gate:
            try:
                from aetherforge.groups.readiness import (
                    readiness_markdown,
                    run_forensics_gate,
                )
                from aetherforge.groups.store import save_group_plan as _save_gp

                gate = run_forensics_gate(
                    plan,
                    affinity=aff_dict,
                    mode=getattr(gcfg, "forensics_gate_mode", "warn") or "warn",
                    auto_bind=bool(getattr(gcfg, "auto_bind_from_forensics", True)),
                    global_domain=self.config.data.domain,
                    only_train_groups=True,
                )
                with open(self.root / "sector_readiness.json", "w", encoding="utf-8") as f:
                    json.dump(gate.to_dict(), f, indent=2, default=str)
                (self.root / "sector_readiness.md").write_text(
                    readiness_markdown(gate), encoding="utf-8"
                )
                # Persist auto-bound domain/topics/keywords
                _save_gp(plan, out)
                self.group_plan = plan
                # Refresh ESFT selection after auto-bind + any freeze changes
                if gcfg.use_for_training and plan.enabled_train_groups():
                    self.plan = group_plan_to_selection(
                        plan, domain=self.config.data.domain
                    )
                    self.plan.metadata["posture"] = getattr(
                        self.config.training, "posture", "specialist"
                    )
                    self.plan.metadata["train_scope"] = scope
                    self.plan.metadata["readiness_overall"] = gate.overall
                    with open(self.root / "selection_plan.json", "w", encoding="utf-8") as f:
                        json.dump(self.plan.to_dict(), f, indent=2)
                log.info("Pre-sector readiness: %s", gate.narrative)
                if gate.overall == "block" and (
                    getattr(gcfg, "forensics_gate_mode", "warn") == "block"
                ):
                    raise RuntimeError(
                        f"Forensic readiness gate blocked training: {gate.narrative}"
                    )
            except RuntimeError:
                raise
            except Exception as e:
                log.warning("Forensic readiness gate skipped: %s", e)

        summary = plan.summary()
        self.state["stages"]["groups"] = summary
        self.audit.record("groups", "complete", summary)
        log.info(
            "Groups: %d sectors, %d train, active≈%.1fB, max_disjoint≈%d",
            summary["n_groups"],
            summary["n_train_groups"],
            summary["active_params_b"],
            summary["max_disjoint_active_groups"],
        )

    def _stage_esft(self) -> None:
        log.info("Stage 3 — ESFT / Expert-LoRA")
        data = self._data_bundle
        train_records = data.train_records if data else []
        sector_mode = (
            getattr(self.config.training, "sector_mode", "sequential") or "sequential"
        ).lower()

        # Sequential: forensics (again per sector) → sector datasets → ESFT each
        if (
            sector_mode == "sequential"
            and self.group_plan is not None
            and self.group_plan.enabled_train_groups()
        ):
            self._stage_esft_sequential(train_records)
            return

        out = self.root / "checkpoints" / "esft"

        if self.config.run.dry_run or self.bundle is None or self.plan is None:
            out.mkdir(parents=True, exist_ok=True)
            # Still build sector dataset plan in joint dry-run when requested
            if (
                getattr(self.config.data, "sector_datasets", True)
                and self.group_plan is not None
            ):
                try:
                    self._build_sector_datasets_only(train_records)
                except Exception as e:
                    log.warning("sector dataset plan (joint) skipped: %s", e)
            result = {
                "dry_run": True,
                "sector_mode": "joint",
                "output_dir": str(out),
                "train_records": len(train_records),
                "selected_experts": len(self.plan.selected) if self.plan else 0,
            }
            with open(out / "esft_result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            self.state["stages"]["esft"] = result
            self.audit.record("esft", "dry_run", result)
            return

        trainer = ESFTTrainer(
            self.bundle, self.config.training, self.plan, audit=self.audit
        )
        esft_result = trainer.train(train_records, out)
        payload = esft_result.to_dict()
        payload["sector_mode"] = "joint"
        self.state["stages"]["esft"] = payload

    def _build_sector_datasets_only(
        self, train_records: list[dict[str, Any]]
    ) -> Any:
        """Forge per-sector shards without training (joint mode helper / CLI)."""
        from aetherforge.data.sector_datasets import SectorDatasetForge
        from aetherforge.groups.forensics import forensics_for_group

        plan = self.group_plan
        assert plan is not None
        aff = self.affinity.to_dict() if self.affinity is not None else None
        forensics_by_id = {
            g.id: forensics_for_group(plan, g.id, affinity=aff)
            for g in plan.enabled_train_groups()
        }
        tcfg = self.config.training
        forge = SectorDatasetForge(
            self.config.data,
            min_match=float(getattr(self.config.data, "sector_min_match", 0.18) or 0.18),
            shared_fraction=float(
                getattr(tcfg, "sector_shared_fraction", 0.15) or 0.15
            ),
            min_samples=int(getattr(tcfg, "sector_min_samples", 8) or 8),
            contract_mode=str(getattr(tcfg, "sector_contract_mode", "warn") or "warn"),
            min_real_fraction=float(
                getattr(tcfg, "sector_min_real_fraction", 0.15) or 0.15
            ),
            max_synth_fraction=float(
                getattr(tcfg, "sector_max_synth_fraction", 0.85) or 0.85
            ),
            min_unique_ratio=float(
                getattr(tcfg, "sector_min_unique_ratio", 0.35) or 0.35
            ),
        )
        data = self._data_bundle
        return forge.build(
            plan,
            train_records,
            output_dir=self.root / "sector_datasets",
            forensics_by_id=forensics_by_id,
            eval_texts=data.eval_texts if data else None,
        )

    def _stage_esft_sequential(self, train_records: list[dict[str, Any]]) -> None:
        """Per-sector forensic assess → dataset → ESFT (siblings frozen)."""
        from aetherforge.training.sector_workflow import SectorWorkflow

        assert self.group_plan is not None
        gcfg = self.config.groups
        tcfg = self.config.training
        out = self.root / "sector_workflow"
        log.info(
            "Stage 3 — Sequential sector ESFT (%d train sectors)",
            len(self.group_plan.enabled_train_groups()),
        )
        if self.progress:
            self.progress.set_sector_mode("sequential")
        wf = SectorWorkflow(
            self.group_plan,
            tcfg,
            self.config.data,
            bundle=self.bundle if not self.config.run.dry_run else None,
            affinity=self.affinity.to_dict() if self.affinity is not None else None,
            audit=self.audit,
            dry_run=bool(self.config.run.dry_run or self.bundle is None),
            gate_mode=getattr(gcfg, "forensics_gate_mode", "warn") or "warn",
            auto_bind=bool(getattr(gcfg, "auto_bind_from_forensics", True)),
            min_samples=int(getattr(tcfg, "sector_min_samples", 8) or 8),
            shared_fraction=float(getattr(tcfg, "sector_shared_fraction", 0.15) or 0.15),
            continue_on_block=bool(getattr(tcfg, "sector_continue_on_block", True)),
            progress=self.progress,
        )
        data = self._data_bundle
        result = wf.run(
            train_records,
            out,
            eval_texts=data.eval_texts if data else None,
        )
        # Promote combined selection + mirror esft summary for scorecard/package
        sel_path = out / "selection_plan.sectors.json"
        if sel_path.exists():
            import shutil

            shutil.copy2(sel_path, self.root / "selection_plan.json")
        # Aggregate joint-style checkpoint pointer for packaging
        joint_ptr = self.root / "checkpoints" / "esft"
        joint_ptr.mkdir(parents=True, exist_ok=True)
        with open(joint_ptr / "esft_result.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sector_mode": "sequential",
                    "n_trained": result.n_trained,
                    "n_skipped": result.n_skipped,
                    "n_blocked": result.n_blocked,
                    "workflow": str(out / "sector_workflow.json"),
                    "sectors": [s.to_dict() for s in result.sectors],
                },
                f,
                indent=2,
                default=str,
            )
        # Keep group plan after auto-bind from workflow
        bound = out / "expert_groups.bound.json"
        if bound.exists():
            from aetherforge.groups.store import load_group_plan

            try:
                self.group_plan = load_group_plan(bound)
                save_path = self.root / "expert_groups.json"
                from aetherforge.groups.store import save_group_plan

                save_group_plan(self.group_plan, save_path)
            except Exception as e:
                log.debug("reload bound groups: %s", e)

        summary = result.to_dict()
        self.state["stages"]["esft"] = summary
        self.audit.record("esft", "sequential_complete", {
            "n_trained": result.n_trained,
            "n_skipped": result.n_skipped,
            "n_blocked": result.n_blocked,
            "duration_sec": result.duration_sec,
        })
        log.info(
            "Sequential sector ESFT: trained=%d blocked=%d → %s",
            result.n_trained,
            result.n_blocked,
            out,
        )

    def _stage_router_hygiene(self) -> None:
        log.info("Stage 4 — Router hygiene")
        data = self._data_bundle
        texts = [r.get("text", "") for r in (data.train_records if data else [])][:256]
        out = self.root / "checkpoints" / "router_hygiene"

        if self.config.run.dry_run or self.bundle is None:
            out.mkdir(parents=True, exist_ok=True)
            result = {"dry_run": True, "texts": len(texts)}
            with open(out / "router_hygiene_result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            self.state["stages"]["router_hygiene"] = result
            return

        rh = RouterHygieneTrainer(self.bundle, self.config.training, audit=self.audit)
        result = rh.run(texts, out)
        self.state["stages"]["router_hygiene"] = result.to_dict()

    def _stage_preference(self) -> None:
        log.info("Stage 5 — Preference / THD")
        data = self._data_bundle
        pairs = []
        if data and data.preference_pairs:
            pairs = [
                PreferencePair(
                    prompt=p["prompt"],
                    chosen=p["chosen"],
                    rejected=p["rejected"],
                    source=p.get("source", "thd"),
                )
                for p in data.preference_pairs
            ]
        out = self.root / "checkpoints" / "preference"
        aligner = PreferenceAligner(method="dpo")
        model = (
            self.bundle.model if self.bundle and not self.config.run.dry_run else None
        )
        tok = (
            self.bundle.tokenizer
            if self.bundle and not self.config.run.dry_run
            else None
        )
        result = aligner.run(pairs, out, model=model, tokenizer=tok)
        self.state["stages"]["preference"] = result.to_dict()
        self.audit.record("preference", "complete", result.to_dict())

    def _stage_lifecycle(self) -> None:
        log.info("Stage — Elastic lifecycle plan")
        if self.affinity is None:
            self.state["stages"]["lifecycle"] = {"skipped": True, "reason": "no_affinity"}
            return
        mon = ExpertUtilizationMonitor(
            low_threshold=self.config.lifecycle.util_low_threshold,
            high_threshold=self.config.lifecycle.util_high_threshold,
        )
        report = mon.from_affinity(self.affinity)
        mgr = ExpertLifecycleManager(self.config.lifecycle)
        plan = mgr.plan_from_report(report)
        out = self.root / "lifecycle"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "utilization.json", "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        mgr.save(plan, out / "lifecycle_plan.json")
        self.state["stages"]["lifecycle"] = {
            "n_actions": len(plan.actions),
            "low": len(report.low),
            "high": len(report.high),
        }
        self.audit.record("lifecycle", "planned", self.state["stages"]["lifecycle"])

    def _stage_scorecard(self) -> None:
        log.info("Stage 6 — Reliability Scorecard")
        data = self._data_bundle
        quality = data.quality if data else None
        keywords = data.keywords if data else list(self.config.data.keywords)
        scorecard = ReliabilityScorecard(
            self.config.eval,
            domain=self.config.data.domain,
            domain_keywords=keywords,
        )
        # Pass sequential sector workflow summary when present
        sector_wf = None
        esft_stage = self.state.get("stages", {}).get("esft") or {}
        if esft_stage.get("schema") in (
            "aetherforge.sector_workflow.v1",
            "aetherforge.sector_workflow.v2",
        ) or esft_stage.get("mode") == "sequential":
            sector_wf = esft_stage
            # load interference if written
            inter_path = self.root / "sector_workflow" / "interference.json"
            if inter_path.exists():
                try:
                    sector_wf = dict(sector_wf)
                    sector_wf["interference"] = json.loads(
                        inter_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    pass
        sc = scorecard.evaluate(
            model=self.bundle.model
            if self.bundle and not self.config.run.dry_run
            else None,
            tokenizer=self.bundle.tokenizer
            if self.bundle and not self.config.run.dry_run
            else None,
            affinity=self.affinity,
            eval_texts=data.eval_texts if data else [],
            dry_run=self.config.run.dry_run,
            quality_report=quality,
            domain_keywords=keywords,
            sector_workflow=sector_wf,
        )
        self.final_scorecard = sc
        path = self.root / "scorecard.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sc.to_dict(), f, indent=2)
        # Human-readable promotion label
        label = (sc.details or {}).get("promotion_label") or ""
        (self.root / "PROMOTION_LABEL.txt").write_text(label + "\n", encoding="utf-8")
        self.state["stages"]["scorecard"] = sc.to_dict()
        self.state["promotion_label"] = label
        self.state["moe_ready"] = bool((sc.details or {}).get("moe_ready"))
        self.state["ci_complete"] = bool((sc.details or {}).get("ci_complete"))
        gate = "pass" if sc.passed else "fail"
        self.audit.record("scorecard", "evaluated", sc.to_dict(), gate_result=gate)
        self.state["promoted"] = False  # finalized after package
        if self.progress:
            self.progress.set_metrics(sc.metrics)
            self.progress.event(
                "scorecard",
                (sc.details or {}).get("scorecard_kind") or "evaluated",
                {
                    "promotion_label": label,
                    "moe_ready": self.state["moe_ready"],
                    "ci_complete": self.state["ci_complete"],
                },
            )

    def _stage_package(self) -> None:
        log.info("Stage 7 — AetherPackage export")
        builder = AetherPackageBuilder(self.config)
        pkg = builder.build(
            run_dir=self.root,
            affinity=self.affinity,
            plan=self.plan,
            scorecard=self.final_scorecard,
            model_summary=self.state.get("stages", {}).get("diagnostics"),
        )
        self.state["stages"]["package"] = pkg.to_dict()
        self.audit.record("package", "exported", pkg.to_dict())

        # Flagship proof artifact for public recipes
        if "flagship" in (self.config.run.name or "").lower():
            try:
                from aetherforge.training.flagship_report import write_flagship_report

                write_flagship_report(
                    self.root,
                    state=self.state,
                    scorecard=self.final_scorecard.to_dict()
                    if self.final_scorecard
                    else None,
                    groups_summary=self.state.get("stages", {}).get("groups"),
                    recipe=self.config.run.name,
                )
            except Exception as e:
                log.debug("flagship report skipped: %s", e)

        # Copy expert_groups into aetherpackage for portability
        eg = self.root / "expert_groups.json"
        if eg.exists() and (self.root / "aetherpackage").is_dir():
            import shutil

            shutil.copy2(eg, self.root / "aetherpackage" / "expert_groups.json")
