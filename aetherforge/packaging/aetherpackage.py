"""
AetherPackage — Stage 7 export.

Not just weights: specialists + router config + affinity maps + agent
orchestration YAML + continuous update hooks + scorecard + model card.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from aetherforge.affinity.expert_selector import SelectionPlan
from aetherforge.affinity.probe import AffinityResult
from aetherforge.eval.scorecard import Scorecard
from aetherforge.utils.config import AetherForgeConfig
from aetherforge.utils.logging import get_logger

log = get_logger("packaging.aetherpackage")


@dataclass
class AetherPackage:
    path: str
    domain: str
    model_name: str
    promoted: bool
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "domain": self.domain,
            "model_name": self.model_name,
            "promoted": self.promoted,
            "files": self.files,
            "metadata": self.metadata,
        }


class AetherPackageBuilder:
    def __init__(self, config: AetherForgeConfig):
        self.config = config

    def build(
        self,
        run_dir: str | Path,
        *,
        affinity: Optional[AffinityResult] = None,
        plan: Optional[SelectionPlan] = None,
        scorecard: Optional[Scorecard] = None,
        model_summary: Optional[dict[str, Any]] = None,
    ) -> AetherPackage:
        run_dir = Path(run_dir)
        pkg_dir = run_dir / "aetherpackage"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        files: dict[str, str] = {}
        promoted = bool(scorecard.passed) if scorecard else False

        # Copy key artifacts
        for name in (
            "affinity.json",
            "selection_plan.json",
            "scorecard.json",
            "config.resolved.yaml",
            "model_summary.json",
            "pipeline_result.json",
            "audit.jsonl",
        ):
            src = run_dir / name
            if src.exists():
                dst = pkg_dir / name
                shutil.copy2(src, dst)
                files[name] = str(dst)

        # Lifecycle artifacts (elastic expert plans)
        life_src = run_dir / "lifecycle"
        if life_src.is_dir():
            life_dst = pkg_dir / "lifecycle"
            if life_dst.exists():
                shutil.rmtree(life_dst)
            shutil.copytree(life_src, life_dst)
            files["lifecycle"] = str(life_dst)

        # Adapter / checkpoint refs
        ckpt = run_dir / "checkpoints" / "esft"
        if ckpt.exists():
            files["adapter_dir"] = str(ckpt)

        # Orchestration YAML for multi-agent hive
        orch = {
            "aetherforge_version": "0.1.0",
            "domain": self.config.data.domain,
            "base_model": self.config.model.name,
            "consult_protocol": self.config.hive.consult_protocol,
            "max_consult_rounds": self.config.hive.max_consult_rounds,
            "specialists": self.config.hive.specialists
            or [self.config.data.domain],
            "affinity_top_k": self.config.affinity.top_k_experts,
            "continuous": {
                "enabled": self.config.continuous.enabled,
                "federated": self.config.continuous.federated,
                "update_frequency": self.config.continuous.update_frequency,
            },
            "openveil": {
                "privacy_mode": self.config.data.privacy_mode,
                "isolate_experts": True,
            },
            "endpoints": {
                # filled by deploy layer
                "inference": None,
                "update_hook": None,
            },
        }
        orch_path = pkg_dir / "orchestration.yaml"
        with open(orch_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(orch, f, sort_keys=False)
        files["orchestration"] = str(orch_path)

        # Model card
        card = self._model_card(affinity, plan, scorecard, model_summary, promoted)
        card_path = pkg_dir / "MODEL_CARD.md"
        card_path.write_text(card, encoding="utf-8")
        files["model_card"] = str(card_path)

        # Manifest
        manifest = {
            "created_at": time.time(),
            "domain": self.config.data.domain,
            "model_name": self.config.model.name,
            "promoted": promoted,
            "files": files,
            "scorecard_passed": promoted,
            "run_dir": str(run_dir),
        }
        man_path = pkg_dir / "manifest.json"
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        files["manifest"] = str(man_path)

        # Continuous update protocol stub
        cont = {
            "protocol": "aetherforge.continuous.v1",
            "on_error_cluster": [
                "cluster_errors",
                "generate_synthetic_and_thd",
                "affinity_probe_delta",
                "targeted_esft",
                "router_hygiene",
                "scorecard_gate",
                "promote_or_rollback",
            ],
            "federated": self.config.continuous.federated,
        }
        cont_path = pkg_dir / "continuous_protocol.json"
        with open(cont_path, "w", encoding="utf-8") as f:
            json.dump(cont, f, indent=2)
        files["continuous_protocol"] = str(cont_path)

        pkg = AetherPackage(
            path=str(pkg_dir),
            domain=self.config.data.domain,
            model_name=self.config.model.name,
            promoted=promoted,
            files=files,
            metadata=manifest,
        )
        log.info("AetherPackage exported → %s (promoted=%s)", pkg_dir, promoted)
        return pkg

    def _model_card(
        self,
        affinity: Optional[AffinityResult],
        plan: Optional[SelectionPlan],
        scorecard: Optional[Scorecard],
        model_summary: Optional[dict[str, Any]],
        promoted: bool,
    ) -> str:
        cfg = self.config
        lines = [
            f"# AetherForge Specialist — `{cfg.data.domain}`",
            "",
            f"**Base model:** `{cfg.model.name}`  ",
            f"**Family:** `{cfg.model.family}`  ",
            f"**Method:** `{cfg.training.method}`  ",
            f"**Promoted:** `{promoted}`  ",
            f"**Privacy mode:** `{cfg.data.privacy_mode}`",
            "",
            "## Intended use",
            f"Domain specialist post-trained via AetherForge AGPS/ESFT for **{cfg.data.domain}**.",
            "Part of a multi-specialist industry hive. Decision-support only — not a substitute "
            "for licensed professional judgment in regulated fields.",
            "",
            "## Training summary",
            f"- LoRA r={cfg.training.lora_r}, alpha={cfg.training.lora_alpha}",
            f"- Specialization loss weight={cfg.training.specialization_loss_weight}",
            f"- Stages: {', '.join(cfg.training.stages)}",
            f"- Selected experts (top-k): {cfg.affinity.top_k_experts}",
            "",
        ]
        if plan:
            lines += [
                f"- Experts selected: {len(plan.selected)}",
                f"- Router frozen during ESFT: {plan.freeze_router}",
                "",
            ]
        if affinity:
            lines += [
                "## Affinity snapshot",
                f"- Layers×experts: {affinity.num_layers}×{affinity.num_experts}",
                f"- Load-balance CV: {affinity.load_balance_cv:.4f}",
                f"- Probe tokens: {affinity.probe_tokens}",
                "",
            ]
        if scorecard:
            lines += [
                "## Reliability Scorecard",
                f"- Passed: {scorecard.passed}",
                f"- Metrics: `{json.dumps(scorecard.metrics)}`",
                f"- Failures: {scorecard.gate.failures or 'none'}",
                "",
            ]
        if model_summary:
            lines += ["## Model load summary", "```json", json.dumps(model_summary, indent=2), "```", ""]
        lines += [
            "## Safety",
            "- Not a substitute for licensed professional judgment.",
            "- High-stakes domains require human-in-the-loop promotion when configured.",
            "- Full audit trail in `audit.jsonl`.",
            "",
            "## Hive integration",
            "See `orchestration.yaml` for consult protocol and continuous update hooks.",
            "",
            "---",
            "*Generated by AetherForge v0.1.0*",
        ]
        return "\n".join(lines)
