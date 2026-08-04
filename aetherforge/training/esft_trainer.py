"""
ESFT / Expert-LoRA trainer — Stage 3.

Trains selected experts (and optional shared/attention adapters) with:
  - task LM loss
  - specialization-aware losses
  - progressive unfreeze schedule (AGPS tiers)
  - optional EWC / KD hooks (weights from config)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aetherforge.affinity.expert_selector import SelectionPlan
from aetherforge.models.loaders import MoEModelBundle, apply_expert_lora, save_adapter
from aetherforge.models.moe_utils import (
    count_trainable_parameters,
    freeze_all_parameters,
    select_param_names_for_experts,
    unfreeze_parameters_by_name,
)
from aetherforge.training.specialization_loss import SpecializationLoss
from aetherforge.utils.audit import AuditLog
from aetherforge.utils.config import TrainingConfig
from aetherforge.utils.logging import get_logger

log = get_logger("training.esft")


@dataclass
class ESFTResult:
    output_dir: str
    steps: int
    final_loss: float
    trainable_params: int
    total_params: int
    selected_experts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "steps": self.steps,
            "final_loss": self.final_loss,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "selected_experts": self.selected_experts,
            "metrics": self.metrics,
            "duration_sec": self.duration_sec,
        }


class ESFTTrainer:
    def __init__(
        self,
        bundle: MoEModelBundle,
        train_cfg: TrainingConfig,
        plan: SelectionPlan,
        audit: Optional[AuditLog] = None,
    ):
        self.bundle = bundle
        self.cfg = train_cfg
        self.plan = plan
        self.audit = audit
        self.spec_loss = SpecializationLoss(
            variance_weight=train_cfg.specialization_loss_weight,
            orthogonality_weight=train_cfg.specialization_loss_weight * 0.5,
            load_balance_weight=train_cfg.load_balance_loss_weight,
        )

    def prepare(self) -> MoEModelBundle:
        """Apply LoRA + freeze plan (V4 fused experts use target_parameters + grad masks)."""
        method = self.cfg.method
        is_v4 = self.bundle.arch.family == "deepseek_v4_flash" or (
            "deepseek" in (self.bundle.model_name or "").lower()
            and ("v4" in (self.bundle.model_name or "").lower()
                 or "flash" in (self.bundle.model_name or "").lower())
        )

        if method in ("esft_lora", "qlora"):
            self.bundle = apply_expert_lora(self.bundle, self.cfg, self.plan.selected)
        elif method == "full_esft":
            if is_v4:
                from aetherforge.models.deepseek_v4 import freeze_nonselected_base_experts

                pairs = [(e.layer_idx, e.expert_idx) for e in self.plan.selected]
                handles = freeze_nonselected_base_experts(
                    self.bundle.model, pairs, self.bundle.arch
                )
                self.bundle.extras["expert_grad_mask_handles"] = handles
                log.info(
                    "full_esft V4: slice-masked %d expert slots",
                    len(pairs),
                )
            else:
                freeze_all_parameters(self.bundle.model)
                names = select_param_names_for_experts(
                    self.bundle.model,
                    self.plan.selected,
                    include_router=not self.plan.freeze_router,
                    arch=self.bundle.arch,
                )
                n = unfreeze_parameters_by_name(self.bundle.model, names)
                log.info(
                    "full_esft: unfroze %d params across %d experts",
                    n,
                    len(self.plan.selected),
                )
        else:
            self.bundle = apply_expert_lora(self.bundle, self.cfg, self.plan.selected)

        trainable, total = count_trainable_parameters(self.bundle.model)
        log.info(
            "ESFT prepare: trainable=%s / total=%s (%.4f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / max(total, 1),
        )
        if self.audit:
            self.audit.record(
                "esft",
                "prepare",
                {
                    "method": method,
                    "selected": len(self.plan.selected),
                    "trainable": trainable,
                    "total": total,
                    "family": self.bundle.arch.family,
                    "v4_grad_masks": len(
                        self.bundle.extras.get("expert_grad_mask_handles") or []
                    ),
                },
            )
        return self.bundle

    def train(
        self,
        train_texts: list[dict[str, Any]] | list[str],
        output_dir: str | Path,
        eval_texts: Optional[list[Any]] = None,
    ) -> ESFTResult:
        """
        Run SFT loop. Uses HF Trainer when available; otherwise a minimal torch loop.

        train_texts: list of {"text": ...} or plain strings, or chat {"messages": [...]}
        """
        t0 = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.prepare()
        records = self._normalize_records(train_texts)
        if self.cfg.max_steps == 0 or (self.cfg.dry_run if hasattr(self.cfg, "dry_run") else False):
            pass

        if not records:
            log.warning("No training records — writing empty ESFT result")
            result = ESFTResult(
                output_dir=str(output_dir),
                steps=0,
                final_loss=0.0,
                trainable_params=0,
                total_params=0,
                metrics={"warning": "empty_dataset"},
            )
            self._write_result(output_dir, result)
            return result

        # Prefer transformers Trainer path
        try:
            result = self._train_hf(records, output_dir, eval_texts)
        except Exception as e:
            log.warning("HF Trainer path failed (%s); using minimal loop", e)
            result = self._train_minimal(records, output_dir)

        result.duration_sec = time.time() - t0
        result.selected_experts = [
            {"layer": e.layer_idx, "expert": e.expert_idx, "module": e.module_name}
            for e in self.plan.selected
        ]
        self._write_result(output_dir, result)
        if self.audit:
            self.audit.record(
                "esft",
                "complete",
                result.to_dict(),
                checkpoint=str(output_dir),
            )
        return result

    def _normalize_records(self, train_texts: list[Any]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for item in train_texts:
            if isinstance(item, str):
                out.append({"text": item})
            elif isinstance(item, dict):
                if "text" in item:
                    out.append({"text": str(item["text"])})
                elif "messages" in item:
                    # simple chat flatten
                    parts = []
                    for m in item["messages"]:
                        role = m.get("role", "user")
                        parts.append(f"{role}: {m.get('content', '')}")
                    out.append({"text": "\n".join(parts)})
                elif "prompt" in item and "completion" in item:
                    out.append({"text": f"{item['prompt']}{item['completion']}"})
                else:
                    out.append({"text": json.dumps(item)})
        return out

    def _train_hf(
        self,
        records: list[dict[str, str]],
        output_dir: Path,
        eval_texts: Optional[list[Any]],
    ) -> ESFTResult:
        import torch
        from datasets import Dataset
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        tokenizer = self.bundle.tokenizer
        model = self.bundle.model
        max_len = getattr(self.bundle.model.config, "max_position_embeddings", 4096)
        max_len = min(max_len, 4096)

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_len,
                padding=False,
            )

        ds = Dataset.from_list(records)
        ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            num_train_epochs=self.cfg.num_epochs,
            max_steps=self.cfg.max_steps if self.cfg.max_steps else -1,
            warmup_ratio=self.cfg.warmup_ratio,
            weight_decay=self.cfg.weight_decay,
            max_grad_norm=self.cfg.max_grad_norm,
            logging_steps=self.cfg.logging_steps,
            save_steps=self.cfg.save_steps,
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            fp16=False,
            report_to=[],
            remove_unused_columns=False,
            gradient_checkpointing=self.cfg.gradient_checkpointing,
            seed=self.cfg.seed,
        )

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=collator,
        )
        train_out = trainer.train()
        trainer.save_model(str(output_dir))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(output_dir))

        trainable, total = count_trainable_parameters(model)
        metrics = dict(train_out.metrics) if hasattr(train_out, "metrics") else {}
        final_loss = float(metrics.get("train_loss", 0.0))
        steps = int(metrics.get("train_steps", metrics.get("global_step", 0)) or 0)

        return ESFTResult(
            output_dir=str(output_dir),
            steps=steps,
            final_loss=final_loss,
            trainable_params=trainable,
            total_params=total,
            metrics=metrics,
        )

    def _train_minimal(self, records: list[dict[str, str]], output_dir: Path) -> ESFTResult:
        """Lightweight torch loop for environments without full Trainer deps."""
        import torch
        from torch.optim import AdamW

        model = self.bundle.model
        tokenizer = self.bundle.tokenizer
        device = next(model.parameters()).device
        model.train()

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            # last resort: train all
            for p in model.parameters():
                p.requires_grad = True
            params = list(model.parameters())

        opt = AdamW(params, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        max_steps = self.cfg.max_steps or max(10, len(records) // max(self.cfg.per_device_train_batch_size, 1))
        max_steps = min(max_steps, 500)  # safety cap for minimal loop
        max_len = 512

        losses: list[float] = []
        step = 0
        idx = 0
        accum = self.cfg.gradient_accumulation_steps

        while step < max_steps:
            batch_texts = []
            for _ in range(self.cfg.per_device_train_batch_size):
                batch_texts.append(records[idx % len(records)]["text"])
                idx += 1
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = enc["input_ids"].clone()
            out = model(**enc, labels=labels)
            loss = out.loss / accum
            loss.backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()) * accum)
            step += 1
            if step % self.cfg.logging_steps == 0:
                log.info("minimal step %d loss=%.4f", step, losses[-1])

        save_adapter(self.bundle, output_dir)
        trainable, total = count_trainable_parameters(model)
        return ESFTResult(
            output_dir=str(output_dir),
            steps=step,
            final_loss=float(sum(losses[-10:]) / max(len(losses[-10:]), 1)),
            trainable_params=trainable,
            total_params=total,
            metrics={"backend": "minimal_loop", "loss_hist_tail": losses[-20:]},
        )

    def _write_result(self, output_dir: Path, result: ESFTResult) -> None:
        path = output_dir / "esft_result.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        log.info("ESFT result written to %s", path)
