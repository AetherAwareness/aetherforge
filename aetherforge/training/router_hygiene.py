"""
Router Hygiene — Stage 4.

After expert adaptation, freeze all experts and briefly recalibrate the router
so anticipatory / domain-aware routing re-aligns without overwriting specialist weights.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from aetherforge.models.loaders import MoEModelBundle
from aetherforge.models.moe_utils import (
    freeze_all_parameters,
    list_router_module_names,
    unfreeze_parameters_by_name,
    count_trainable_parameters,
)
from aetherforge.utils.audit import AuditLog
from aetherforge.utils.config import TrainingConfig
from aetherforge.utils.logging import get_logger

log = get_logger("training.router_hygiene")


@dataclass
class RouterHygieneResult:
    output_dir: str
    steps: int
    final_loss: float
    router_modules: list[str]
    duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "steps": self.steps,
            "final_loss": self.final_loss,
            "router_modules": self.router_modules,
            "duration_sec": self.duration_sec,
        }


class RouterHygieneTrainer:
    def __init__(
        self,
        bundle: MoEModelBundle,
        train_cfg: TrainingConfig,
        audit: Optional[AuditLog] = None,
    ):
        self.bundle = bundle
        self.cfg = train_cfg
        self.audit = audit

    def run(self, texts: list[str], output_dir: str | Path) -> RouterHygieneResult:
        import torch
        from torch.optim import AdamW

        t0 = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self.bundle.model
        tokenizer = self.bundle.tokenizer
        freeze_all_parameters(model)

        router_mods = list_router_module_names(model, self.bundle.arch)
        # unfreeze router params
        router_params = []
        for name, p in model.named_parameters():
            if any(name == r or name.startswith(r + ".") for r in router_mods):
                p.requires_grad = True
                router_params.append(p)

        if not router_params:
            # fallback: unfreeze anything with gate/router in name
            for name, p in model.named_parameters():
                low = name.lower()
                if "gate" in low or "router" in low:
                    p.requires_grad = True
                    router_params.append(p)
                    router_mods.append(name)

        trainable, total = count_trainable_parameters(model)
        log.info(
            "Router hygiene: %d router tensors, trainable=%s / %s",
            len(router_params),
            f"{trainable:,}",
            f"{total:,}",
        )

        if not router_params or not texts:
            result = RouterHygieneResult(
                output_dir=str(output_dir),
                steps=0,
                final_loss=0.0,
                router_modules=router_mods,
                duration_sec=time.time() - t0,
            )
            self._save(output_dir, result)
            return result

        device = next(model.parameters()).device
        opt = AdamW(router_params, lr=self.cfg.router_learning_rate)
        steps = min(self.cfg.router_hygiene_steps, max(1, len(texts) * 2))
        model.train()
        losses: list[float] = []

        # Capture router logits for entropy regularization (anti-collapse)
        router_logits: list[Any] = []

        def _router_hook(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t):
                router_logits.append(t)

        hooks = []
        for name, mod in model.named_modules():
            low = name.lower()
            if any(k in low for k in ("gate", "router")) and any(
                k in low for k in ("mlp", "moe")
            ):
                try:
                    hooks.append(mod.register_forward_hook(_router_hook))
                except Exception:
                    continue

        entropy_w = 0.01
        try:
            for step in range(steps):
                router_logits.clear()
                text = texts[step % len(texts)]
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = enc["input_ids"].clone()
                out = model(**enc, labels=labels)
                loss = out.loss
                # Maximize mean routing entropy → minimize negative entropy
                reg = torch.tensor(0.0, device=device)
                if router_logits:
                    ents = []
                    for logits in router_logits:
                        x = logits.float()
                        if x.shape[-1] < 2:
                            continue
                        p = torch.softmax(x.reshape(-1, x.shape[-1]), dim=-1)
                        ent = -(p * (p + 1e-12).log()).sum(dim=-1).mean()
                        ents.append(ent)
                    if ents:
                        # target: keep entropy from collapsing (maximize)
                        mean_ent = torch.stack(ents).mean()
                        reg = -mean_ent  # added to loss with small weight
                loss = loss + entropy_w * reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(router_params, self.cfg.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
                if (step + 1) % max(1, self.cfg.logging_steps) == 0:
                    log.info("router hygiene step %d loss=%.4f", step + 1, losses[-1])
        finally:
            for h in hooks:
                h.remove()

        # save
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(output_dir))
        result = RouterHygieneResult(
            output_dir=str(output_dir),
            steps=steps,
            final_loss=float(sum(losses[-5:]) / max(len(losses[-5:]), 1)),
            router_modules=sorted(set(router_mods)),
            duration_sec=time.time() - t0,
        )
        self._save(output_dir, result)
        if self.audit:
            self.audit.record("router_hygiene", "complete", result.to_dict(), checkpoint=str(output_dir))
        return result

    def _save(self, output_dir: Path, result: RouterHygieneResult) -> None:
        with open(output_dir / "router_hygiene_result.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
