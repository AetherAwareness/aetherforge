"""
Preference & outcome alignment — Stage 5.

Supports DPO-style pair training when peft/trl available; otherwise
records Trajectory Hive Distillation pairs for offline processing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aetherforge.utils.logging import get_logger

log = get_logger("training.preference")


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    source: str = "thd"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferenceResult:
    output_dir: str
    pairs_used: int
    method: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "pairs_used": self.pairs_used,
            "method": self.method,
            "metrics": self.metrics,
        }


_LIVE_CHOSEN_SYS = (
    "You are the winning specialist. Give the better, more grounded answer. "
    "Structured. Explicit assumptions, discriminators, stop rules. No overclaim."
)
_LIVE_REJECTED_SYS = (
    "You are the losing specialist. Give a weaker answer: vaguer, missing stop "
    "conditions, slightly overconfident, thinner evidence. Still on-topic."
)


class PreferenceAligner:
    """Stage-5 preference alignment (DPO / GRPO / offline export)."""

    def __init__(self, method: str = "dpo"):
        self.method = method

    def synthesize_live(
        self,
        prompts: list[str],
        *,
        llm_fn: Any,
        domain: str = "general",
        source: str = "thd_live",
    ) -> list[PreferencePair]:
        """Mint chosen/rejected pairs via OpenAI-compat LLM. Caller skips on dry-run."""
        pairs: list[PreferencePair] = []
        for prompt in prompts:
            if not prompt or not str(prompt).strip():
                continue
            user = f"[{domain}] {prompt}"
            try:
                chosen = str(llm_fn(_LIVE_CHOSEN_SYS, user) or "").strip()
                rejected = str(llm_fn(_LIVE_REJECTED_SYS, user) or "").strip()
            except Exception as e:
                log.warning("live THD synthesize failed: %s", e)
                continue
            if not chosen or not rejected or chosen == rejected:
                continue
            pairs.append(
                PreferencePair(
                    prompt=user,
                    chosen=chosen,
                    rejected=rejected,
                    source=source,
                    meta={"live": True, "domain": domain},
                )
            )
        log.info("live THD synthesized %d pairs from %d prompts", len(pairs), len(prompts))
        return pairs

    def run(
        self,
        pairs: list[PreferencePair],
        output_dir: str | Path,
        model: Any = None,
        tokenizer: Any = None,
    ) -> PreferenceResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Always persist pairs for audit / offline TRL
        pairs_path = output_dir / "preference_pairs.jsonl"
        with open(pairs_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(
                    json.dumps(
                        {
                            "prompt": p.prompt,
                            "chosen": p.chosen,
                            "rejected": p.rejected,
                            "source": p.source,
                            "meta": p.meta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        method_used = "export_only"
        metrics: dict[str, Any] = {"pairs_path": str(pairs_path)}

        if model is not None and pairs and self.method in ("dpo", "grpo"):
            try:
                metrics.update(self._try_trl_dpo(model, tokenizer, pairs, output_dir))
                method_used = "trl_dpo"
            except Exception as e:
                log.warning("TRL DPO unavailable (%s); pairs exported for offline run", e)
                method_used = "export_only"

        result = PreferenceResult(
            output_dir=str(output_dir),
            pairs_used=len(pairs),
            method=method_used,
            metrics=metrics,
        )
        with open(output_dir / "preference_result.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        return result

    def _try_trl_dpo(
        self,
        model: Any,
        tokenizer: Any,
        pairs: list[PreferencePair],
        output_dir: Path,
    ) -> dict[str, Any]:
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer  # type: ignore

        data = Dataset.from_list(
            [
                {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}
                for p in pairs
            ]
        )
        args = DPOConfig(
            output_dir=str(output_dir / "dpo"),
            per_device_train_batch_size=1,
            max_steps=min(100, len(pairs)),
            learning_rate=5e-6,
            logging_steps=5,
            report_to=[],
        )
        trainer = DPOTrainer(
            model=model,
            args=args,
            train_dataset=data,
            processing_class=tokenizer,
        )
        out = trainer.train()
        trainer.save_model(str(output_dir / "dpo"))
        return {"train_metrics": dict(getattr(out, "metrics", {}) or {})}
