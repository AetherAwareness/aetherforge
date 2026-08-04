"""
Affinity Probe — Stage 0/2 of AetherForge.

Runs domain probe tokens through the MoE and records:
  - routing frequency per expert (layer-local and global)
  - optional gradient contribution magnitude w.r.t. domain loss
  - routing entropy and load-balance statistics

Output feeds ExpertSelector for AGPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from aetherforge.models.loaders import MoEModelBundle
from aetherforge.models.moe_utils import ExpertRef, expert_global_id
from aetherforge.utils.config import AffinityConfig
from aetherforge.utils.logging import get_logger

log = get_logger("affinity.probe")


@dataclass
class AffinityResult:
    """Full probe snapshot for a domain."""

    domain: str
    family: str
    num_experts: int
    num_layers: int
    # shape [num_layers, num_experts] — mean routing probability / count
    routing_freq: np.ndarray
    # shape [num_layers, num_experts] — optional grad contribution (zeros if disabled)
    grad_contrib: np.ndarray
    # combined affinity score [num_layers, num_experts]
    affinity: np.ndarray
    # global expert ranking: list of (layer, expert_idx, score)
    ranked: list[tuple[int, int, float]] = field(default_factory=list)
    entropy_per_layer: list[float] = field(default_factory=list)
    load_balance_cv: float = 0.0
    probe_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def top_k(self, k: int) -> list[tuple[int, int, float]]:
        return self.ranked[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "family": self.family,
            "num_experts": self.num_experts,
            "num_layers": self.num_layers,
            "routing_freq": self.routing_freq.tolist(),
            "grad_contrib": self.grad_contrib.tolist(),
            "affinity": self.affinity.tolist(),
            "ranked": self.ranked[:256],
            "entropy_per_layer": self.entropy_per_layer,
            "load_balance_cv": self.load_balance_cv,
            "probe_tokens": self.probe_tokens,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AffinityResult":
        return cls(
            domain=d["domain"],
            family=d.get("family", "generic_moe"),
            num_experts=d["num_experts"],
            num_layers=d["num_layers"],
            routing_freq=np.array(d["routing_freq"], dtype=np.float64),
            grad_contrib=np.array(d["grad_contrib"], dtype=np.float64),
            affinity=np.array(d["affinity"], dtype=np.float64),
            ranked=[tuple(x) for x in d.get("ranked", [])],  # type: ignore[misc]
            entropy_per_layer=list(d.get("entropy_per_layer", [])),
            load_balance_cv=float(d.get("load_balance_cv", 0.0)),
            probe_tokens=int(d.get("probe_tokens", 0)),
            metadata=dict(d.get("metadata", {})),
        )


def _layer_entropy(freq_row: np.ndarray) -> float:
    p = freq_row.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log(p + 1e-12)).sum())


class AffinityProbe:
    """
    Probe-driven expert affinity measurement.

    Implementation strategy (robust across HF MoE variants):
      1. Prefer hooking router/gate modules to capture routing weights.
      2. Fall back to uniform pseudo-affinity + activation-norm proxy if hooks fail.
      3. Optional: one backward pass for gradient contribution ranking.
    """

    def __init__(self, bundle: MoEModelBundle, config: AffinityConfig, domain: str = "domain"):
        self.bundle = bundle
        self.config = config
        self.domain = domain
        self.arch = bundle.arch

    def run(
        self,
        texts: list[str],
        *,
        max_length: int = 512,
        batch_size: int = 1,
    ) -> AffinityResult:
        import torch

        model = self.bundle.model
        tokenizer = self.bundle.tokenizer
        n_layers = max(self.arch.num_layers, 1)
        n_exp = max(self.arch.num_experts, 1)

        routing = np.zeros((n_layers, n_exp), dtype=np.float64)
        hooks = []
        captured_batches = {"n": 0}

        def _make_hook(layer_idx: int):
            def hook(_mod, _inp, out):
                # out may be logits [*, n_exp] or tuple
                t = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(t):
                    return
                with torch.no_grad():
                    x = t.detach().float()
                    # softmax if looks like logits
                    if x.dim() >= 2 and x.shape[-1] == n_exp:
                        probs = torch.softmax(x.reshape(-1, n_exp), dim=-1)
                        routing[layer_idx] += probs.sum(dim=0).cpu().numpy()
                        captured_batches["n"] += probs.shape[0]
                    elif x.dim() >= 1 and x.numel() == n_exp:
                        probs = torch.softmax(x.reshape(-1), dim=-1)
                        routing[layer_idx] += probs.cpu().numpy()
                        captured_batches["n"] += 1

            return hook

        # Attach hooks to router-like modules
        router_names = []
        for name, mod in model.named_modules():
            low = name.lower()
            if any(k in low for k in ("gate", "router")) and any(
                k in low for k in ("mlp", "moe", "experts")
            ):
                # extract layer index
                import re

                m = re.search(r"layers\.(\d+)", name)
                layer_idx = int(m.group(1)) if m else 0
                if layer_idx >= n_layers:
                    continue
                try:
                    h = mod.register_forward_hook(_make_hook(layer_idx))
                    hooks.append(h)
                    router_names.append(name)
                except Exception:
                    continue

        log.info(
            "Affinity probe: %d texts, %d router hooks, layers=%d experts=%d",
            len(texts),
            len(hooks),
            n_layers,
            n_exp,
        )

        model_was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        probe_tokens = 0

        texts = texts[: self.config.probe_size]
        try:
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    enc = tokenizer(
                        batch,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding=True,
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    probe_tokens += int(enc["input_ids"].numel())
                    try:
                        model(**enc)
                    except Exception as e:
                        log.debug("Forward probe batch failed: %s", e)
        finally:
            for h in hooks:
                h.remove()
            if model_was_training:
                model.train()

        # Normalize routing
        if routing.sum() > 0:
            # keep raw counts for load balance; affinity uses row-normalized
            pass
        else:
            # Fallback: uniform prior (still allows pipeline to run offline)
            log.warning(
                "No router activations captured; using uniform affinity prior "
                "(hooks may not match this architecture — refine patterns later)"
            )
            routing[:] = 1.0

        grad_contrib = np.zeros_like(routing)
        if self.config.use_gradient_contribution and len(texts) > 0:
            grad_contrib = self._grad_contribution(texts[: min(32, len(texts))], max_length)

        # Combined affinity: 0.7 * route_norm + 0.3 * grad_norm
        route_norm = routing / (routing.sum(axis=1, keepdims=True) + 1e-12)
        gsum = grad_contrib.sum(axis=1, keepdims=True)
        grad_norm = np.where(gsum > 0, grad_contrib / (gsum + 1e-12), 0.0)
        if grad_norm.sum() > 0:
            affinity = 0.7 * route_norm + 0.3 * grad_norm
        else:
            affinity = route_norm

        ranked: list[tuple[int, int, float]] = []
        for li in range(n_layers):
            for ei in range(n_exp):
                ranked.append((li, ei, float(affinity[li, ei])))
        ranked.sort(key=lambda x: x[2], reverse=True)

        entropies = [_layer_entropy(routing[li]) for li in range(n_layers)]
        flat = routing.sum(axis=0) if n_layers else routing.flatten()
        mean = flat.mean() if flat.size else 1.0
        std = flat.std() if flat.size else 0.0
        cv = float(std / (mean + 1e-12))

        result = AffinityResult(
            domain=self.domain,
            family=self.arch.family,
            num_experts=n_exp,
            num_layers=n_layers,
            routing_freq=routing,
            grad_contrib=grad_contrib,
            affinity=affinity,
            ranked=ranked,
            entropy_per_layer=entropies,
            load_balance_cv=cv,
            probe_tokens=probe_tokens,
            metadata={
                "router_hooks": router_names[:32],
                "captured_token_rows": captured_batches["n"],
                "probe_size_requested": self.config.probe_size,
                "texts_used": len(texts),
            },
        )
        log.info(
            "Affinity done: top expert L%d/E%d score=%.4f entropy_mean=%.3f cv=%.3f",
            ranked[0][0] if ranked else -1,
            ranked[0][1] if ranked else -1,
            ranked[0][2] if ranked else 0.0,
            float(np.mean(entropies)) if entropies else 0.0,
            cv,
        )
        return result

    def _grad_contribution(self, texts: list[str], max_length: int) -> np.ndarray:
        """Cheap gradient-magnitude proxy per expert param group."""
        import torch

        model = self.bundle.model
        tokenizer = self.bundle.tokenizer
        n_layers = max(self.arch.num_layers, 1)
        n_exp = max(self.arch.num_experts, 1)
        contrib = np.zeros((n_layers, n_exp), dtype=np.float64)

        # Only if model can train and has enough free VRAM — best-effort
        try:
            model.train()
            device = next(model.parameters()).device
            enc = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = enc["input_ids"].clone()
            if "attention_mask" in enc:
                labels = labels.masked_fill(enc["attention_mask"] == 0, -100)
            out = model(**enc, labels=labels)
            loss = out.loss if hasattr(out, "loss") else out[0]
            loss.backward()

            import re

            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                m = re.search(r"layers\.(\d+).*experts\.(\d+)", name)
                if not m:
                    continue
                li, ei = int(m.group(1)), int(m.group(2))
                if li < n_layers and ei < n_exp:
                    contrib[li, ei] += float(p.grad.detach().float().abs().mean().cpu())

            model.zero_grad(set_to_none=True)
        except Exception as e:
            log.debug("Grad contribution probe skipped: %s", e)
            contrib[:] = 0.0
        finally:
            model.eval()

        return contrib


def affinity_to_expert_refs(
    result: AffinityResult,
    experts: list[ExpertRef],
    selected: list[tuple[int, int, float]],
) -> list[ExpertRef]:
    """Map (layer, expert_idx) selections back to ExpertRef objects when available."""
    index = {(e.layer_idx, e.expert_idx): e for e in experts}
    out: list[ExpertRef] = []
    for li, ei, _score in selected:
        if (li, ei) in index:
            out.append(index[(li, ei)])
        else:
            out.append(
                ExpertRef(
                    layer_idx=li,
                    expert_idx=ei,
                    module_name=f"model.layers.{li}.mlp.experts.{ei}",
                    family=result.family,
                )
            )
    return out
