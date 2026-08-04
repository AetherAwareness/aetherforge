"""
Specialization-aware losses — fight expert homogenization during domain FT.

Components:
  1. Activation variance: encourage selected experts to produce diversified outputs
  2. Orthogonality: push expert weight-update directions apart (full FT / adapter)
  3. Mild load-balance aux (optional, usually lower weight than pure LB pretraining)
"""

from __future__ import annotations

from typing import Any, Optional

from aetherforge.utils.logging import get_logger

log = get_logger("training.spec_loss")


class SpecializationLoss:
    def __init__(
        self,
        variance_weight: float = 0.1,
        orthogonality_weight: float = 0.05,
        load_balance_weight: float = 0.01,
    ):
        self.variance_weight = variance_weight
        self.orthogonality_weight = orthogonality_weight
        self.load_balance_weight = load_balance_weight
        self._activation_buffers: list[Any] = []

    def clear(self) -> None:
        self._activation_buffers.clear()

    def register_activations(self, expert_outputs: Any) -> None:
        """Call from hooks with stacked expert activations [E, B, D] or list."""
        self._activation_buffers.append(expert_outputs)

    def variance_term(self) -> Any:
        import torch

        if not self._activation_buffers:
            return torch.tensor(0.0)

        terms = []
        for act in self._activation_buffers:
            if isinstance(act, (list, tuple)):
                try:
                    act = torch.stack([a.float().reshape(a.shape[0], -1) for a in act], dim=0)
                except Exception:
                    continue
            if not hasattr(act, "float"):
                continue
            x = act.float()
            # maximize variance across expert dim → minimize negative variance
            if x.dim() >= 2:
                # var over expert axis 0
                v = x.var(dim=0, unbiased=False).mean()
                terms.append(-v)  # maximize diversity
        if not terms:
            return torch.tensor(0.0)
        return torch.stack(terms).mean() * self.variance_weight

    def orthogonality_term(self, expert_params: list[Any]) -> Any:
        """Encourage pairwise cosine dissimilarity of flattened expert tensors."""
        import torch

        if len(expert_params) < 2 or self.orthogonality_weight <= 0:
            return torch.tensor(0.0)

        vecs = []
        for p in expert_params:
            if p is None:
                continue
            v = p.float().reshape(-1)
            v = v / (v.norm() + 1e-8)
            vecs.append(v)
        if len(vecs) < 2:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=vecs[0].device)
        n = 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                # minimize |cos| → encourage orthogonality
                loss = loss + torch.abs(torch.dot(vecs[i], vecs[j]))
                n += 1
        if n == 0:
            return torch.tensor(0.0)
        return (loss / n) * self.orthogonality_weight

    def load_balance_term(self, routing_probs: Any) -> Any:
        """Switch-style auxiliary LB: encourage uniform expert usage."""
        import torch

        if routing_probs is None or self.load_balance_weight <= 0:
            return torch.tensor(0.0)
        p = routing_probs.float()
        if p.dim() == 1:
            mean = p.mean()
            return ((p - mean) ** 2).mean() * self.load_balance_weight
        # [tokens, experts]
        freq = p.mean(dim=0)
        n = freq.numel()
        # f * P formulation proxy
        return (freq * freq * n).sum() * self.load_balance_weight

    def total(
        self,
        task_loss: Any,
        *,
        expert_params: Optional[list[Any]] = None,
        routing_probs: Any = None,
    ) -> tuple[Any, dict[str, float]]:
        import torch

        var = self.variance_term()
        ortho = self.orthogonality_term(expert_params or [])
        lb = self.load_balance_term(routing_probs)
        # ensure same device as task loss
        if hasattr(task_loss, "device"):
            var = var.to(task_loss.device) if hasattr(var, "to") else var
            ortho = ortho.to(task_loss.device) if hasattr(ortho, "to") else ortho
            lb = lb.to(task_loss.device) if hasattr(lb, "to") else lb

        total = task_loss + var + ortho + lb
        stats = {
            "task": float(task_loss.detach().cpu()) if hasattr(task_loss, "detach") else float(task_loss),
            "variance": float(var.detach().cpu()) if hasattr(var, "detach") else float(var),
            "ortho": float(ortho.detach().cpu()) if hasattr(ortho, "detach") else float(ortho),
            "load_balance": float(lb.detach().cpu()) if hasattr(lb, "detach") else float(lb),
            "total": float(total.detach().cpu()) if hasattr(total, "detach") else float(total),
        }
        return total, stats
