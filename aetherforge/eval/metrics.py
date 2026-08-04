"""Core metrics for MoE reliability scoring."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def routing_entropy(freq: np.ndarray, axis: int = -1) -> float:
    """Mean Shannon entropy of routing distribution."""
    x = np.asarray(freq, dtype=np.float64)
    if x.ndim == 1:
        p = x / (x.sum() + 1e-12)
        p = p[p > 0]
        return float(-(p * np.log(p + 1e-12)).sum())
    ent = []
    for row in x.reshape(-1, x.shape[-1]):
        p = row / (row.sum() + 1e-12)
        p = p[p > 0]
        ent.append(float(-(p * np.log(p + 1e-12)).sum()))
    return float(np.mean(ent)) if ent else 0.0


def load_balance_cv(freq: np.ndarray) -> float:
    x = np.asarray(freq, dtype=np.float64).flatten()
    if x.size == 0:
        return 0.0
    mean = x.mean()
    return float(x.std() / (mean + 1e-12))


def domain_score_proxy(texts: list[str], keywords: list[str]) -> float:
    """Cheap offline proxy: fraction of eval texts containing domain keywords."""
    if not texts:
        return 0.0
    hits = 0
    kws = [k.lower() for k in keywords]
    for t in texts:
        tl = t.lower()
        if any(k in tl for k in kws):
            hits += 1
    return hits / len(texts)


def perplexity_proxy_loss(losses: list[float]) -> float:
    if not losses:
        return float("inf")
    return float(np.exp(np.mean(losses)))
