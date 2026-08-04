"""Expert ranking helpers for AGPS progressive tiers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from aetherforge.affinity.probe import AffinityResult


def rank_experts(
    result: AffinityResult,
    top_k: Optional[int] = None,
    top_k_fraction: Optional[float] = None,
    min_score: float = 0.0,
) -> list[tuple[int, int, float]]:
    """
    Return ranked (layer, expert_idx, score) list.

    Prefer fixed top_k; else fraction of total (layer * experts); else all above min_score.
    """
    ranked = [r for r in result.ranked if r[2] >= min_score]
    total = max(result.num_layers * result.num_experts, 1)

    if top_k_fraction is not None and top_k_fraction > 0:
        k = max(1, int(total * top_k_fraction))
        return ranked[:k]
    if top_k is not None and top_k > 0:
        return ranked[:top_k]
    return ranked


def progressive_tiers(
    result: AffinityResult,
    tier_sizes: list[int],
) -> list[list[tuple[int, int, float]]]:
    """
    Split ranked experts into progressive unfreeze tiers.

    Example tier_sizes=[16, 32, 64] → first train top-16, then expand, etc.
    """
    remaining = list(result.ranked)
    tiers: list[list[tuple[int, int, float]]] = []
    used: set[tuple[int, int]] = set()
    for size in tier_sizes:
        tier: list[tuple[int, int, float]] = []
        for item in remaining:
            key = (item[0], item[1])
            if key in used:
                continue
            tier.append(item)
            used.add(key)
            if len(tier) >= size:
                break
        tiers.append(tier)
    return tiers


def overloaded_experts(
    result: AffinityResult,
    threshold: float = 3.0,
) -> list[tuple[int, int, float]]:
    """Experts with routing freq > threshold * mean (candidates for mitosis)."""
    freq = result.routing_freq
    mean = float(freq.mean()) if freq.size else 0.0
    if mean <= 0:
        return []
    out: list[tuple[int, int, float]] = []
    for li in range(freq.shape[0]):
        for ei in range(freq.shape[1]):
            rel = float(freq[li, ei] / mean)
            if rel >= threshold:
                out.append((li, ei, rel))
    out.sort(key=lambda x: x[2], reverse=True)
    return out


def underused_experts(
    result: AffinityResult,
    threshold: float = 0.15,
) -> list[tuple[int, int, float]]:
    """Experts below threshold * mean utilization (rebirth candidates)."""
    freq = result.routing_freq
    mean = float(freq.mean()) if freq.size else 0.0
    if mean <= 0:
        return []
    out: list[tuple[int, int, float]] = []
    for li in range(freq.shape[0]):
        for ei in range(freq.shape[1]):
            rel = float(freq[li, ei] / (mean + 1e-12))
            if rel <= threshold:
                out.append((li, ei, rel))
    out.sort(key=lambda x: x[2])
    return out
