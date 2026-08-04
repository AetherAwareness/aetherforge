"""Expert utilization monitoring for elastic lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from aetherforge.affinity.probe import AffinityResult
from aetherforge.utils.logging import get_logger

log = get_logger("lifecycle.monitor")


@dataclass
class UtilizationReport:
    mean: float
    std: float
    per_expert: list[dict[str, Any]] = field(default_factory=list)
    low: list[tuple[int, int, float]] = field(default_factory=list)
    high: list[tuple[int, int, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "low": self.low[:64],
            "high": self.high[:64],
            "n_experts": len(self.per_expert),
        }


class ExpertUtilizationMonitor:
    def __init__(self, low_threshold: float = 0.15, high_threshold: float = 3.0):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def from_affinity(self, affinity: AffinityResult) -> UtilizationReport:
        freq = affinity.routing_freq
        flat_mean = float(freq.mean()) if freq.size else 0.0
        flat_std = float(freq.std()) if freq.size else 0.0
        per = []
        low = []
        high = []
        for li in range(freq.shape[0]):
            for ei in range(freq.shape[1]):
                v = float(freq[li, ei])
                rel = v / (flat_mean + 1e-12)
                per.append({"layer": li, "expert": ei, "freq": v, "rel": rel})
                if rel <= self.low_threshold:
                    low.append((li, ei, rel))
                if rel >= self.high_threshold:
                    high.append((li, ei, rel))
        low.sort(key=lambda x: x[2])
        high.sort(key=lambda x: x[2], reverse=True)
        report = UtilizationReport(
            mean=flat_mean, std=flat_std, per_expert=per, low=low, high=high
        )
        log.info(
            "Utilization: mean=%.4f low=%d high=%d",
            flat_mean,
            len(low),
            len(high),
        )
        return report
