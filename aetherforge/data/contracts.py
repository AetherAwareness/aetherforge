"""
Per-sector dataset contracts — fail or hard-warn train eligibility.

Contracts (checkable, industry-agnostic):
  - min_samples
  - min_real_fraction (non-synth)
  - max_synth_fraction
  - min_unique_ratio (unique text prefixes)
  - min_matched (soft-assign matches, optional)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from aetherforge.utils.logging import get_logger

log = get_logger("data.contracts")

ContractMode = Literal["block", "warn", "off"]


def _text_of(rec: dict[str, Any]) -> str:
    if not isinstance(rec, dict):
        return str(rec)
    if rec.get("text"):
        return str(rec["text"])
    if rec.get("prompt"):
        return str(rec.get("prompt", "")) + "\n" + str(rec.get("response", ""))
    return str(rec)[:500]


def _is_synth(rec: dict[str, Any]) -> bool:
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    src = str(meta.get("source") or rec.get("source") or "").lower()
    if "synth" in src or src in ("sector_synth", "self_instruct", "synthetic"):
        return True
    if meta.get("sector_id") and src == "sector_synth":
        return True
    if rec.get("synthetic") is True:
        return True
    return False


def _is_shared_mix(rec: dict[str, Any]) -> bool:
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    return bool(meta.get("shared_mix"))


def _is_matched(rec: dict[str, Any]) -> bool:
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    if meta.get("match_score") is not None:
        return float(meta["match_score"]) > 0
    if meta.get("assigned_sector"):
        return True
    return not _is_synth(rec) and not _is_shared_mix(rec)


@dataclass
class DataContractSpec:
    min_samples: int = 8
    min_real_fraction: float = 0.15  # non-synth
    max_synth_fraction: float = 0.85
    min_unique_ratio: float = 0.35
    min_matched: int = 0  # 0 = don't enforce
    mode: ContractMode = "warn"  # block | warn | off

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "min_real_fraction": self.min_real_fraction,
            "max_synth_fraction": self.max_synth_fraction,
            "min_unique_ratio": self.min_unique_ratio,
            "min_matched": self.min_matched,
            "mode": self.mode,
        }


@dataclass
class DataContractReport:
    group_id: str
    name: str
    passed: bool
    status: str  # pass | warn | fail
    n_train: int
    n_real: int
    n_synth: int
    n_shared: int
    n_matched: int
    unique_ratio: float
    real_fraction: float
    synth_fraction: float
    violations: list[str] = field(default_factory=list)
    train_eligible: bool = True
    spec: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aetherforge.data_contract.v1",
            "group_id": self.group_id,
            "name": self.name,
            "passed": self.passed,
            "status": self.status,
            "n_train": self.n_train,
            "n_real": self.n_real,
            "n_synth": self.n_synth,
            "n_shared": self.n_shared,
            "n_matched": self.n_matched,
            "unique_ratio": round(self.unique_ratio, 4),
            "real_fraction": round(self.real_fraction, 4),
            "synth_fraction": round(self.synth_fraction, 4),
            "violations": self.violations,
            "train_eligible": self.train_eligible,
            "spec": self.spec,
        }


def evaluate_sector_contract(
    records: list[dict[str, Any]],
    *,
    group_id: str,
    name: str = "",
    spec: Optional[DataContractSpec] = None,
) -> DataContractReport:
    spec = spec or DataContractSpec()
    if spec.mode == "off":
        return DataContractReport(
            group_id=group_id,
            name=name,
            passed=True,
            status="pass",
            n_train=len(records),
            n_real=len(records),
            n_synth=0,
            n_shared=0,
            n_matched=len(records),
            unique_ratio=1.0,
            real_fraction=1.0,
            synth_fraction=0.0,
            train_eligible=True,
            spec=spec.to_dict(),
        )

    n = len(records)
    n_synth = sum(1 for r in records if _is_synth(r))
    n_shared = sum(1 for r in records if _is_shared_mix(r))
    n_matched = sum(1 for r in records if _is_matched(r) and not _is_synth(r))
    n_real = n - n_synth
    texts = [_text_of(r)[:200] for r in records]
    unique_ratio = (len(set(texts)) / n) if n else 0.0
    real_fraction = (n_real / n) if n else 0.0
    synth_fraction = (n_synth / n) if n else 1.0

    violations: list[str] = []
    if n < spec.min_samples:
        violations.append(f"n_train {n} < min_samples {spec.min_samples}")
    if real_fraction < spec.min_real_fraction and n > 0:
        violations.append(
            f"real_fraction {real_fraction:.2f} < min_real_fraction {spec.min_real_fraction}"
        )
    if synth_fraction > spec.max_synth_fraction and n > 0:
        violations.append(
            f"synth_fraction {synth_fraction:.2f} > max_synth_fraction {spec.max_synth_fraction}"
        )
    if unique_ratio < spec.min_unique_ratio and n > 0:
        violations.append(
            f"unique_ratio {unique_ratio:.2f} < min_unique_ratio {spec.min_unique_ratio}"
        )
    if spec.min_matched > 0 and n_matched < spec.min_matched:
        violations.append(f"n_matched {n_matched} < min_matched {spec.min_matched}")

    passed = len(violations) == 0
    if passed:
        status = "pass"
        train_eligible = True
    elif spec.mode == "warn":
        status = "warn"
        train_eligible = True
    else:
        status = "fail"
        train_eligible = False

    report = DataContractReport(
        group_id=group_id,
        name=name,
        passed=passed,
        status=status,
        n_train=n,
        n_real=n_real,
        n_synth=n_synth,
        n_shared=n_shared,
        n_matched=n_matched,
        unique_ratio=unique_ratio,
        real_fraction=real_fraction,
        synth_fraction=synth_fraction,
        violations=violations,
        train_eligible=train_eligible,
        spec=spec.to_dict(),
    )
    if violations:
        log.warning(
            "Data contract %s for %s: %s",
            status,
            name or group_id,
            "; ".join(violations),
        )
    return report
