"""Multi-stage data quality gates for domain corpora."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from aetherforge.utils.config import QualityGatesConfig
from aetherforge.utils.logging import get_logger

log = get_logger("data.quality")


@dataclass
class GateReport:
    total_in: int
    total_out: int
    dropped: dict[str, int] = field(default_factory=dict)
    diversity: float = 0.0
    diversity_components: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "dropped": self.dropped,
            "diversity": self.diversity,
            "diversity_components": self.diversity_components,
            "passed": self.passed,
            "notes": self.notes,
        }


_TOXIC_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"\bkill yourself\b",
        r"\bhow to make a bomb\b",
        r"\bmake meth\b",
    ]
]

_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
]


class QualityGateRunner:
    def __init__(self, config: QualityGatesConfig):
        self.config = config

    def filter_records(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], GateReport]:
        cfg = self.config
        dropped: dict[str, int] = {
            "length": 0,
            "dedupe": 0,
            "toxicity": 0,
            "pii": 0,
            "empty": 0,
        }
        seen_hashes: set[str] = set()
        kept: list[dict[str, Any]] = []

        for rec in records:
            text = self._text_of(rec)
            if not text or not text.strip():
                dropped["empty"] += 1
                continue
            if len(text) < cfg.min_length:
                dropped["length"] += 1
                continue
            if len(text) > cfg.max_length:
                text = text[: cfg.max_length]
                rec = {**rec, "text": text}

            norm = re.sub(r"\s+", " ", text.strip().lower())
            h = hashlib.sha256(norm.encode()).hexdigest()
            # near-dedupe: hash of first 80 tokens
            tokens = norm.split()
            near_key = " ".join(tokens[:80])
            near = hashlib.sha256(near_key.encode()).hexdigest()
            if h in seen_hashes or near in seen_hashes:
                dropped["dedupe"] += 1
                continue
            seen_hashes.add(h)
            seen_hashes.add(near)

            if any(p.search(text) for p in _TOXIC_PATTERNS):
                dropped["toxicity"] += 1
                continue
            if any(p.search(text) for p in _PII_PATTERNS):
                dropped["pii"] += 1
                continue

            kept.append(rec if "text" in rec else {**rec, "text": text})

        diversity, components = self._diversity(kept)
        report = GateReport(
            total_in=len(records),
            total_out=len(kept),
            dropped=dropped,
            diversity=diversity,
            diversity_components=components,
            passed=True,
            notes=[],
        )
        if diversity < cfg.min_diversity and len(kept) >= 20:
            report.notes.append(
                f"diversity {diversity:.3f} < min {cfg.min_diversity} "
                f"(components={{{', '.join(f'{k}={v:.2f}' for k,v in components.items())}}}) "
                "— consider more sources / topics"
            )
            # Soft fail for synthetic scaffolds; hard fail only if extremely collapsed
            if diversity < cfg.min_diversity * 0.35:
                report.passed = False
                report.notes.append("hard_fail_extreme_diversity_collapse")
            else:
                report.notes.append("soft_pass_with_diversity_warning")

        log.info(
            "Quality gates: %d → %d (dropped=%s diversity=%.3f components=%s)",
            report.total_in,
            report.total_out,
            dropped,
            diversity,
            components,
        )
        return kept, report

    def _text_of(self, rec: dict[str, Any]) -> str:
        if "text" in rec:
            return str(rec["text"])
        if "prompt" in rec and "completion" in rec:
            return f"{rec['prompt']}{rec['completion']}"
        if "messages" in rec:
            return " ".join(str(m.get("content", "")) for m in rec["messages"])
        return str(rec)

    def _diversity(self, records: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
        """
        Composite diversity in [0, 1]:
          - type-token ratio (unigram, length-adjusted)
          - unique bigram ratio
          - topic/source entropy (if fields present)
          - length variance (normalized)
          - unique-document ratio (near-dup resistance)
        """
        if len(records) < 2:
            return 1.0, {
                "ttr": 1.0,
                "bigram": 1.0,
                "field_entropy": 1.0,
                "length_cv": 1.0,
                "unique_docs": 1.0,
            }

        all_tokens: list[str] = []
        bigrams: list[str] = []
        lengths: list[int] = []
        topics: list[str] = []
        sources: list[str] = []
        doc_hashes: set[str] = set()

        for rec in records[:3000]:
            t = self._text_of(rec).lower()
            words = re.findall(r"[a-z0-9']+", t)
            all_tokens.extend(words)
            lengths.append(len(words))
            for i in range(len(words) - 1):
                bigrams.append(f"{words[i]}_{words[i+1]}")
            topics.append(str(rec.get("topic") or rec.get("domain") or "na"))
            sources.append(str(rec.get("source") or "na"))
            # fingerprint on content body (skip shared headers)
            body = " ".join(words[10:80]) if len(words) > 20 else " ".join(words)
            doc_hashes.add(hashlib.sha256(body.encode()).hexdigest()[:16])

        def _ttr(toks: list[str]) -> float:
            if not toks:
                return 0.0
            # Herdan/Heaps-ish: unique / n^0.6 → softer than raw TTR
            return min(1.0, len(set(toks)) / max(len(toks) ** 0.6, 1.0))

        def _unique_ratio(items: list[str]) -> float:
            if not items:
                return 0.0
            return min(1.0, len(set(items)) / max(len(items) * 0.25, 1.0))

        def _entropy_norm(labels: list[str]) -> float:
            if not labels:
                return 0.0
            c = Counter(labels)
            n = sum(c.values())
            if n == 0 or len(c) <= 1:
                return 0.0
            ent = -sum((v / n) * math.log(v / n + 1e-12) for v in c.values())
            max_ent = math.log(len(c))
            if max_ent <= 0:
                return 0.0
            return float(max(0.0, min(1.0, ent / max_ent)))

        ttr = _ttr(all_tokens)
        big = _unique_ratio(bigrams)
        field_ent = 0.5 * _entropy_norm(topics) + 0.5 * _entropy_norm(sources)
        if lengths:
            mean_l = sum(lengths) / len(lengths)
            var = sum((x - mean_l) ** 2 for x in lengths) / len(lengths)
            cv = math.sqrt(var) / (mean_l + 1e-9)
            length_div = min(1.0, cv / 0.35)
        else:
            length_div = 0.0
        unique_docs = min(1.0, len(doc_hashes) / max(len(records[:3000]), 1))

        score = (
            0.20 * ttr
            + 0.25 * big
            + 0.20 * field_ent
            + 0.10 * length_div
            + 0.25 * unique_docs
        )
        components = {
            "ttr": round(ttr, 4),
            "bigram": round(big, 4),
            "field_entropy": round(field_ent, 4),
            "length_cv": round(length_div, 4),
            "unique_docs": round(unique_docs, 4),
        }
        return float(min(1.0, max(0.0, score))), components
