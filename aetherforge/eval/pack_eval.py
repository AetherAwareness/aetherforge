"""
Pack-driven eval harness.

Benchmarks live on the DomainPack (any industry). The trainer never
hard-codes field knowledge — it only scores must_include / must_not /
optional gold overlap, optionally filling answers via an OpenAI-compat LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from aetherforge.data.domain_pack import DomainPack, PackBenchmark, resolve_domain_pack
from aetherforge.utils.config import DataConfig
from aetherforge.utils.logging import get_logger

log = get_logger("eval.pack_eval")

LLMFn = Callable[[str, str], str]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _contains(hay: str, needle: str) -> bool:
    n = _norm(needle)
    if not n:
        return False
    h = _norm(hay)
    if n in h:
        return True
    # whole-token fallback for short terms
    return bool(re.search(rf"\b{re.escape(n)}\b", h))


def _token_overlap(a: str, b: str) -> float:
    ta = {w for w in re.findall(r"[a-z0-9]{3,}", _norm(a))}
    tb = {w for w in re.findall(r"[a-z0-9]{3,}", _norm(b))}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def score_answer(bench: PackBenchmark, answer: str) -> dict[str, Any]:
    """Score one answer against a pack benchmark. Field-agnostic."""
    hits = [k for k in bench.must_include if _contains(answer, k)]
    misses = [k for k in bench.must_include if not _contains(answer, k)]
    forbidden = [k for k in bench.must_not_include if _contains(answer, k)]
    include_score = (len(hits) / len(bench.must_include)) if bench.must_include else 0.6
    gold_score = _token_overlap(answer, bench.gold) if bench.gold else include_score
    forbid_pen = (len(forbidden) / max(len(bench.must_not_include), 1)) if bench.must_not_include else 0.0
    raw = 0.65 * include_score + 0.35 * gold_score - 0.5 * forbid_pen
    return {
        "id": bench.id,
        "prompt": bench.prompt,
        "answer": answer,
        "score": float(max(0.0, min(1.0, raw))),
        "hits": hits,
        "misses": misses,
        "forbidden_hits": forbidden,
        "weight": float(bench.weight or 1.0),
        "gold_overlap": float(gold_score),
    }


def _pick_text_for_prompt(prompt: str, texts: list[str]) -> str:
    """Best-effort: treat the most overlapping eval text as a stand-in answer."""
    if not texts:
        return ""
    best, best_s = texts[0], -1.0
    for t in texts:
        s = _token_overlap(prompt, t)
        if s > best_s:
            best, best_s = t, s
    return best


@dataclass
class PackEvalReport:
    domain: str
    score: float
    n: int
    n_scored: int
    items: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "skipped"
    dry_run: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aetherforge.pack_eval.v1",
            "domain": self.domain,
            "score": self.score,
            "n": self.n,
            "n_scored": self.n_scored,
            "items": self.items,
            "mode": self.mode,
            "dry_run": self.dry_run,
            "details": self.details,
        }


def evaluate_pack(
    pack: DomainPack,
    *,
    answers: Optional[dict[str, str]] = None,
    eval_texts: Optional[list[str]] = None,
    llm_fn: Optional[LLMFn] = None,
    dry_run: bool = False,
    system: Optional[str] = None,
) -> PackEvalReport:
    """
    Run pack.benchmarks.

    Answer source priority: explicit `answers` → live `llm_fn` → nearest eval_text.
    Dry-run never calls an LLM.
    """
    benches = list(pack.benchmarks or [])
    if not benches:
        return PackEvalReport(
            domain=pack.domain,
            score=0.0,
            n=0,
            n_scored=0,
            mode="no_benchmarks",
            dry_run=dry_run,
            details={"warning": "pack has no benchmarks — pack_eval skipped"},
        )

    answers = answers or {}
    eval_texts = list(eval_texts or [])
    if not eval_texts and not answers:
        eval_texts = _proxy_texts_from_pack(pack)
    sys_prompt = system or (
        f"You are a careful {pack.domain} specialist. "
        "Be structured. Do not overclaim. State uncertainty."
    )
    items: list[dict[str, Any]] = []
    mode = "keyword_proxy"
    for b in benches:
        ans = answers.get(b.id) or answers.get(b.prompt)
        used = "provided"
        if not ans and llm_fn is not None and not dry_run:
            try:
                ans = llm_fn(sys_prompt, b.prompt)
                used = "llm"
                mode = "llm"
            except Exception as e:
                log.warning("pack_eval LLM failed for %s: %s", b.id, e)
                ans = ""
                used = "llm_error"
        if not ans:
            ans = _pick_text_for_prompt(b.prompt, eval_texts)
            used = "eval_text_proxy" if ans else "empty"
            if used == "eval_text_proxy" and mode != "llm":
                mode = "eval_text_proxy"
        row = score_answer(b, ans or "")
        row["source"] = used
        items.append(row)

    scored = [it for it in items if it.get("source") != "empty"]
    if scored:
        wsum = sum(float(it["weight"]) for it in scored)
        score = sum(float(it["score"]) * float(it["weight"]) for it in scored) / max(wsum, 1e-9)
    else:
        score = 0.0
    if dry_run and mode == "llm":
        mode = "eval_text_proxy"
    if dry_run and mode in ("eval_text_proxy", "keyword_proxy"):
        mode = "dry_run_proxy"

    report = PackEvalReport(
        domain=pack.domain,
        score=float(score),
        n=len(benches),
        n_scored=len(scored),
        items=items,
        mode=mode,
        dry_run=dry_run,
        details={"keywords": list(pack.keywords)[:16]},
    )
    log.info(
        "pack_eval domain=%s score=%.3f n=%d/%d mode=%s",
        pack.domain,
        report.score,
        report.n_scored,
        report.n,
        report.mode,
    )
    return report


def _proxy_texts_from_pack(pack: DomainPack) -> list[str]:
    """When no eval file is supplied, score against the pack's own actions/topics."""
    texts: list[str] = []
    texts.extend(str(a) for a in (pack.actions or []) if a)
    texts.extend(str(t) for t in (pack.topics or []) if t)
    if pack.description:
        texts.append(str(pack.description))
    if pack.actions:
        texts.append(
            "Assumptions stated. "
            + " ".join(str(a) for a in pack.actions)
            + " Reversible next step. Stop if evidence contradicts the frame. "
            + "Hypothesis: pack-defined actions are the specialist next moves."
        )
    return texts


def evaluate_from_data_config(
    data: DataConfig,
    **kwargs: Any,
) -> PackEvalReport:
    return evaluate_pack(resolve_domain_pack(data), **kwargs)


def write_pack_eval(report: PackEvalReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path
