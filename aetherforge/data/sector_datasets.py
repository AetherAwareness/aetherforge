"""
Per-sector dataset forge — bind corpora to MoE expert sectors.

Given a global DataBundle (or raw records) + GroupPlan + forensic dossiers,
produce a train/eval shard for each train-enabled sector so sequential ESFT
receives data that matches what that sector *contains* / is assigned.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aetherforge.data.domain_pack import DomainPack, resolve_domain_pack
from aetherforge.data.quality_gates import QualityGateRunner
from aetherforge.data.synthetic import generate_self_instruct
from aetherforge.groups.forensics import DEFAULT_THEME_BANK
from aetherforge.groups.models import ExpertGroup, GroupPlan
from aetherforge.utils.config import DataConfig, QualityGatesConfig, SyntheticConfig
from aetherforge.utils.hashing import dataset_fingerprint
from aetherforge.utils.logging import get_logger

log = get_logger("data.sector_datasets")


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+]{3,}", (text or "").lower()))


def _text_of(rec: dict[str, Any]) -> str:
    if not isinstance(rec, dict):
        return str(rec)
    if rec.get("text"):
        return str(rec["text"])
    if rec.get("prompt"):
        return str(rec.get("prompt", "")) + "\n" + str(rec.get("response", ""))
    if rec.get("messages"):
        parts = []
        for m in rec["messages"]:
            if isinstance(m, dict):
                parts.append(str(m.get("content", "")))
        return "\n".join(parts)
    if rec.get("instruction") or rec.get("output"):
        return f"{rec.get('instruction', '')}\n{rec.get('output', '')}"
    return json.dumps(rec, ensure_ascii=False)[:2000]


def sector_match_score(
    rec: dict[str, Any],
    group: ExpertGroup,
    *,
    forensics: Optional[dict[str, Any]] = None,
    theme_bank: Optional[dict[str, dict[str, Any]]] = None,
) -> float:
    """
    Soft score 0..1 for how well a record belongs to this sector.

    Evidence: domain tags, keywords, topics, forensic themes, explicit meta.sector_id.
    """
    bank = theme_bank or DEFAULT_THEME_BANK
    text = _text_of(rec)
    toks = _token_set(text)
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    score = 0.0

    # Explicit assignment wins
    if meta.get("sector_id") == group.id or rec.get("sector_id") == group.id:
        return 1.0
    if meta.get("group_id") == group.id:
        return 1.0
    if group.domain and (
        str(rec.get("domain", "")).lower() == str(group.domain).lower()
        or str(meta.get("domain", "")).lower() == str(group.domain).lower()
    ):
        score = max(score, 0.85)

    # Keyword / topic bag
    bag = list(group.keywords or []) + list(group.topics or [])
    if group.domain:
        bag.append(group.domain)
    if bag and toks:
        hits = 0
        for kw in bag:
            parts = str(kw).lower().split()
            if all(p in toks or p in text.lower() for p in parts):
                hits += 1
        score = max(score, min(1.0, hits / max(len(bag) * 0.35, 1.0)))

    # Forensic themes
    content = (forensics or {}).get("content") or {}
    themes = content.get("top_themes") or []
    for t in themes[:4]:
        tid = t.get("id")
        tscore = float(t.get("score") or 0)
        if tscore < 0.1 or not tid:
            continue
        theme = bank.get(tid, {})
        kws = theme.get("keywords") or []
        if not kws:
            continue
        th = 0
        for kw in kws:
            parts = str(kw).lower().split()
            if all(p in toks or p in text.lower() for p in parts):
                th += 1
        theme_match = min(1.0, th / max(len(kws) * 0.35, 1.0))
        score = max(score, theme_match * min(1.0, tscore + 0.2))

    return float(min(1.0, score))


@dataclass
class SectorDataShard:
    group_id: str
    name: str
    domain: Optional[str]
    train_records: list[dict[str, Any]] = field(default_factory=list)
    eval_texts: list[str] = field(default_factory=list)
    fingerprint: str = ""
    match_stats: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    forensics_summary: str = ""
    quality: Optional[dict[str, Any]] = None
    contract: Optional[dict[str, Any]] = None
    train_eligible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "domain": self.domain,
            "n_train": len(self.train_records),
            "n_eval": len(self.eval_texts),
            "fingerprint": self.fingerprint,
            "match_stats": self.match_stats,
            "paths": self.paths,
            "forensics_summary": self.forensics_summary,
            "quality": self.quality,
            "contract": self.contract,
            "train_eligible": self.train_eligible,
        }


@dataclass
class SectorDatasetPlan:
    """All sector shards + residual general pool."""

    domain: str
    shards: list[SectorDataShard]
    general_pool: list[dict[str, Any]] = field(default_factory=list)
    unassigned: list[dict[str, Any]] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)
    method: str = "soft_assign+synthesize"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aetherforge.sector_datasets.v1",
            "domain": self.domain,
            "method": self.method,
            "n_shards": len(self.shards),
            "n_general_pool": len(self.general_pool),
            "n_unassigned": len(self.unassigned),
            "paths": self.paths,
            "shards": [s.to_dict() for s in self.shards],
        }

    def shard_for(self, group_id: str) -> Optional[SectorDataShard]:
        for s in self.shards:
            if s.group_id == group_id:
                return s
        return None

    def sample_counts(self) -> dict[str, int]:
        return {s.group_id: len(s.train_records) for s in self.shards}


class SectorDatasetForge:
    """
    Build per-sector datasets from a global corpus + forensics.

    Pipeline:
      1. Score every train record against each train-enabled sector
      2. Assign to best sector above threshold (else general pool)
      3. Optionally inject shared_fraction of general pool into each shard
      4. Synthesize more samples for thin sectors using sector pack signature
      5. Quality-gate each shard
      6. Write sector_datasets/<id>/{train,eval}.jsonl + plan.json
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        *,
        min_match: float = 0.18,
        shared_fraction: float = 0.15,
        min_samples: int = 8,
        synthesize_fill: bool = True,
        max_synth_per_sector: int = 48,
        contract_mode: str = "block",
        min_real_fraction: float = 0.15,
        max_synth_fraction: float = 0.85,
        min_unique_ratio: float = 0.35,
    ):
        self.data_cfg = data_cfg
        self.min_match = min_match
        self.shared_fraction = max(0.0, min(1.0, shared_fraction))
        self.min_samples = min_samples
        self.synthesize_fill = synthesize_fill
        self.max_synth_per_sector = max_synth_per_sector
        self.contract_mode = contract_mode
        self.min_real_fraction = min_real_fraction
        self.max_synth_fraction = max_synth_fraction
        self.min_unique_ratio = min_unique_ratio
        self.gates = QualityGateRunner(data_cfg.quality_gates or QualityGatesConfig())

    def build(
        self,
        plan: GroupPlan,
        train_records: list[dict[str, Any]],
        *,
        output_dir: str | Path,
        forensics_by_id: Optional[dict[str, dict[str, Any]]] = None,
        eval_texts: Optional[list[str]] = None,
        groups: Optional[list[ExpertGroup]] = None,
    ) -> SectorDatasetPlan:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        forensics_by_id = forensics_by_id or {}
        targets = groups or plan.enabled_train_groups()
        eval_texts = list(eval_texts or [])

        # Score matrix
        assignments: dict[str, list[tuple[float, dict[str, Any]]]] = {
            g.id: [] for g in targets
        }
        general: list[dict[str, Any]] = []
        unassigned: list[dict[str, Any]] = []

        for rec in train_records:
            best_id = None
            best_score = 0.0
            scores_row: dict[str, float] = {}
            for g in targets:
                sc = sector_match_score(
                    rec, g, forensics=forensics_by_id.get(g.id)
                )
                scores_row[g.id] = sc
                if sc > best_score:
                    best_score = sc
                    best_id = g.id
            tagged = dict(rec)
            meta = dict(tagged.get("meta") or {})
            meta["sector_scores"] = {k: round(v, 4) for k, v in scores_row.items()}
            tagged["meta"] = meta

            if best_id is not None and best_score >= self.min_match:
                meta["assigned_sector"] = best_id
                meta["match_score"] = round(best_score, 4)
                assignments[best_id].append((best_score, tagged))
            else:
                unassigned.append(tagged)
                general.append(tagged)

        # Shared general mix
        n_share = int(len(general) * self.shared_fraction) if general else 0
        shared = general[: max(n_share, 0)]

        shards: list[SectorDataShard] = []
        for g in targets:
            scored = sorted(assignments[g.id], key=lambda x: x[0], reverse=True)
            records = [r for _, r in scored]

            # inject shared
            for srec in shared:
                if srec not in records:
                    copy = dict(srec)
                    meta = dict(copy.get("meta") or {})
                    meta["shared_mix"] = True
                    copy["meta"] = meta
                    records.append(copy)

            # sector-specific curated path
            if g.curated_path and Path(g.curated_path).exists():
                extra = self._load_path(g.curated_path)
                for er in extra:
                    er = dict(er)
                    er.setdefault("meta", {})
                    if isinstance(er["meta"], dict):
                        er["meta"]["source"] = "sector_curated"
                    records.append(er)

            # synthesize fill from forensic / pack signature
            synth_n = 0
            if self.synthesize_fill and len(records) < self.min_samples:
                need = min(
                    self.max_synth_per_sector,
                    max(self.min_samples - len(records), self.min_samples // 2),
                )
                synth = self._synthesize_for_sector(
                    g, forensics_by_id.get(g.id), n=need
                )
                records.extend(synth)
                synth_n = len(synth)

            # quality gate
            filtered, qreport = self.gates.filter_records(records)
            if not filtered and records:
                filtered = records  # never empty a non-empty sector solely on gates

            # curriculum: shorter first
            if self.data_cfg.curriculum:
                filtered = sorted(filtered, key=lambda r: len(_text_of(r)))

            # Sector eval: held-out slice from sector records OR matched global eval
            # (never the full global pool — keep sector eval separate)
            n_eval = max(1, min(16, len(filtered) // 10)) if len(filtered) >= 10 else 0
            if n_eval:
                sector_eval = [_text_of(r) for r in filtered[:n_eval]]
                train_body = filtered[n_eval:]
            else:
                sector_eval = []
                train_body = list(filtered)
            # matched global eval texts (sector-specific), not full global eval dump
            if eval_texts:
                matched_eval = []
                for t in eval_texts:
                    fake = {"text": t}
                    if sector_match_score(
                        fake, g, forensics=forensics_by_id.get(g.id)
                    ) >= self.min_match:
                        matched_eval.append(t)
                if matched_eval:
                    # prefer matched global as eval when local holdout tiny
                    if len(sector_eval) < 4:
                        sector_eval = (sector_eval + matched_eval)[:32]
                    else:
                        sector_eval = (sector_eval + matched_eval[:8])[:32]

            fp = dataset_fingerprint(train_body) if train_body else "empty"

            # Data contract
            from aetherforge.data.contracts import DataContractSpec, evaluate_sector_contract

            contract = evaluate_sector_contract(
                train_body,
                group_id=g.id,
                name=g.name,
                spec=DataContractSpec(
                    min_samples=self.min_samples,
                    min_real_fraction=self.min_real_fraction,
                    max_synth_fraction=self.max_synth_fraction,
                    min_unique_ratio=self.min_unique_ratio,
                    mode=self.contract_mode,  # type: ignore[arg-type]
                ),
            )

            sector_dir = output_dir / g.id
            sector_dir.mkdir(parents=True, exist_ok=True)
            train_path = sector_dir / "train.jsonl"
            eval_path = sector_dir / "eval.jsonl"
            with open(train_path, "w", encoding="utf-8") as f:
                for r in train_body:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(eval_path, "w", encoding="utf-8") as f:
                for t in sector_eval:
                    f.write(json.dumps({"text": t, "sector_eval": True}, ensure_ascii=False) + "\n")
            contract_path = sector_dir / "data_contract.json"
            with open(contract_path, "w", encoding="utf-8") as f:
                json.dump(contract.to_dict(), f, indent=2)

            freport = forensics_by_id.get(g.id) or {}
            summary = (freport.get("content") or {}).get("summary") or ""
            match_stats = {
                "n_matched": len(scored),
                "n_shared_injected": len(shared),
                "n_synthesized": synth_n,
                "n_after_gates": len(filtered),
                "mean_match": (
                    sum(s for s, _ in scored) / len(scored) if scored else 0.0
                ),
                "min_match_threshold": self.min_match,
                "n_sector_eval": len(sector_eval),
            }

            meta_path = sector_dir / "shard_meta.json"
            shard = SectorDataShard(
                group_id=g.id,
                name=g.name,
                domain=g.domain,
                train_records=train_body,
                eval_texts=sector_eval,
                fingerprint=fp,
                match_stats=match_stats,
                paths={
                    "train": str(train_path),
                    "eval": str(eval_path),
                    "meta": str(meta_path),
                    "contract": str(contract_path),
                    "dir": str(sector_dir),
                },
                forensics_summary=summary,
                quality=qreport.to_dict() if qreport else None,
                contract=contract.to_dict(),
                train_eligible=contract.train_eligible,
            )
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(shard.to_dict(), f, indent=2, default=str)
            shards.append(shard)
            log.info(
                "Sector dataset %s (%s): %d train (matched=%d synth=%d)",
                g.name,
                g.id,
                len(train_body),
                len(scored),
                synth_n,
            )

        # write general / unassigned
        gen_path = output_dir / "general_pool.jsonl"
        with open(gen_path, "w", encoding="utf-8") as f:
            for r in general:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        una_path = output_dir / "unassigned.jsonl"
        with open(una_path, "w", encoding="utf-8") as f:
            for r in unassigned:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        plan_out = SectorDatasetPlan(
            domain=self.data_cfg.domain,
            shards=shards,
            general_pool=general,
            unassigned=unassigned,
            paths={
                "root": str(output_dir),
                "general_pool": str(gen_path),
                "unassigned": str(una_path),
            },
        )
        with open(output_dir / "sector_dataset_plan.json", "w", encoding="utf-8") as f:
            json.dump(plan_out.to_dict(), f, indent=2, default=str)

        # human summary
        md = self._markdown(plan_out)
        (output_dir / "sector_datasets.md").write_text(md, encoding="utf-8")
        plan_out.paths["plan"] = str(output_dir / "sector_dataset_plan.json")
        plan_out.paths["markdown"] = str(output_dir / "sector_datasets.md")
        return plan_out

    def _synthesize_for_sector(
        self,
        group: ExpertGroup,
        forensics: Optional[dict[str, Any]],
        *,
        n: int,
    ) -> list[dict[str, Any]]:
        content = (forensics or {}).get("content") or {}
        topics = list(group.topics or content.get("assigned_topics") or [])
        keywords = list(group.keywords or content.get("assigned_keywords") or [])
        themes = content.get("top_themes") or []
        bank = DEFAULT_THEME_BANK
        for t in themes[:2]:
            theme = bank.get(t.get("id") or "", {})
            topics.append(str(theme.get("label") or t.get("label") or ""))
            keywords.extend(list(theme.get("keywords") or [])[:4])
        topics = [t for t in topics if t]
        if not topics:
            topics = [f"specialist work for sector {group.name}"]

        domain = group.domain or self.data_cfg.domain or "sector"
        # Load pack if sector has one
        pack: Optional[DomainPack] = None
        if group.domain_pack:
            try:
                from aetherforge.data.domain_pack import load_domain_pack

                pack = load_domain_pack(group.domain_pack)
            except Exception as e:
                log.debug("sector domain_pack load failed: %s", e)

        if pack is None:
            pack = resolve_domain_pack(
                DataConfig(
                    domain=domain,
                    topics=topics,
                    keywords=keywords,
                    description=group.description or f"Sector {group.name}",
                    seed=self.data_cfg.seed,
                )
            )
            # ensure topics from forensics stick
            if topics:
                pack.topics = list(dict.fromkeys(list(pack.topics) + topics))[:24]
            if keywords:
                pack.keywords = list(dict.fromkeys(list(pack.keywords) + keywords))[:48]

        synth_cfg = SyntheticConfig(
            enabled=True,
            num_samples=max(1, n),
            trajectory_hive=False,
        )
        records = generate_self_instruct(
            domain,
            synth_cfg,
            seed=self.data_cfg.seed + hash(group.id) % 10000,
            topics=topics,
            pack=pack,
        )
        out = []
        for r in records:
            r = dict(r)
            r.setdefault("meta", {})
            if isinstance(r["meta"], dict):
                r["meta"]["sector_id"] = group.id
                r["meta"]["source"] = "sector_synth"
                r["meta"]["sector_name"] = group.name
            r["sector_id"] = group.id
            r["domain"] = domain
            out.append(r)
        return out

    def _load_path(self, path: str) -> list[dict[str, Any]]:
        p = Path(path)
        records: list[dict[str, Any]] = []
        if p.suffix == ".jsonl":
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        elif p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = data.get("records") or data.get("train") or []
        else:
            for block in p.read_text(encoding="utf-8").split("\n\n"):
                if block.strip():
                    records.append({"text": block.strip(), "source": "sector_txt"})
        return records

    def _markdown(self, plan: SectorDatasetPlan) -> str:
        lines = [
            f"# Sector datasets — `{plan.domain}`",
            "",
            f"Method: `{plan.method}` · Shards: **{len(plan.shards)}** · "
            f"General pool: **{len(plan.general_pool)}** · "
            f"Unassigned: **{len(plan.unassigned)}**",
            "",
            "| Sector | Domain | Train | Matched | Synth | Mean match |",
            "|--------|--------|-------|---------|-------|------------|",
        ]
        for s in plan.shards:
            ms = s.match_stats or {}
            lines.append(
                f"| {s.name} | {s.domain or '—'} | {len(s.train_records)} | "
                f"{ms.get('n_matched', 0)} | {ms.get('n_synthesized', 0)} | "
                f"{float(ms.get('mean_match') or 0):.2f} |"
            )
        lines += ["", "## Forensics-linked summaries", ""]
        for s in plan.shards:
            lines.append(f"### {s.name}")
            if s.forensics_summary:
                lines.append(s.forensics_summary)
            else:
                lines.append("_No forensics summary attached._")
            lines.append("")
        lines.append("---")
        lines.append(
            "*Each shard is the fuel for one sector ESFT pass — "
            "matched to what that sector contains.*"
        )
        return "\n".join(lines)
