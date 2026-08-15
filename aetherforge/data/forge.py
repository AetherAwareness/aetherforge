"""DataForge — Stage 1 orchestration: curate, synthesize, gate, version."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aetherforge.data.domain_pack import DomainPack, pack_to_dict, resolve_domain_pack
from aetherforge.data.quality_gates import GateReport, QualityGateRunner
from aetherforge.data.synthetic import generate_self_instruct
from aetherforge.data.trajectory_hive import TrajectoryHive
from aetherforge.utils.audit import AuditLog
from aetherforge.utils.config import DataConfig
from aetherforge.utils.hashing import dataset_fingerprint
from aetherforge.utils.logging import get_logger

log = get_logger("data.forge")


@dataclass
class DataBundle:
    domain: str
    train_records: list[dict[str, Any]]
    eval_texts: list[str]
    probe_texts: list[str]
    preference_pairs: list[dict[str, str]] = field(default_factory=list)
    fingerprint: str = ""
    quality: Optional[dict[str, Any]] = None
    paths: dict[str, str] = field(default_factory=dict)
    domain_pack: Optional[dict[str, Any]] = None
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "n_train": len(self.train_records),
            "n_eval": len(self.eval_texts),
            "n_probe": len(self.probe_texts),
            "n_preference_pairs": len(self.preference_pairs),
            "fingerprint": self.fingerprint,
            "quality": self.quality,
            "paths": self.paths,
            "keywords": self.keywords[:48],
            "domain_pack": {
                k: self.domain_pack.get(k)
                for k in ("domain", "description", "high_stakes")
                if self.domain_pack
            }
            if self.domain_pack
            else None,
        }


class DataForge:
    def __init__(
        self,
        config: DataConfig,
        audit: Optional[AuditLog] = None,
        llm_fn: Optional[Any] = None,
        live_thd: bool = False,
    ):
        self.config = config
        self.audit = audit
        self.gates = QualityGateRunner(config.quality_gates)
        self.llm_fn = llm_fn
        self.live_thd = live_thd

    def build(self, output_dir: str | Path) -> DataBundle:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config

        pack = resolve_domain_pack(cfg)
        with open(output_dir / "domain_pack.resolved.json", "w", encoding="utf-8") as f:
            json.dump(pack_to_dict(pack), f, indent=2)

        curated = self._load_curated(cfg.curated_path)
        # Broad work: mix additional corpora
        mixed = self._load_mix_paths(
            list(cfg.mix_paths or []),
            max_per=cfg.mix_max_per_source,
        )
        if mixed:
            log.info("DataForge mix_paths: +%d records from %d sources", len(mixed), len(cfg.mix_paths or []))
            for rec in mixed:
                if "domain" not in rec and cfg.domains:
                    # round-robin light tag if multi-domain list provided
                    rec.setdefault("meta", {})
                curated.append(rec)

        synthetic: list[dict[str, Any]] = []
        if cfg.synthetic.enabled:
            # Broad: generate for primary domain + optional extra domain labels
            domain_list = [pack.domain] + [
                d for d in (cfg.domains or []) if d and d != pack.domain
            ]
            per = max(1, cfg.synthetic.num_samples // max(len(domain_list), 1))
            from copy import deepcopy
            from aetherforge.utils.config import DataConfig as _DC

            for di, dom in enumerate(domain_list):
                sc = deepcopy(cfg.synthetic)
                sc.num_samples = per if di < len(domain_list) - 1 else (
                    cfg.synthetic.num_samples - per * (len(domain_list) - 1)
                )
                sc.num_samples = max(1, sc.num_samples)
                dom_cfg = (
                    cfg
                    if dom == pack.domain
                    else _DC(domain=dom, seed=cfg.seed + di)
                )
                synthetic.extend(
                    generate_self_instruct(
                        dom,
                        sc,
                        seed=cfg.seed + di,
                        pack=pack if dom == pack.domain else None,
                        data_cfg=dom_cfg,
                    )
                )

        thd_traj: list[dict[str, Any]] = []
        pairs: list[dict[str, str]] = []
        if cfg.synthetic.trajectory_hive:
            seeds = [
                r.get("prompt") or r.get("text", "")[:200]
                for r in (curated + synthetic)[:64]
            ]
            seeds = [s for s in seeds if s] or [
                f"Hard case {i} in {pack.domain}" for i in range(16)
            ]
            hive = TrajectoryHive(
                specialists=pack.specialists,
                seed=cfg.seed,
                llm_fn=self.llm_fn,
            )
            thd_traj, pairs = hive.generate(
                seeds,
                pack.domain,
                live=bool(self.live_thd and self.llm_fn is not None),
            )

        merged = curated + synthetic + thd_traj
        filtered, report = self.gates.filter_records(merged)

        if cfg.max_train_samples:
            filtered = filtered[: cfg.max_train_samples]

        # curriculum: shorter / simpler first if enabled
        if cfg.curriculum:
            filtered = sorted(filtered, key=lambda r: len(str(r.get("text", ""))))

        # splits
        n = len(filtered)
        n_eval = max(1, min(64, n // 10)) if n else 0
        eval_recs = filtered[:n_eval]
        train_recs = filtered[n_eval:] if n > n_eval else filtered
        probe = [str(r.get("text", ""))[:1000] for r in train_recs[:256]]
        # load external probe if provided
        if cfg.probe_path and Path(cfg.probe_path).exists():
            probe = self._load_texts(cfg.probe_path)[:2048]
        eval_texts = [str(r.get("text", "")) for r in eval_recs]
        if cfg.eval_path and Path(cfg.eval_path).exists():
            eval_texts = self._load_texts(cfg.eval_path)

        fp = dataset_fingerprint(train_recs)
        paths = self._write(output_dir, train_recs, eval_texts, probe, pairs, report, fp)
        paths["domain_pack"] = str(output_dir / "domain_pack.resolved.json")

        bundle = DataBundle(
            domain=pack.domain,
            train_records=train_recs,
            eval_texts=eval_texts,
            probe_texts=probe,
            preference_pairs=pairs,
            fingerprint=fp,
            quality=report.to_dict(),
            paths=paths,
            domain_pack=pack_to_dict(pack),
            keywords=list(pack.keywords),
        )
        with open(output_dir / "data_bundle.json", "w", encoding="utf-8") as f:
            json.dump(bundle.to_dict(), f, indent=2)

        if self.audit:
            self.audit.record(
                "data",
                "forge_complete",
                bundle.to_dict(),
                data_hash=fp,
            )
        log.info("DataForge complete: %s", bundle.to_dict())
        return bundle

    def _load_mix_paths(
        self,
        paths: list[str],
        max_per: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in paths:
            recs = self._load_curated(path)
            if max_per is not None and max_per > 0:
                recs = recs[:max_per]
            for r in recs:
                r = dict(r)
                r.setdefault("source_path", path)
                out.append(r)
        return out

    def _load_curated(self, path: Optional[str]) -> list[dict[str, Any]]:
        if not path:
            return []
        p = Path(path)
        if not p.exists():
            log.warning("curated_path not found: %s", path)
            return []
        records: list[dict[str, Any]] = []
        if p.suffix == ".jsonl":
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif p.suffix == ".json":
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "records" in data:
                records = data["records"]
        else:
            # plain text — one example per paragraph
            text = p.read_text(encoding="utf-8")
            for block in text.split("\n\n"):
                if block.strip():
                    records.append({"text": block.strip(), "source": "curated_txt"})
        # Normalize chat / instruction formats → text for SFT
        normed: list[dict[str, Any]] = []
        for r in records:
            if not isinstance(r, dict):
                normed.append({"text": str(r)})
                continue
            if "text" in r and r["text"]:
                normed.append(r)
                continue
            if "messages" in r and isinstance(r["messages"], list):
                parts = []
                for m in r["messages"]:
                    if not isinstance(m, dict):
                        continue
                    role = m.get("role", "user")
                    parts.append(f"{role}: {m.get('content', '')}")
                normed.append({**r, "text": "\n".join(parts)})
                continue
            if "instruction" in r or "output" in r:
                instr = str(r.get("instruction") or r.get("input") or "")
                out = str(r.get("output") or r.get("response") or "")
                normed.append({**r, "text": f"{instr}\n\n{out}".strip()})
                continue
            normed.append(r)
        records = normed
        log.info("Loaded %d curated records from %s", len(records), path)
        return records

    def _load_texts(self, path: str) -> list[str]:
        p = Path(path)
        if p.suffix == ".jsonl":
            out = []
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        out.append(str(rec.get("text") or rec))
            return out
        return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def _write(
        self,
        output_dir: Path,
        train: list[dict[str, Any]],
        eval_texts: list[str],
        probe: list[str],
        pairs: list[dict[str, str]],
        report: GateReport,
        fp: str,
    ) -> dict[str, str]:
        paths = {}
        train_path = output_dir / "train.jsonl"
        with open(train_path, "w", encoding="utf-8") as f:
            for r in train:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths["train"] = str(train_path)

        eval_path = output_dir / "eval.jsonl"
        with open(eval_path, "w", encoding="utf-8") as f:
            for t in eval_texts:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        paths["eval"] = str(eval_path)

        probe_path = output_dir / "probe.jsonl"
        with open(probe_path, "w", encoding="utf-8") as f:
            for t in probe:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        paths["probe"] = str(probe_path)

        pairs_path = output_dir / "preference_pairs.jsonl"
        with open(pairs_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        paths["preference_pairs"] = str(pairs_path)

        with open(output_dir / "quality_report.json", "w", encoding="utf-8") as f:
            json.dump(report.to_dict() | {"fingerprint": fp}, f, indent=2)
        paths["quality"] = str(output_dir / "quality_report.json")
        return paths
