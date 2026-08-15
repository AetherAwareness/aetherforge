#!/usr/bin/env python3
"""AetherForge CLI — reliable MoE post-training entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from aetherforge import __version__
from aetherforge.utils.overrides import parse_overrides


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-c",
        "--config",
        action="append",
        default=[],
        help="YAML config path (repeatable; later overrides earlier)",
    )
    p.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Dotlist override e.g. data.domain=logistics training.max_steps=50",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Run without loading heavy models"
    )
    p.add_argument(
        "--recipe",
        default=None,
        help="Named preset: dryrun | broad-flash | wide-flash | flash-domain | a3b-logistics | qwen38-27b",
    )


def _load(args: argparse.Namespace):
    from aetherforge.utils.config import load_config

    try:
        overrides = parse_overrides(getattr(args, "override", None) or [])
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if getattr(args, "dry_run", False):
        overrides.setdefault("run", {})["dry_run"] = True
    paths = list(getattr(args, "config", None) or [])
    recipe_meta = None
    if getattr(args, "recipe", None):
        from aetherforge.ux.recipes import resolve_recipe

        try:
            recipe_meta = resolve_recipe(args.recipe)
        except (KeyError, FileNotFoundError) as e:
            raise SystemExit(str(e)) from e
        # recipe configs first; explicit -c overrides later
        paths = list(recipe_meta["config_paths"]) + paths
        if recipe_meta.get("default_dry_run") and not getattr(args, "dry_run", False):
            # only force dry-run for the smoke recipe when user didn't pass live intent
            pass
    if not paths:
        default = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
        if default.exists():
            paths = [str(default)]
    cfg = load_config(*paths, overrides=overrides)
    # stash recipe id for train summary
    if recipe_meta is not None:
        try:
            cfg.run.name = cfg.run.name or recipe_meta.get("id", cfg.run.name)
        except Exception:
            pass
        args._recipe_meta = recipe_meta  # type: ignore[attr-defined]
    return cfg


def _print_train_summary(cfg, pipe, result: dict[str, Any], args: argparse.Namespace) -> None:
    """Friendly end-of-run card (plus JSON for scripts) — aesthetic sector forge view."""
    esft = (result.get("stages") or {}).get("esft") or {}
    sector_mode = getattr(cfg.training, "sector_mode", "sequential")
    payload = {
        "run_id": result.get("run_id"),
        "root": str(pipe.root),
        "promoted": result.get("promoted"),
        "promote_blocked": result.get("promote_blocked"),
        "stages": list(result.get("stages", {}).keys()),
        "duration_sec": result.get("duration_sec"),
        "posture": getattr(cfg.training, "posture", "specialist"),
        "sector_mode": sector_mode,
        "domain": cfg.data.domain,
        "model": cfg.model.name,
        "dry_run": bool(cfg.run.dry_run or getattr(args, "dry_run", False)),
    }
    if esft.get("schema") == "aetherforge.sector_workflow.v1" or esft.get("mode") == "sequential":
        payload["sectors"] = {
            "n_trained": esft.get("n_trained"),
            "n_blocked": esft.get("n_blocked"),
            "n_skipped": esft.get("n_skipped"),
            "items": [
                {
                    "name": s.get("name"),
                    "status": s.get("status"),
                    "readiness": s.get("readiness_status"),
                    "n_experts": s.get("n_experts"),
                    "n_train": s.get("n_train"),
                }
                for s in (esft.get("sectors") or [])[:24]
            ],
        }
    recipe = getattr(args, "_recipe_meta", None)
    if recipe:
        payload["recipe"] = recipe.get("id")

    # Human card to stderr so JSON stays parseable on stdout if --json
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
        return

    root = Path(payload["root"])
    promo = payload.get("promoted")
    status = "PROMOTED ★" if promo else ("DRY-RUN complete" if payload["dry_run"] else "completed (not promoted)")
    lines = [
        "",
        "╔" + "═" * 54 + "╗",
        f"║  AetherForge · {status:<38} ║",
        "╠" + "═" * 54 + "╣",
        f"║  run:      {root.name:<42} ║",
        f"║  path:     {str(root)[:42]:<42} ║",
        f"║  model:    {str(payload.get('model') or '')[:42]:<42} ║",
        f"║  domain:   {str(payload.get('domain') or '')[:42]:<42} ║",
        f"║  posture:  {str(payload.get('posture') or '')[:42]:<42} ║",
        f"║  sectors:  {str(payload.get('sector_mode') or '')[:42]:<42} ║",
    ]
    if payload.get("recipe"):
        lines.append(f"║  recipe:   {str(payload['recipe'])[:42]:<42} ║")
    lines.append(f"║  stages:   {', '.join(payload.get('stages') or [])[:42]:<42} ║")
    sec = payload.get("duration_sec") or 0
    dur = f"  duration: {sec:.1f}s"
    lines.append("║" + f"{dur:<54}" + "║")

    # Sector forge wave card
    sec_info = payload.get("sectors")
    if sec_info:
        lines.append("╠" + "═" * 54 + "╣")
        lines.append(
            f"║  SECTOR FORGE  trained={sec_info.get('n_trained')}  "
            f"blocked={sec_info.get('n_blocked')}  "
            f"skipped={sec_info.get('n_skipped')}"
        )
        # pad box line roughly
        lines[-1] = (lines[-1] + " " * 56)[:55] + " ║"
        for it in sec_info.get("items") or []:
            mark = {
                "trained": "●",
                "dry_run": "○",
                "blocked": "✗",
                "skipped": "·",
                "error": "!",
            }.get(it.get("status") or "", "·")
            row = (
                f"║  {mark} {it.get('name') or '?':<14} "
                f"{(it.get('status') or ''):<9} "
                f"rdy={(it.get('readiness') or '—'):<5} "
                f"e={it.get('n_experts') or 0:<3} "
                f"n={it.get('n_train') or 0}"
            )
            lines.append((row + " " * 56)[:55] + " ║")

    lines.append("╠" + "═" * 54 + "╣")
    for name in (
        "sector_forensics.md",
        "sector_readiness.md",
        "sector_workflow/sector_workflow.md",
        "scorecard.json",
        "expert_groups.json",
        "live_status.json",
    ):
        p = root / name
        if p.exists():
            lines.append(f"║  · {name:<50} ║")
    lines += [
        "╠" + "═" * 54 + "╣",
        "║  Next:                                                ║",
        "║    aetherforge dashboard     # Sector Forge + Studio  ║",
        f"║    aetherforge forensics --plan {str(root / 'expert_groups.json')[:20]:<20}… ║",
    ]
    if payload["dry_run"]:
        lines.append("║    aetherforge connect vast --host HOST --port PORT   ║")
        lines.append("║    aetherforge remote launch --exec --recipe broad-flash ║")
    elif not promo:
        lines.append("║    # open dashboard → Force promote if intentional    ║")
    lines.append("╚" + "═" * 54 + "╝")
    print("\n".join(lines))
    if not getattr(args, "quiet", False):
        print(json.dumps(payload, indent=2, default=str))


def cmd_train(args: argparse.Namespace) -> int:
    from aetherforge.training.pipeline import TrainingPipeline

    cfg = _load(args)
    if getattr(args, "sector_mode", None):
        cfg.training.sector_mode = args.sector_mode
    stages = [s.strip() for s in args.stages.split(",") if s.strip()] if args.stages else None
    pipe = TrainingPipeline(cfg)
    result = pipe.run(stages=stages)
    _print_train_summary(cfg, pipe, result, args)
    # dry-run: success if pipeline completed; live: require promotion unless --allow-fail
    if cfg.run.dry_run or args.dry_run:
        return 0
    if getattr(args, "allow_fail", False):
        return 0
    if result.get("promoted"):
        return 0
    return 2


def cmd_status(args: argparse.Namespace) -> int:
    from aetherforge.ux.status import gather_status, format_status_text

    report = gather_status()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_status_text(report))
    return 0


def cmd_recipes(args: argparse.Namespace) -> int:
    from aetherforge.ux.recipes import list_recipes, recipe_help_text

    if args.json:
        print(json.dumps(list_recipes(), indent=2, default=str))
    else:
        print(recipe_help_text())
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    from aetherforge.ux.init_project import init_domain

    result = init_domain(
        args.domain,
        description=args.description or "",
        curated_path=args.curated,
        posture=args.posture,
        high_stakes=args.high_stakes,
        recipe=args.recipe or "broad-flash",
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"""
AetherForge init complete
  domain:  {result['domain']}
  pack:    {result['pack_path']}
  card:    {result['card_path']}
  posture: {result['posture']}

First dry-run:
  {result['dry_run_cmd']}

Then:
  aetherforge dashboard
""".strip()
        )
    return 0 if result.get("ok") else 1


def cmd_quickstart(args: argparse.Namespace) -> int:
    """doctor → optional init → dry-run recipe → print status."""
    print("── AetherForge quickstart ──\n")
    rc = cmd_doctor(argparse.Namespace())
    if rc != 0:
        print("\nDoctor reported missing critical deps — fix those first.", file=sys.stderr)
        return rc
    print()
    if args.domain:
        cmd_init(
            argparse.Namespace(
                domain=args.domain,
                description=args.description or "",
                curated=None,
                posture=args.posture or "broad",
                high_stakes=False,
                recipe=args.recipe or "dryrun",
                json=False,
            )
        )
        print()
    # dry-run
    recipe = args.recipe or "dryrun"
    print(f"Running dry-run recipe: {recipe}\n")
    train_args = argparse.Namespace(
        config=[],
        override=[],
        dry_run=True,
        recipe=recipe,
        stages=None,
        json=False,
        quiet=False,
        allow_fail=True,
    )
    if args.domain:
        pack = Path("configs/domains") / f"{args.domain.strip().lower().replace(' ', '_')}.yaml"
        if pack.exists():
            train_args.config = [str(pack)]
    trc = cmd_train(train_args)
    print()
    cmd_status(argparse.Namespace(json=False))
    return trc


def cmd_probe(args: argparse.Namespace) -> int:
    from aetherforge.affinity.probe import AffinityProbe
    from aetherforge.affinity.expert_selector import ExpertSelector
    from aetherforge.models.loaders import load_moe_model

    cfg = _load(args)
    if cfg.run.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "msg": "probe requires a loaded model; use train --dry-run for synthetic affinity",
                }
            )
        )
        return 0
    bundle = load_moe_model(cfg.model, cfg.training)
    if args.texts:
        texts = Path(args.texts).read_text(encoding="utf-8").splitlines()
    else:
        texts = [f"{cfg.data.domain} probe sample {i}" for i in range(32)]
    probe = AffinityProbe(bundle, cfg.affinity, domain=cfg.data.domain)
    result = probe.run(texts)
    plan = ExpertSelector(cfg.affinity).select(result, bundle.experts)
    out = Path(args.output or "artifacts/affinity_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"affinity": result.to_dict(), "plan": plan.to_dict()}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def cmd_scorecard(args: argparse.Namespace) -> int:
    from aetherforge.eval.scorecard import ReliabilityScorecard
    from aetherforge.affinity.probe import AffinityResult

    cfg = _load(args)
    scorer = ReliabilityScorecard(cfg.eval, domain=cfg.data.domain)
    affinity = None
    if args.affinity:
        affinity = AffinityResult.from_dict(json.loads(Path(args.affinity).read_text()))
    eval_texts = []
    if args.eval:
        eval_texts = [
            json.loads(l).get("text", l) if l.strip().startswith("{") else l
            for l in Path(args.eval).read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
    pack_eval_payload = None
    try:
        from aetherforge.data.domain_pack import resolve_domain_pack
        from aetherforge.eval.pack_eval import evaluate_pack

        pe = evaluate_pack(
            resolve_domain_pack(cfg.data),
            eval_texts=eval_texts,
            dry_run=args.dry_run,
        )
        pack_eval_payload = pe.to_dict()
    except Exception:
        pack_eval_payload = None
    sc = scorer.evaluate(
        affinity=affinity,
        eval_texts=eval_texts,
        dry_run=args.dry_run,
        pack_eval=pack_eval_payload,
    )
    print(json.dumps(sc.to_dict(), indent=2))
    return 0 if sc.passed else 2


def cmd_eval(args: argparse.Namespace) -> int:
    """Pack-driven eval harness (benchmarks on the DomainPack)."""
    from aetherforge.data.domain_pack import resolve_domain_pack
    from aetherforge.eval.pack_eval import evaluate_pack, write_pack_eval
    from aetherforge.utils.llm_client import LLMConfig, OpenAICompatClient, make_llm_fn

    cfg = _load(args)
    pack = resolve_domain_pack(cfg.data)
    eval_texts: list[str] = []
    if args.eval:
        raw = Path(args.eval).read_text(encoding="utf-8")
        for line in raw.splitlines():
            if not line.strip():
                continue
            if line.strip().startswith("{"):
                try:
                    eval_texts.append(json.loads(line).get("text") or line)
                    continue
                except json.JSONDecodeError:
                    pass
            eval_texts.append(line)
    llm_fn = None
    if args.llm and not args.dry_run:
        if args.llm_base:
            llm_fn = make_llm_fn(
                LLMConfig(base_url=args.llm_base, model=args.llm_model or "local")
            )
        else:
            client = OpenAICompatClient()
            if client.available():
                llm_fn = make_llm_fn()
    report = evaluate_pack(
        pack,
        eval_texts=eval_texts,
        llm_fn=llm_fn,
        dry_run=bool(args.dry_run or cfg.run.dry_run),
    )
    if args.out:
        write_pack_eval(report, args.out)
    print(json.dumps(report.to_dict(), indent=2, default=str))
    if report.n == 0:
        return 0
    thr = getattr(cfg.eval.scorecard_thresholds, "pack_eval_min", None)
    if thr is not None and report.score < float(thr):
        return 2
    return 0


def cmd_thd(args: argparse.Namespace) -> int:
    """Generate Trajectory Hive Distillation pairs (stub or live OpenAI-compat)."""
    from aetherforge.data.domain_pack import resolve_domain_pack
    from aetherforge.data.trajectory_hive import TrajectoryHive
    from aetherforge.training.preference import PreferenceAligner
    from aetherforge.utils.llm_client import LLMConfig, OpenAICompatClient, make_llm_fn

    cfg = _load(args)
    pack = resolve_domain_pack(cfg.data)
    if args.prompts:
        problems = [
            ln.strip()
            for ln in Path(args.prompts).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    else:
        problems = list(pack.topics)[:8] or [
            f"Hard case in {pack.domain}: state assumptions and a stop rule."
        ]
    llm_fn = None
    live = bool(args.live) and not args.dry_run
    if live:
        if args.llm_base:
            llm_fn = make_llm_fn(
                LLMConfig(base_url=args.llm_base, model=args.llm_model or "local")
            )
        else:
            client = OpenAICompatClient()
            if not client.available():
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "error": "no OpenAI-compat LLM (set AETHERFORGE_LLM_BASE or --llm-base)",
                        }
                    )
                )
                return 2
            llm_fn = make_llm_fn()
    hive = TrajectoryHive(specialists=pack.specialists, seed=cfg.data.seed, llm_fn=llm_fn)
    traj, pairs = hive.generate(problems, pack.domain, live=live)
    extra = []
    if live and llm_fn is not None:
        extra = PreferenceAligner().synthesize_live(
            problems, llm_fn=llm_fn, domain=pack.domain
        )
        pairs = list(pairs) + [
            {
                "prompt": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
                "source": p.source,
                "meta": p.meta,
            }
            for p in extra
        ]
    out = Path(args.out or "artifacts/thd/preference_pairs.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "ok": True,
                "live": live,
                "n_problems": len(problems),
                "n_trajectories": len(traj),
                "n_pairs": len(pairs),
                "n_live_extra": len(extra),
                "path": str(out),
                "domain": pack.domain,
            },
            indent=2,
        )
    )
    return 0


def cmd_package(args: argparse.Namespace) -> int:
    from aetherforge.packaging.aetherpackage import AetherPackageBuilder

    cfg = _load(args)
    builder = AetherPackageBuilder(cfg)
    pkg = builder.build(run_dir=args.run_dir)
    print(json.dumps(pkg.to_dict(), indent=2))
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    from aetherforge.data.forge import DataForge

    cfg = _load(args)
    forge = DataForge(cfg.data)
    out = args.output or "artifacts/data"
    bundle = forge.build(out)
    result: dict[str, Any] = {"bundle": bundle.to_dict()}

    # Optional: also carve per-sector shards from forensics + group plan
    if getattr(args, "sectors", False):
        from aetherforge.data.sector_datasets import SectorDatasetForge
        from aetherforge.groups.forensics import forensics_for_group
        from aetherforge.groups.studio import create_studio_plan
        from aetherforge.groups.store import load_group_plan

        gcfg = cfg.groups
        if gcfg.plan_path and Path(gcfg.plan_path).exists():
            plan = load_group_plan(gcfg.plan_path)
        else:
            plan = create_studio_plan(
                family=cfg.model.family if cfg.model.family != "auto" else "qwen_a3b",
                model_name=cfg.model.name,
                num_groups=gcfg.target_num_groups,
                strategy=gcfg.strategy,
                arch_experts=cfg.model.num_experts,
                arch_topk=cfg.model.num_experts_per_tok,
                total_params_b=gcfg.total_params_b,
                active_params_b=gcfg.active_params_b,
            )
        # Auto-bind unbound sectors from forensics before sharding
        if getattr(gcfg, "auto_bind_from_forensics", True):
            from aetherforge.groups.readiness import run_forensics_gate

            run_forensics_gate(
                plan,
                mode="skip",
                auto_bind=True,
                global_domain=cfg.data.domain,
            )
        forensics_by_id = {
            g.id: forensics_for_group(plan, g.id) for g in plan.enabled_train_groups()
        }
        sector_out = Path(out) / "sector_datasets"
        sforge = SectorDatasetForge(
            cfg.data,
            min_match=float(getattr(cfg.data, "sector_min_match", 0.18) or 0.18),
            shared_fraction=float(
                getattr(cfg.training, "sector_shared_fraction", 0.15) or 0.15
            ),
            min_samples=int(getattr(cfg.training, "sector_min_samples", 8) or 8),
        )
        ds = sforge.build(
            plan,
            bundle.train_records,
            output_dir=sector_out,
            forensics_by_id=forensics_by_id,
            eval_texts=bundle.eval_texts,
        )
        result["sector_datasets"] = ds.to_dict()
        print(
            f"Sector datasets → {sector_out} "
            f"({len(ds.shards)} shards, general={len(ds.general_pool)})",
            file=sys.stderr,
        )

    print(json.dumps(result if getattr(args, "sectors", False) else bundle.to_dict(), indent=2))
    return 0


def cmd_workflow(args: argparse.Namespace) -> int:
    """
    Sector workflow CLI: forensics gate → per-sector datasets → optional sequential ESFT.

    Sub-intents via flags:
      --plan-only   readiness + datasets (no ESFT)
      default       full sequential sector workflow (dry-run unless live model)
    """
    from aetherforge.data.forge import DataForge
    from aetherforge.groups.readiness import readiness_markdown, run_forensics_gate
    from aetherforge.groups.studio import create_studio_plan
    from aetherforge.groups.store import load_group_plan, save_group_plan
    from aetherforge.training.sector_workflow import SectorWorkflow

    cfg = _load(args)
    out = Path(args.output or f"artifacts/workflow/{cfg.data.domain}")
    out.mkdir(parents=True, exist_ok=True)

    # Data
    data_dir = out / "data"
    bundle = DataForge(cfg.data).build(data_dir)

    # Groups
    gcfg = cfg.groups
    if gcfg.plan_path and Path(gcfg.plan_path).exists():
        plan = load_group_plan(gcfg.plan_path)
    else:
        plan = create_studio_plan(
            family=cfg.model.family if cfg.model.family != "auto" else "qwen_a3b",
            model_name=cfg.model.name,
            num_groups=gcfg.target_num_groups,
            strategy=gcfg.strategy,
            arch_experts=cfg.model.num_experts,
            arch_topk=cfg.model.num_experts_per_tok,
            total_params_b=gcfg.total_params_b,
            active_params_b=gcfg.active_params_b,
        )
    # train_scope
    scope = getattr(gcfg, "train_scope", "selected") or "selected"
    if scope == "all":
        for g in plan.groups:
            g.enabled = True
            g.train = True
            g.freeze = False
    elif scope == "top_n":
        n = max(1, int(getattr(gcfg, "train_top_n", 4) or 4))
        for i, g in enumerate(plan.groups):
            g.train = i < n
            g.freeze = i >= n
            g.enabled = True

    save_group_plan(plan, out / "expert_groups.json")

    only_ids = None
    if getattr(args, "sector", None):
        only_ids = [s.strip() for s in args.sector.split(",") if s.strip()]

    if args.plan_only:
        gate = run_forensics_gate(
            plan,
            mode=getattr(gcfg, "forensics_gate_mode", "warn") or "warn",
            auto_bind=bool(getattr(gcfg, "auto_bind_from_forensics", True)),
            global_domain=cfg.data.domain,
        )
        save_group_plan(plan, out / "expert_groups.bound.json")
        with open(out / "sector_readiness.json", "w", encoding="utf-8") as f:
            json.dump(gate.to_dict(), f, indent=2, default=str)
        (out / "sector_readiness.md").write_text(
            readiness_markdown(gate), encoding="utf-8"
        )
        from aetherforge.data.sector_datasets import SectorDatasetForge
        from aetherforge.groups.forensics import forensics_for_group

        forensics_by_id = {
            s.group_id: s.forensics for s in gate.sectors if s.forensics
        }
        # fill any missing
        for g in plan.enabled_train_groups():
            if g.id not in forensics_by_id:
                forensics_by_id[g.id] = forensics_for_group(plan, g.id)
        ds = SectorDatasetForge(
            cfg.data,
            min_samples=int(getattr(cfg.training, "sector_min_samples", 8) or 8),
            shared_fraction=float(
                getattr(cfg.training, "sector_shared_fraction", 0.15) or 0.15
            ),
        ).build(
            plan,
            bundle.train_records,
            output_dir=out / "sector_datasets",
            forensics_by_id=forensics_by_id,
            eval_texts=bundle.eval_texts,
            groups=(
                [g for g in plan.enabled_train_groups() if g.id in set(only_ids)]
                if only_ids
                else None
            ),
        )
        payload = {
            "plan_only": True,
            "readiness": gate.to_dict(),
            "datasets": ds.to_dict(),
            "output": str(out),
        }
        print(json.dumps(payload, indent=2, default=str))
        print(f"\nWorkflow plan → {out}", file=sys.stderr)
        print(f"  readiness: {gate.overall}  shards: {len(ds.shards)}", file=sys.stderr)
        return 0 if gate.overall != "block" else 2

    # Full sequential workflow (dry unless live weights + not --dry-run)
    dry = bool(cfg.run.dry_run or args.dry_run)
    wf = SectorWorkflow(
        plan,
        cfg.training,
        cfg.data,
        bundle=None,  # live load is via full `train` pipeline
        affinity=None,
        dry_run=True if dry else True,  # CLI workflow is always dry without model load
        gate_mode=getattr(gcfg, "forensics_gate_mode", "warn") or "warn",
        auto_bind=bool(getattr(gcfg, "auto_bind_from_forensics", True)),
        min_samples=int(getattr(cfg.training, "sector_min_samples", 8) or 8),
        shared_fraction=float(
            getattr(cfg.training, "sector_shared_fraction", 0.15) or 0.15
        ),
        continue_on_block=bool(
            getattr(cfg.training, "sector_continue_on_block", True)
        ),
    )
    result = wf.run(
        bundle.train_records,
        out,
        eval_texts=bundle.eval_texts,
        only_group_ids=only_ids,
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    print(
        f"\nSector workflow → {out}  "
        f"trained={result.n_trained} blocked={result.n_blocked}",
        file=sys.stderr,
    )
    if result.n_blocked and not result.n_trained:
        return 2
    return 0


def cmd_consult(args: argparse.Namespace) -> int:
    from aetherforge.orchestrate.hive import HiveOrchestrator
    from aetherforge.utils.llm_client import LLMConfig, make_llm_fn, make_llm_fn_from_providers

    specs = (
        [s.strip() for s in args.specialists.split(",") if s.strip()]
        if args.specialists
        else ["specialist_a", "specialist_b", "generalist"]
    )
    llm = None
    if args.llm:
        if args.llm_base:
            llm = make_llm_fn(
                LLMConfig(base_url=args.llm_base, model=args.llm_model or "local")
            )
        else:
            # Use OpenRouter / connected provider when --llm without explicit base
            llm = make_llm_fn_from_providers()
    hive = HiveOrchestrator(
        specs, protocol=args.protocol, max_rounds=args.rounds, llm=llm
    )
    result = hive.consult(args.question)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate config merge + pydantic schema without running stages."""
    from aetherforge.utils.config import load_config
    from aetherforge.training.pipeline import resolve_stages

    try:
        cfg = _load(args)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}, indent=2))
        return 1
    stages = resolve_stages(
        args.stages.split(",") if args.stages else None,
        cfg.training.stages,
    )
    report = {
        "ok": True,
        "model": cfg.model.name,
        "family": cfg.model.family,
        "domain": cfg.data.domain,
        "dry_run": cfg.run.dry_run,
        "method": cfg.training.method,
        "stages_resolved": stages,
        "scorecard_thresholds": cfg.eval.scorecard_thresholds.model_dump(),
        "privacy_mode": cfg.data.privacy_mode,
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_validate_flash(args: argparse.Namespace) -> int:
    """
    Prove AetherForge can train DeepSeek-V4-Flash-0731:
    config, weight map, PEFT target_parameters, meta structure, grad masks.
    Does not download full weights.
    """
    from aetherforge.models.deepseek_v4 import (
        FLASH_0731_ID,
        validate_flash_training_stack,
    )

    model_id = args.model or FLASH_0731_ID
    report = validate_flash_training_stack(
        model_id,
        check_weights_index=not args.skip_weights,
        try_meta_init=not args.skip_meta,
        try_peft_attach=not args.skip_peft_smoke,
    )
    payload = report.to_dict()
    print(json.dumps(payload, indent=2, default=str))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {out}", file=sys.stderr)
    return 0 if report.ok else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    """Environment / dependency check."""
    report: dict[str, Any] = {"aetherforge": __version__, "checks": {}}
    checks = report["checks"]

    try:
        import torch

        checks["torch"] = {
            "ok": True,
            "version": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception as e:
        checks["torch"] = {"ok": False, "error": str(e)}

    for mod in (
        "transformers",
        "peft",
        "datasets",
        "omegaconf",
        "yaml",
        "pydantic",
        "numpy",
    ):
        try:
            m = __import__(mod if mod != "yaml" else "yaml")
            checks[mod] = {"ok": True, "version": getattr(m, "__version__", "?")}
        except Exception as e:
            checks[mod] = {"ok": False, "error": str(e)}

    try:
        import unsloth  # noqa: F401

        checks["unsloth"] = {"ok": True}
    except Exception:
        checks["unsloth"] = {
            "ok": False,
            "error": "optional — install for A3B speed path",
        }

    # Flash-0731 PEFT path (target_parameters)
    try:
        from peft import LoraConfig

        LoraConfig(
            r=4,
            target_parameters=["gate_up_proj", "down_proj"],
            target_modules=[],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        )
        checks["flash_peft_target_parameters"] = {"ok": True}
    except Exception as e:
        checks["flash_peft_target_parameters"] = {
            "ok": False,
            "error": str(e),
            "hint": "peft>=0.15 required for DeepSeek-V4-Flash fused expert LoRA",
        }

    try:
        import deepspeed  # noqa: F401

        checks["deepspeed"] = {"ok": True}
    except Exception:
        checks["deepspeed"] = {
            "ok": False,
            "error": "optional — multi-node Flash scale",
        }

    # Local OpenAI-compatible endpoint (optional)
    try:
        from aetherforge.utils.llm_client import OpenAICompatClient

        client = OpenAICompatClient()
        checks["llm_endpoint"] = {
            "ok": client.available(),
            "base_url": client.config.base_url,
            "model": client.config.model,
        }
    except Exception as e:
        checks["llm_endpoint"] = {"ok": False, "error": str(e)}

    try:
        from aetherforge.providers import connect as pconn

        st = pconn.status()
        checks["providers"] = {
            "ok": True,
            "compute": (st.get("compute") or {}).get("ok"),
            "llm": (st.get("llm") or {}).get("ok"),
            "compute_msg": (st.get("compute") or {}).get("message"),
            "llm_msg": (st.get("llm") or {}).get("message"),
        }
    except Exception as e:
        checks["providers"] = {"ok": False, "error": str(e)}

    # Local Flash checkpoint + actionable next steps
    try:
        from aetherforge.ux.status import gather_status

        snap = gather_status()
        checks["flash_local"] = snap.get("flash_local") or {"ok": False}
        report["next_steps"] = snap.get("next_steps") or []
        report["dashboard"] = snap.get("dashboard")
    except Exception as e:
        checks["flash_local"] = {"ok": False, "error": str(e)}
        report["next_steps"] = [
            "aetherforge train --recipe dryrun --dry-run",
            "aetherforge dashboard",
            "aetherforge connect vast --host HOST --port PORT",
        ]

    if getattr(args, "human", False):
        print(f"AetherForge doctor v{__version__}")
        for k, v in checks.items():
            if not isinstance(v, dict):
                continue
            mark = "✓" if v.get("ok") else "·"
            extra = v.get("version") or v.get("message") or v.get("error") or ""
            if k == "flash_local" and v.get("path"):
                extra = f"{v.get('shards')} shards @ {v.get('path')}"
            elif k == "providers":
                extra = (
                    f"compute={v.get('compute')} llm={v.get('llm')} "
                    f"{v.get('compute_msg') or ''}"
                )
            print(f"  {mark} {k}: {extra}")
        print("\nNext steps:")
        for i, s in enumerate(report.get("next_steps") or [], 1):
            print(f"  {i}. {s}")
    else:
        print(json.dumps(report, indent=2, default=str))

    critical_ok = all(
        checks.get(k, {}).get("ok")
        for k in ("torch", "transformers", "pydantic", "numpy")
    )
    return 0 if critical_ok else 1


def cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def cmd_connect(args: argparse.Namespace) -> int:
    """Connect Vast.ai / RunPod / SSH compute or OpenRouter-style LLM APIs."""
    from aetherforge.providers import connect as conn
    from aetherforge.providers.credentials import set_secret, mask_secret, get_secret
    from aetherforge.providers.remote_train import build_remote_bundle

    action = args.connect_action
    if action == "list":
        print(json.dumps(conn.catalog(), indent=2, default=str))
        return 0
    if action == "status":
        print(json.dumps(conn.status(), indent=2, default=str))
        return 0
    if action == "key":
        # aetherforge connect key openrouter [--value KEY]  or from env
        provider = args.provider
        if args.value:
            print(json.dumps(conn.save_api_key(provider, args.value), indent=2))
            return 0
        # read from stdin-safe: env only if --from-env
        if args.from_env:
            env_map = {
                "vast": ["VAST_API_KEY", "VASTAI_API_KEY"],
                "runpod": ["RUNPOD_API_KEY"],
                "openrouter": ["OPENROUTER_API_KEY"],
                "openai": ["OPENAI_API_KEY"],
                "together": ["TOGETHER_API_KEY"],
                "fireworks": ["FIREWORKS_API_KEY"],
                "groq": ["GROQ_API_KEY"],
                "deepseek": ["DEEPSEEK_API_KEY"],
            }
            import os

            val = None
            for e in env_map.get(provider, []):
                if os.environ.get(e):
                    val = os.environ[e]
                    break
            if not val:
                print(json.dumps({"ok": False, "error": f"No env key found for {provider}"}))
                return 1
            print(json.dumps(conn.save_api_key(provider, val), indent=2))
            return 0
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Pass --value KEY or --from-env",
                    "hint": f"export KEY=… then: aetherforge connect key {provider} --from-env",
                },
                indent=2,
            )
        )
        return 1
    if action in ("vast", "runpod", "ssh"):
        result = conn.connect_compute(
            action,
            name=args.name,
            host=args.host or "",
            port=args.port,
            user=args.user,
            identity_file=args.identity,
            remote_dir=args.remote_dir,
            instance_id=args.instance_id,
            pod_id=args.pod_id,
            test=not args.no_test,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") or args.no_test else 2
    if action in (
        "openrouter",
        "openai",
        "together",
        "fireworks",
        "groq",
        "deepseek",
        "custom",
    ):
        if args.value:
            conn.save_api_key(action, args.value)
        result = conn.connect_llm(
            action,
            name=args.name,
            model=args.model,
            base_url=args.base_url,
            api_key=None,  # already saved if --value
            test=not args.no_test,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    print(json.dumps({"ok": False, "error": f"Unknown connect action: {action}"}))
    return 1


def _remote_config_paths(args: argparse.Namespace) -> list[str]:
    paths = list(getattr(args, "config", None) or [])
    if getattr(args, "recipe", None):
        from aetherforge.ux.recipes import resolve_recipe

        meta = resolve_recipe(args.recipe)
        paths = list(meta["config_paths"]) + paths
    if not paths:
        paths = ["configs/base.yaml"]
    return paths


def cmd_remote(args: argparse.Namespace) -> int:
    """Sync / launch / pull / logs on connected Vast/RunPod/SSH instance."""
    from aetherforge.providers.remote_train import (
        build_remote_bundle,
        exec_sync,
        exec_remote_train,
        exec_pull_artifacts,
        tail_remote_logs,
    )

    action = args.remote_action
    if action == "plan":
        bundle = build_remote_bundle(
            config_paths=_remote_config_paths(args),
            overrides=args.override or [],
            stages=args.stages,
            dry_run=args.dry_run,
        )
        print(json.dumps(bundle, indent=2, default=str))
        return 0 if bundle.get("ok") else 1
    if action == "sync":
        print(json.dumps(exec_sync(), indent=2, default=str))
        return 0
    if action == "launch":
        background = not getattr(args, "foreground", False)
        cfg_paths = _remote_config_paths(args)
        if not args.exec:
            bundle = build_remote_bundle(
                config_paths=cfg_paths,
                overrides=args.override or [],
                stages=args.stages,
                dry_run=args.dry_run,
                background=background,
            )
            print(json.dumps(bundle, indent=2, default=str))
            print("\n# Dry plan only. Re-run with --exec to SSH-launch (will use GPU $).")
            return 0 if bundle.get("ok") else 1
        result = exec_remote_train(
            cfg_paths,
            overrides=args.override or [],
            stages=args.stages,
            dry_run=args.dry_run,
            background=background,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 2
    if action == "pull":
        result = exec_pull_artifacts(local_dir=args.dest)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 2
    if action == "logs":
        result = tail_remote_logs(n=args.tail, run_glob=args.run_glob or "")
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 2
    return 1


def cmd_groups(args: argparse.Namespace) -> int:
    """Create / inspect / repartition expert groups (CLI companion to Studio UI)."""
    from aetherforge.groups.studio import create_studio_plan, lattice_view, analyze_group
    from aetherforge.groups.store import load_group_plan, save_group_plan

    if args.preview or not args.plan:
        plan = create_studio_plan(
            family=args.family,
            model_name=args.model or "",
            num_groups=args.num_groups,
            strategy=args.strategy,
            total_params_b=args.total_params_b,
            active_params_b=args.active_params_b,
        )
        if getattr(args, "forensics", False):
            from aetherforge.groups.forensics import (
                run_model_forensics,
                forensics_markdown,
                apply_forensics_to_plan,
            )

            if args.label:
                apply_forensics_to_plan(plan)
            report = run_model_forensics(plan, apply_labels=bool(args.label))
            if args.out:
                save_group_plan(plan, args.out)
            if args.markdown:
                print(forensics_markdown(report))
            else:
                print(json.dumps(report.to_dict(), indent=2, default=str))
            return 0
        if args.out:
            save_group_plan(plan, args.out)
        print(
            json.dumps(
                {
                    "summary": plan.summary(),
                    "capacity": plan.capacity.to_dict(),
                    "hint": (
                        f"~{plan.capacity.active_params_b}B active / "
                        f"{plan.capacity.total_params_b}B total · "
                        f"up to ~{plan.capacity.max_disjoint_active_groups} "
                        f"disjoint active-scale sectors"
                    ),
                },
                indent=2,
            )
        )
        return 0

    plan = load_group_plan(args.plan)
    if getattr(args, "forensics", False):
        from aetherforge.groups.forensics import (
            run_model_forensics,
            forensics_markdown,
            apply_forensics_to_plan,
        )

        affinity = None
        if args.affinity:
            affinity = json.loads(Path(args.affinity).read_text(encoding="utf-8"))
        if args.label:
            apply_forensics_to_plan(plan, affinity=affinity)
            if args.out:
                save_group_plan(plan, args.out)
            else:
                save_group_plan(plan, args.plan)
        report = run_model_forensics(plan, affinity=affinity)
        if args.markdown:
            print(forensics_markdown(report))
        else:
            print(json.dumps(report.to_dict(), indent=2, default=str))
        return 0
    if args.analyze:
        print(json.dumps(analyze_group(plan, args.analyze), indent=2, default=str))
        return 0
    print(json.dumps(plan.summary(), indent=2, default=str))
    return 0


def cmd_forensics(args: argparse.Namespace) -> int:
    """
    Sector forensics — inventory what each MoE sector contains so you can edit
    efficiently (params vs active fire, themes, layer roles, edit recs).
    """
    from aetherforge.groups.studio import create_studio_plan
    from aetherforge.groups.store import load_group_plan, save_group_plan
    from aetherforge.groups.forensics import (
        run_model_forensics,
        forensics_markdown,
        forensics_for_group,
        apply_forensics_to_plan,
        probe_texts_from_theme_bank,
    )

    affinity = None
    if args.affinity:
        affinity = json.loads(Path(args.affinity).read_text(encoding="utf-8"))

    if args.plan:
        plan = load_group_plan(args.plan)
    else:
        plan = create_studio_plan(
            family=args.family,
            model_name=args.model or "",
            num_groups=args.num_groups,
            strategy=args.strategy,
            total_params_b=args.total_params_b,
            active_params_b=args.active_params_b,
        )

    if args.probes:
        probes = probe_texts_from_theme_bank()
        out = Path(args.probes)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(json.dumps(p) for p in probes) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"ok": True, "n_probes": len(probes), "path": str(out)}, indent=2))
        return 0

    if args.sector:
        dossier = forensics_for_group(plan, args.sector, affinity=affinity)
        print(json.dumps(dossier, indent=2, default=str))
        return 0 if not dossier.get("error") else 1

    if args.label:
        apply_forensics_to_plan(plan, affinity=affinity)
        dest = args.out or args.plan
        if dest:
            save_group_plan(plan, dest)

    report = run_model_forensics(plan, affinity=affinity, apply_labels=False)

    if args.out_report:
        rp = Path(args.out_report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        if str(rp).endswith(".md"):
            rp.write_text(forensics_markdown(report), encoding="utf-8")
        else:
            rp.write_text(
                json.dumps(report.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
        print(f"Wrote {rp}", file=sys.stderr)

    if args.markdown:
        print(forensics_markdown(report))
    else:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch the visual Training Console (pipeline, scorecard, affinity, controls)."""
    from aetherforge.viz.server import serve
    from aetherforge.viz.run_store import default_runs_root

    root = args.runs_root or str(default_runs_root())
    print(f"AetherForge Training Console v{__version__}")
    print(f"  URL:  http://{args.host}:{args.port}/")
    print(f"  Runs: {root}")
    print("  Controls: Approve · Reject · Force promote")
    serve(host=args.host, port=args.port, runs_root=root)
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    """List training runs (JSON) for scripts / TUI."""
    from aetherforge.viz.run_store import list_runs, default_runs_root

    root = args.runs_root or str(default_runs_root())
    runs = list_runs(root, limit=args.limit)
    print(json.dumps({"runs_root": root, "runs": runs}, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    main_parser = argparse.ArgumentParser(
        prog="aetherforge",
        description="AetherForge — reliable post-training for open sparse MoE models",
    )
    main_parser.add_argument(
        "--version", action="version", version=f"aetherforge {__version__}"
    )
    sub = main_parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Run full or partial training pipeline")
    _add_config_args(t)
    t.add_argument(
        "--stages",
        default=None,
        help="Comma stages: diagnostics,data,affinity,groups,esft,router_hygiene,"
        "preference,lifecycle,scorecard,package  "
        "(esft uses training.sector_mode=sequential|joint)",
    )
    t.add_argument(
        "--sector-mode",
        choices=["sequential", "joint"],
        default=None,
        help="Override training.sector_mode: sequential (per-sector forensics+data+ESFT) "
        "or joint (one ESFT over all selected experts)",
    )
    t.add_argument(
        "--json",
        action="store_true",
        help="JSON-only summary (default prints human card + JSON)",
    )
    t.add_argument(
        "--quiet",
        action="store_true",
        help="Human card only (no trailing JSON)",
    )
    t.add_argument(
        "--allow-fail",
        action="store_true",
        help="Exit 0 even if scorecard did not promote",
    )
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("probe", help="Run affinity probe only")
    _add_config_args(pr)
    pr.add_argument("--texts", help="Path to probe texts (one per line)")
    pr.add_argument("--output", help="Output JSON path")
    pr.set_defaults(func=cmd_probe)

    sc = sub.add_parser("scorecard", help="Evaluate reliability scorecard")
    _add_config_args(sc)
    sc.add_argument("--affinity", help="affinity.json path")
    sc.add_argument("--eval", help="eval jsonl/text path")
    sc.set_defaults(func=cmd_scorecard)

    ev = sub.add_parser(
        "eval",
        help="Pack-driven eval harness (DomainPack.benchmarks → pack_eval.json)",
    )
    _add_config_args(ev)
    ev.add_argument("--eval", help="eval jsonl/text path used as answer proxies")
    ev.add_argument("--out", help="Write pack_eval.json path")
    ev.add_argument(
        "--llm",
        action="store_true",
        help="Generate answers via OpenAI-compat (skipped on --dry-run)",
    )
    ev.add_argument("--llm-base", default=None)
    ev.add_argument("--llm-model", default=None)
    ev.set_defaults(func=cmd_eval)

    th = sub.add_parser(
        "thd",
        help="Trajectory Hive Distillation pairs (stub, or --live OpenAI-compat)",
    )
    _add_config_args(th)
    th.add_argument("--prompts", help="Prompts file, one problem per line")
    th.add_argument("--out", help="preference_pairs.jsonl path")
    th.add_argument(
        "--live",
        action="store_true",
        help="Call OpenAI-compat LLM (never on --dry-run)",
    )
    th.add_argument("--llm-base", default=None)
    th.add_argument("--llm-model", default=None)
    th.set_defaults(func=cmd_thd)

    pk = sub.add_parser("package", help="Build AetherPackage from a run directory")
    _add_config_args(pk)
    pk.add_argument("--run-dir", required=True, help="Pipeline run directory")
    pk.set_defaults(func=cmd_package)

    d = sub.add_parser("data", help="Build DataForge corpus only")
    _add_config_args(d)
    d.add_argument("--output", help="Output directory")
    d.add_argument(
        "--sectors",
        action="store_true",
        help="Also partition corpus into per-sector shards (forensics-aware)",
    )
    d.set_defaults(func=cmd_data)

    wf = sub.add_parser(
        "workflow",
        help="Sector workflow: forensic readiness → datasets → sequential ESFT plan",
    )
    _add_config_args(wf)
    wf.add_argument("--output", help="Workflow output directory")
    wf.add_argument(
        "--plan-only",
        action="store_true",
        help="Only forensics gate + sector datasets (no ESFT dry-run cards)",
    )
    wf.add_argument(
        "--sector",
        default=None,
        help="Comma group ids to include (default: all train-enabled)",
    )
    wf.set_defaults(func=cmd_workflow)

    c = sub.add_parser(
        "consult", help="Multi-specialist hive consult (echo or live LLM)"
    )
    c.add_argument("question", help="Question for the hive")
    c.add_argument(
        "--specialists",
        default="specialist_a,specialist_b,generalist",
        help="Comma specialist names for this industry hive",
    )
    c.add_argument(
        "--protocol",
        default="debate",
        choices=["debate", "router", "round_robin"],
    )
    c.add_argument("--rounds", type=int, default=2)
    c.add_argument(
        "--llm",
        action="store_true",
        help="Use connected LLM (OpenRouter/…) or --llm-base",
    )
    c.add_argument(
        "--llm-base",
        default=None,
        help="OpenAI-compatible base URL (default: active provider or local)",
    )
    c.add_argument("--llm-model", default=None)
    c.set_defaults(func=cmd_consult)

    v = sub.add_parser("validate", help="Validate config without training")
    _add_config_args(v)
    v.add_argument("--stages", default=None)
    v.set_defaults(func=cmd_validate)

    vf = sub.add_parser(
        "validate-flash",
        help="Prove DeepSeek-V4-Flash-0731 trainability (config/PEFT/weight map)",
    )
    vf.add_argument(
        "--model",
        default=None,
        help="HF id (default: deepseek-ai/DeepSeek-V4-Flash-0731)",
    )
    vf.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip huggingface weight-index fetch",
    )
    vf.add_argument(
        "--skip-meta",
        action="store_true",
        help="Skip from_config structure probe",
    )
    vf.add_argument(
        "--skip-peft-smoke",
        action="store_true",
        help="Skip tiny PEFT attach + grad-mask smoke",
    )
    vf.add_argument("--out", help="Write report JSON path")
    vf.set_defaults(func=cmd_validate_flash)

    doc = sub.add_parser("doctor", help="Check environment and dependencies")
    doc.add_argument(
        "--human",
        action="store_true",
        help="Print checkmarks + next steps instead of raw JSON",
    )
    doc.set_defaults(func=cmd_doctor)

    ver = sub.add_parser("version", help="Print version")
    ver.set_defaults(func=cmd_version)

    st = sub.add_parser(
        "status",
        help="Human-readable status: dashboard, Vast, Flash weights, next steps",
    )
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_status)

    rc = sub.add_parser("recipes", help="List named train recipes (--recipe presets)")
    rc.add_argument("--json", action="store_true")
    rc.set_defaults(func=cmd_recipes)

    ini = sub.add_parser(
        "init",
        help="Scaffold a domain pack + first-run command card",
    )
    ini.add_argument("domain", help="Domain slug e.g. logistics or aether_public")
    ini.add_argument("--description", default="")
    ini.add_argument("--curated", help="Path to curated jsonl")
    ini.add_argument(
        "--posture",
        default="broad",
        choices=["specialist", "broad", "wide"],
    )
    ini.add_argument(
        "--recipe",
        default="broad-flash",
        help="Named recipe for the generated card",
    )
    ini.add_argument("--high-stakes", action="store_true")
    ini.add_argument("--json", action="store_true")
    ini.set_defaults(func=cmd_init)

    qs = sub.add_parser(
        "quickstart",
        help="Doctor + dry-run recipe (+ optional domain init) in one shot",
    )
    qs.add_argument("--domain", default=None, help="Optional domain to scaffold")
    qs.add_argument("--description", default="")
    qs.add_argument(
        "--recipe",
        default="dryrun",
        help="Recipe for dry-run (default: dryrun)",
    )
    qs.add_argument(
        "--posture",
        default="broad",
        choices=["specialist", "broad", "wide"],
    )
    qs.set_defaults(func=cmd_quickstart)

    # ── connect (Vast / OpenRouter / …) ─────────────────────────────
    cn = sub.add_parser(
        "connect",
        help="Connect Vast.ai / RunPod / SSH compute or OpenRouter-style LLM APIs",
    )
    cn_sub = cn.add_subparsers(dest="connect_action", required=True)

    cn_list = cn_sub.add_parser("list", help="Catalog of providers + saved profiles")
    cn_list.set_defaults(func=cmd_connect)

    cn_st = cn_sub.add_parser("status", help="Health-check active compute + LLM")
    cn_st.set_defaults(func=cmd_connect)

    cn_key = cn_sub.add_parser("key", help="Save API key for a provider")
    cn_key.add_argument(
        "provider",
        help="vast|runpod|openrouter|openai|together|fireworks|groq|deepseek",
    )
    cn_key.add_argument("--value", help="API key value (prefer --from-env)")
    cn_key.add_argument(
        "--from-env",
        action="store_true",
        help="Read key from standard env var for that provider",
    )
    cn_key.set_defaults(func=cmd_connect)

    def _add_compute_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--name", help="Profile name (default: provider id)")
        p.add_argument("--host", default="", help="SSH host from Vast/RunPod dashboard")
        p.add_argument("--port", type=int, default=22)
        p.add_argument("--user", default="root")
        p.add_argument("--identity", help="SSH private key path")
        p.add_argument("--remote-dir", default="/workspace/aetherforge")
        p.add_argument("--instance-id", help="Vast instance id (optional)")
        p.add_argument("--pod-id", help="RunPod pod id (optional)")
        p.add_argument("--no-test", action="store_true")
        p.set_defaults(func=cmd_connect)

    for prov in ("vast", "runpod", "ssh"):
        cp = cn_sub.add_parser(prov, help=f"Connect {prov} compute instance")
        _add_compute_args(cp)

    def _add_llm_args(lp: argparse.ArgumentParser) -> None:
        lp.add_argument("--name", help="Profile name")
        lp.add_argument("--model", help="Default model id")
        lp.add_argument("--base-url", help="Override API base URL")
        lp.add_argument("--value", help="API key (or use: connect key … --from-env)")
        lp.add_argument("--no-test", action="store_true")
        lp.set_defaults(func=cmd_connect)

    for prov in (
        "openrouter",
        "openai",
        "together",
        "fireworks",
        "groq",
        "deepseek",
        "custom",
    ):
        lp = cn_sub.add_parser(prov, help=f"Connect {prov} LLM API")
        _add_llm_args(lp)

    # ── remote train ────────────────────────────────────────────────
    rm = sub.add_parser(
        "remote", help="Sync/launch/pull/logs on connected GPU box"
    )
    rm_sub = rm.add_subparsers(dest="remote_action", required=True)
    for act, help_ in (
        ("plan", "Show rsync + remote train commands (no side effects)"),
        ("sync", "Rsync project to remote"),
        ("launch", "Plan or --exec remote train"),
        ("pull", "Pull remote artifacts (+ log tails) to artifacts/remote/"),
        ("logs", "Tail newest remote run log + live_status"),
    ):
        rp = rm_sub.add_parser(act, help=help_)
        if act in ("plan", "launch"):
            _add_config_args(rp)
            rp.add_argument("--stages", default=None)
        if act == "launch":
            rp.add_argument(
                "--exec",
                action="store_true",
                help="Actually SSH and run (uses remote GPU — costs money)",
            )
            rp.add_argument(
                "--foreground",
                action="store_true",
                help="Block on remote train (default: nohup background on the box)",
            )
        if act == "pull":
            rp.add_argument(
                "--dest",
                default=None,
                help="Local destination (default: artifacts/remote)",
            )
        if act == "logs":
            rp.add_argument("--tail", type=int, default=80)
            rp.add_argument(
                "--run-glob",
                default="",
                help="Optional run dir glob e.g. flagship-logistics-a3b-*",
            )
        rp.set_defaults(func=cmd_remote)

    g = sub.add_parser(
        "groups",
        help="Expert Group Studio CLI — carve MoE into train-able sectors",
    )
    g.add_argument(
        "--family",
        default="deepseek_v4_flash",
        choices=["deepseek_v4_flash", "qwen_a3b", "qwen38_dense", "generic_moe"],
    )
    g.add_argument("--model", default="")
    g.add_argument("--num-groups", type=int, default=8, help="How many sectors to call up")
    g.add_argument(
        "--strategy",
        default="active_slots",
        choices=["active_slots", "affinity", "round_robin", "layer_bands"],
    )
    g.add_argument("--total-params-b", type=float, default=None)
    g.add_argument("--active-params-b", type=float, default=None)
    g.add_argument("--preview", action="store_true", help="Preview without a plan file")
    g.add_argument("--plan", help="Load expert_groups.json")
    g.add_argument("--analyze", help="Group id to deep-analyze")
    g.add_argument(
        "--forensics",
        action="store_true",
        help="Sector forensics inventory (what each sector contains)",
    )
    g.add_argument("--affinity", help="affinity.json path for routing-aware forensics")
    g.add_argument(
        "--label",
        action="store_true",
        help="Write forensic theme labels onto group description/tags",
    )
    g.add_argument(
        "--markdown",
        action="store_true",
        help="Print forensics as Markdown inventory",
    )
    g.add_argument("--out", help="Write plan JSON")
    g.set_defaults(func=cmd_groups)

    fo = sub.add_parser(
        "forensics",
        help="MoE sector forensics — inventory what each expert sector contains",
    )
    fo.add_argument(
        "--family",
        default="deepseek_v4_flash",
        choices=["deepseek_v4_flash", "qwen_a3b", "qwen38_dense", "generic_moe"],
    )
    fo.add_argument("--model", default="")
    fo.add_argument("--num-groups", type=int, default=12)
    fo.add_argument(
        "--strategy",
        default="active_slots",
        choices=["active_slots", "affinity", "round_robin", "layer_bands"],
    )
    fo.add_argument("--total-params-b", type=float, default=None)
    fo.add_argument("--active-params-b", type=float, default=None)
    fo.add_argument("--plan", help="Existing expert_groups.json")
    fo.add_argument("--affinity", help="affinity.json for routing-aware content scores")
    fo.add_argument("--sector", help="Single group id dossier")
    fo.add_argument(
        "--label",
        action="store_true",
        help="Stamp theme/role labels onto the plan groups",
    )
    fo.add_argument("--out", help="Write labeled plan JSON")
    fo.add_argument(
        "--out-report",
        help="Write full report (.json or .md by extension)",
    )
    fo.add_argument("--markdown", action="store_true", help="Print Markdown inventory")
    fo.add_argument(
        "--probes",
        help="Write theme-probe JSONL path (for live AffinityProbe multi-pass)",
    )
    fo.set_defaults(func=cmd_forensics)

    dash = sub.add_parser(
        "dashboard",
        aliases=["viz", "ui"],
        help="Open visual Training Console (stages, scorecard, affinity, promote)",
    )
    dash.add_argument("--host", default="127.0.0.1")
    dash.add_argument("--port", type=int, default=8765)
    dash.add_argument(
        "--runs-root",
        default=None,
        help="Directory of training runs (default: artifacts/runs)",
    )
    dash.set_defaults(func=cmd_dashboard)

    runs = sub.add_parser("runs", help="List training runs as JSON")
    runs.add_argument("--runs-root", default=None)
    runs.add_argument("--limit", type=int, default=30)
    runs.set_defaults(func=cmd_runs)

    return main_parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
