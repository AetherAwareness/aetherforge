"""Read/list training runs for the dashboard API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from aetherforge.utils.logging import get_logger

log = get_logger("viz.run_store")


def public_path(path: str | Path) -> Path:
    """Prefer ~/aetherforge when artifacts are bind-mounted / shared with hiveforge."""
    p = Path(path).resolve()
    s = str(p)
    if "/hiveforge/" in s:
        alt = Path(s.replace("/hiveforge/", "/aetherforge/", 1))
        try:
            if alt.exists() and alt.samefile(p):
                return alt
        except OSError:
            pass
    return p


def default_runs_root() -> Path:
    # Prefer repo-local artifacts, then cwd
    candidates = [
        Path(__file__).resolve().parents[2] / "artifacts" / "runs",
        Path.cwd() / "artifacts" / "runs",
        Path.home() / "aetherforge" / "artifacts" / "runs",
    ]
    for c in candidates:
        if c.exists():
            return public_path(c)
    return public_path(candidates[0])


def list_runs(runs_root: Optional[str | Path] = None, limit: int = 50) -> list[dict[str, Any]]:
    root = Path(runs_root) if runs_root else default_runs_root()
    if not root.exists():
        return []
    runs = []
    for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        summary = summarize_run(d)
        if summary:
            runs.append(summary)
        if len(runs) >= limit:
            break
    return runs


def summarize_run(run_dir: str | Path) -> Optional[dict[str, Any]]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    live = _read_json(run_dir / "live_status.json")
    result = _read_json(run_dir / "pipeline_result.json")
    scorecard = _read_json(run_dir / "scorecard.json")
    cfg = _read_yaml(run_dir / "config.resolved.yaml")

    domain = (
        (live or {}).get("domain")
        or (cfg or {}).get("data", {}).get("domain")
        or "unknown"
    )
    run_id = (
        (live or {}).get("run_id")
        or (result or {}).get("run_id")
        or run_dir.name
    )
    status = (live or {}).get("status") or (
        "completed" if result else "unknown"
    )
    promoted = (live or {}).get("promoted")
    if promoted is None and result is not None:
        promoted = result.get("promoted", False)

    mtime = run_dir.stat().st_mtime
    posture = (cfg or {}).get("training", {}).get("posture") or "specialist"
    model = (
        (live or {}).get("model")
        or (cfg or {}).get("model", {}).get("name")
        or ""
    )
    has_forensics = (run_dir / "sector_forensics.json").exists() or (
        run_dir / "sector_forensics.md"
    ).exists()
    has_sector_workflow = (run_dir / "sector_workflow" / "sector_workflow.json").exists()
    sector_mode = (
        (live or {}).get("sector_mode")
        or (cfg or {}).get("training", {}).get("sector_mode")
        or ("sequential" if has_sector_workflow else "joint")
    )
    live_sectors = (live or {}).get("sectors") or {}
    return {
        "run_id": run_id,
        "name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "domain": domain,
        "model": model,
        "posture": posture,
        "status": status,
        "promoted": bool(promoted),
        "promote_blocked": (live or {}).get("promote_blocked")
        or (result or {}).get("promote_blocked"),
        "percent": (live or {}).get("percent"),
        "current_stage": (live or {}).get("current_stage"),
        "scorecard_passed": (scorecard or {}).get("passed")
        if scorecard
        else None,
        "dry_run": (live or {}).get("dry_run")
        if live
        else (cfg or {}).get("run", {}).get("dry_run"),
        "has_forensics": has_forensics,
        "has_sector_workflow": has_sector_workflow,
        "sector_mode": sector_mode,
        "sectors_trained": live_sectors.get("n_trained"),
        "sectors_total": live_sectors.get("n_total"),
        "updated_at": (live or {}).get("updated_at") or mtime,
        "mtime": mtime,
        "has_live": live is not None,
        "has_result": result is not None,
    }


def load_run_bundle(run_dir: str | Path) -> dict[str, Any]:
    """Full visualization payload for one run."""
    run_dir = Path(run_dir)
    live = _read_json(run_dir / "live_status.json")
    result = _read_json(run_dir / "pipeline_result.json")
    scorecard = _read_json(run_dir / "scorecard.json")
    affinity = _read_json(run_dir / "affinity.json")
    selection = _read_json(run_dir / "selection_plan.json")
    data_bundle = _read_json(run_dir / "data" / "data_bundle.json")
    quality = _read_json(run_dir / "data" / "quality_report.json")
    domain_pack = _read_json(run_dir / "data" / "domain_pack.resolved.json")
    lifecycle = _read_json(run_dir / "lifecycle" / "lifecycle_plan.json")
    utilization = _read_json(run_dir / "lifecycle" / "utilization.json")
    esft = _read_json(run_dir / "checkpoints" / "esft" / "esft_result.json")
    router = _read_json(
        run_dir / "checkpoints" / "router_hygiene" / "router_hygiene_result.json"
    )
    preference = _read_json(
        run_dir / "checkpoints" / "preference" / "preference_result.json"
    )
    expert_groups = _read_json(run_dir / "expert_groups.json")
    controls = _read_json(run_dir / "operator_controls.json") or {}
    cfg = _read_yaml(run_dir / "config.resolved.yaml")
    audit_tail = _read_jsonl_tail(run_dir / "audit.jsonl", n=80)
    log_tail = _read_text_tail(run_dir / "aetherforge.log", n=120)
    # Sector forge artifacts (sequential workflow)
    sector_workflow = _read_json(run_dir / "sector_workflow" / "sector_workflow.json")
    sector_readiness = _read_json(run_dir / "sector_readiness.json") or _read_json(
        run_dir / "sector_workflow" / "sector_readiness.json"
    )
    sector_readiness_post = _read_json(
        run_dir / "sector_workflow" / "sector_readiness_post_data.json"
    )
    sector_datasets = _read_json(
        run_dir / "sector_workflow" / "sector_datasets" / "sector_dataset_plan.json"
    )
    sector_forensics = _read_json(run_dir / "sector_forensics.json")

    groups_view = None
    if expert_groups:
        try:
            from aetherforge.groups.models import GroupPlan
            from aetherforge.groups.studio import lattice_view

            gp = GroupPlan.model_validate(expert_groups)
            groups_view = {
                "summary": gp.summary(),
                "lattice": lattice_view(gp),
                "plan": expert_groups,
            }
        except Exception as e:
            log.debug("groups view: %s", e)
            groups_view = {"plan": expert_groups, "error": str(e)}

    # Compact affinity for heatmap (limit size)
    affinity_view = None
    if affinity and "affinity" in affinity:
        aff = affinity["affinity"]
        # may be nested list
        if isinstance(aff, list) and aff:
            # keep full if small, else downsample
            affinity_view = {
                "num_layers": affinity.get("num_layers"),
                "num_experts": affinity.get("num_experts"),
                "matrix": _downsample_matrix(aff, max_rows=24, max_cols=64),
                "load_balance_cv": affinity.get("load_balance_cv"),
                "entropy_per_layer": (affinity.get("entropy_per_layer") or [])[:48],
                "ranked_head": (affinity.get("ranked") or [])[:24],
                "probe_tokens": affinity.get("probe_tokens"),
            }

    scorecard_view = None
    if scorecard:
        scorecard_view = {
            "passed": scorecard.get("passed"),
            "domain": scorecard.get("domain"),
            "metrics": scorecard.get("metrics") or {},
            "gate": scorecard.get("gate") or {},
            "details": scorecard.get("details") or {},
        }

    selection_view = None
    if selection:
        selection_view = {
            "domain": selection.get("domain"),
            "selected_count": len(selection.get("selected") or []),
            "selected_head": (selection.get("selected") or [])[:32],
            "mitosis_candidates": (selection.get("mitosis_candidates") or [])[:16],
            "freeze_router": selection.get("freeze_router"),
            "metadata": selection.get("metadata") or {},
        }

    # Compact sector forge view for dashboard
    sector_forge = None
    if sector_workflow or (live or {}).get("sectors") or sector_readiness:
        live_sec = (live or {}).get("sectors") or {}
        wf_secs = (sector_workflow or {}).get("sectors") or []
        items = live_sec.get("items") or []
        if not items and wf_secs:
            items = [
                {
                    "group_id": s.get("group_id"),
                    "name": s.get("name"),
                    "state": s.get("status"),
                    "readiness": s.get("readiness_status"),
                    "n_experts": s.get("n_experts"),
                    "n_samples": s.get("n_train"),
                    "forensics_summary": s.get("forensics_summary"),
                    "duration_sec": s.get("duration_sec"),
                    "error": s.get("error"),
                }
                for s in wf_secs
            ]
        shards = (sector_datasets or {}).get("shards") or []
        readiness_rows = (sector_readiness_post or sector_readiness or {}).get(
            "sectors"
        ) or []
        sector_forge = {
            "mode": (
                (sector_workflow or {}).get("mode")
                or (cfg or {}).get("training", {}).get("sector_mode")
                or ("sequential" if sector_workflow or live_sec else "joint")
            ),
            "live": live_sec,
            "workflow": {
                "n_trained": (sector_workflow or {}).get("n_trained"),
                "n_skipped": (sector_workflow or {}).get("n_skipped"),
                "n_blocked": (sector_workflow or {}).get("n_blocked"),
                "duration_sec": (sector_workflow or {}).get("duration_sec"),
            }
            if sector_workflow
            else None,
            "items": items,
            "readiness_overall": (sector_readiness_post or sector_readiness or {}).get(
                "overall"
            ),
            "readiness_narrative": (sector_readiness_post or sector_readiness or {}).get(
                "narrative"
            ),
            "readiness": readiness_rows[:32],
            "datasets": {
                "n_shards": len(shards),
                "n_general_pool": (sector_datasets or {}).get("n_general_pool"),
                "shards": [
                    {
                        "group_id": s.get("group_id"),
                        "name": s.get("name"),
                        "domain": s.get("domain"),
                        "n_train": s.get("n_train"),
                        "match_stats": s.get("match_stats"),
                        "forensics_summary": (s.get("forensics_summary") or "")[:240],
                    }
                    for s in shards[:24]
                ],
            }
            if sector_datasets
            else None,
            "forensics_inventory": (sector_forensics or {}).get("inventory_table")
            if sector_forensics
            else None,
            "forensics_narrative": (sector_forensics or {}).get("narrative")
            if sector_forensics
            else None,
        }

    return {
        "summary": summarize_run(run_dir),
        "live": live,
        "result": result,
        "config": cfg,
        "scorecard": scorecard_view,
        "affinity": affinity_view,
        "selection": selection_view,
        "data": data_bundle,
        "quality": quality,
        "domain_pack": domain_pack,
        "lifecycle": lifecycle,
        "utilization": utilization,
        "esft": esft,
        "router_hygiene": router,
        "preference": preference,
        "expert_groups": groups_view,
        "sector_forge": sector_forge,
        "controls": controls,
        "audit_tail": audit_tail,
        "log_tail": log_tail,
        "artifacts": {
            "promoted": (run_dir / "promoted").is_dir(),
            "aetherpackage": (run_dir / "aetherpackage").is_dir(),
            "live_status": (run_dir / "live_status.json").exists(),
            "expert_groups": (run_dir / "expert_groups.json").exists(),
            "sector_workflow": (run_dir / "sector_workflow").is_dir(),
            "sector_forensics": (run_dir / "sector_forensics.json").exists(),
            "sector_readiness": (run_dir / "sector_readiness.json").exists()
            or (run_dir / "sector_workflow" / "sector_readiness.json").exists(),
        },
        "server_time": time.time(),
    }


def write_controls(run_dir: str | Path, patch: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run_dir)
    path = run_dir / "operator_controls.json"
    cur = _read_json(path) or {}
    cur.update(patch)
    cur["updated_at"] = time.time()
    path.write_text(json.dumps(cur, indent=2), encoding="utf-8")

    # Mirror into live_status if present
    live_path = run_dir / "live_status.json"
    live = _read_json(live_path)
    if live is not None:
        live.setdefault("controls", {}).update(cur)
        live["updated_at"] = time.time()
        live_path.write_text(json.dumps(live, indent=2, default=str), encoding="utf-8")
    return cur


def apply_human_approve(run_dir: str | Path, note: str = "") -> dict[str, Any]:
    return write_controls(
        run_dir,
        {
            "human_approved": True,
            "rejected": False,
            "notes": note,
            "action": "human_approve",
        },
    )


def apply_reject(run_dir: str | Path, note: str = "") -> dict[str, Any]:
    return write_controls(
        run_dir,
        {
            "human_approved": False,
            "rejected": True,
            "force_promote": False,
            "notes": note,
            "action": "reject",
        },
    )


def apply_force_promote_flag(run_dir: str | Path, note: str = "") -> dict[str, Any]:
    """Flag only — actual copy performed by controls.force_promote()."""
    return write_controls(
        run_dir,
        {
            "force_promote": True,
            "human_approved": True,
            "rejected": False,
            "notes": note,
            "action": "force_promote_requested",
        },
    )


def force_promote_package(run_dir: str | Path, note: str = "") -> dict[str, Any]:
    """Copy aetherpackage → promoted even if scorecard failed (operator override)."""
    import shutil

    run_dir = Path(run_dir)
    pkg = run_dir / "aetherpackage"
    if not pkg.is_dir():
        raise FileNotFoundError("No aetherpackage/ to promote — run package stage first")
    dest = run_dir / "promoted"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(pkg, dest)
    controls = apply_force_promote_flag(run_dir, note=note)

    # Update pipeline_result if present
    result_path = run_dir / "pipeline_result.json"
    result = _read_json(result_path) or {}
    result["promoted"] = True
    result["promote_blocked"] = None
    result["operator_force_promote"] = True
    result["promoted_path"] = str(dest)
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    live_path = run_dir / "live_status.json"
    live = _read_json(live_path)
    if live:
        live["promoted"] = True
        live["promote_blocked"] = None
        live["promoted_path"] = str(dest)
        live["updated_at"] = time.time()
        live_path.write_text(json.dumps(live, indent=2, default=str), encoding="utf-8")

    return {"ok": True, "promoted_path": str(dest), "controls": controls}


def _downsample_matrix(
    matrix: list, max_rows: int = 24, max_cols: int = 64
) -> list[list[float]]:
    if not matrix:
        return []
    rows = matrix
    if len(rows) > max_rows:
        step = max(1, len(rows) // max_rows)
        rows = rows[::step][:max_rows]
    out = []
    for row in rows:
        if not isinstance(row, list):
            continue
        r = row
        if len(r) > max_cols:
            step = max(1, len(r) // max_cols)
            r = r[::step][:max_cols]
        out.append([float(x) for x in r])
    return out


def _read_json(path: Path) -> Optional[dict | list]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.debug("json read %s: %s", path, e)
        return None


def _read_yaml(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl_tail(path: Path, n: int = 50) -> list[dict]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines[-n:]:
            if line.strip():
                out.append(json.loads(line))
        return out
    except Exception:
        return []


def _read_text_tail(path: Path, n: int = 100) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except Exception:
        return []
