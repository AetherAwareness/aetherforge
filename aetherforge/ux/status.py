"""
Human-readable system status + next-step recommendations.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from aetherforge import __version__

ROOT = Path(__file__).resolve().parents[2]


def _flash_local() -> dict[str, Any]:
    candidates = [
        Path.home() / "Downloads" / "LLM's" / "DeepSeek-V4-Flash-0731",
        Path.home() / "Downloads" / "LLMs" / "DeepSeek-V4-Flash-0731",
        Path("/models/DeepSeek-V4-Flash-0731"),
    ]
    for p in candidates:
        if p.is_dir():
            n = len(list(p.glob("model-*-of-*.safetensors")))
            return {
                "ok": n >= 1,
                "path": str(p),
                "shards": n,
                "has_config": (p / "config.json").exists(),
                "has_index": (p / "model.safetensors.index.json").exists(),
            }
    return {"ok": False, "path": None, "shards": 0}


def _dashboard_up(host: str = "127.0.0.1", port: int = 8765) -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=1.0) as r:
            return r.status == 200
    except Exception:
        return False


def _latest_runs(limit: int = 5) -> list[dict[str, Any]]:
    try:
        from aetherforge.viz.run_store import default_runs_root, list_runs

        return list_runs(default_runs_root(), limit=limit)
    except Exception:
        return []


def gather_status() -> dict[str, Any]:
    from aetherforge.providers import connect as pconn

    st = pconn.status()
    flash = _flash_local()
    runs = _latest_runs(8)
    disk = shutil.disk_usage(str(ROOT))
    compute = st.get("compute") or {}
    llm = st.get("llm") or {}

    next_steps: list[str] = []
    if not runs:
        next_steps.append(
            "Run a dry-run:  aetherforge train --recipe dryrun --dry-run"
        )
    else:
        next_steps.append(
            "Open the console:  aetherforge dashboard   → http://127.0.0.1:8765/"
        )
    if not compute.get("ok"):
        next_steps.append(
            "Connect Vast GPU:  aetherforge connect vast --host HOST --port PORT"
        )
    else:
        next_steps.append(
            "Launch remote train:  aetherforge remote launch --exec --recipe broad-flash"
        )
    if not flash.get("ok"):
        next_steps.append(
            "Optional: place Flash-0731 safetensors under ~/Downloads/LLM's/DeepSeek-V4-Flash-0731"
        )
    next_steps.append(
        "Inspect sectors:  aetherforge forensics --family deepseek_v4_flash --num-groups 12 --markdown"
    )
    next_steps.append(
        "Recommended posture on 2×96GB:  aetherforge train --recipe broad-flash --dry-run"
    )

    return {
        "aetherforge": __version__,
        "project_root": str(ROOT),
        "dashboard": {
            "up": _dashboard_up(),
            "url": "http://127.0.0.1:8765/",
        },
        "compute": {
            "ok": bool(compute.get("ok")),
            "message": compute.get("message") or compute.get("error") or "not connected",
            "details": {
                k: compute.get(k)
                for k in ("provider", "host", "port")
                if compute.get(k) is not None
            }
            or compute,
        },
        "llm": {
            "ok": bool(llm.get("ok")),
            "message": llm.get("message") or llm.get("error") or "not connected",
        },
        "flash_local": flash,
        "disk_gb": {
            "free": round(disk.free / 1e9, 1),
            "total": round(disk.total / 1e9, 1),
        },
        "recent_runs": [
            {
                "name": r.get("name"),
                "promoted": r.get("promoted"),
                "domain": r.get("domain"),
                "model": r.get("model"),
                "mtime": r.get("mtime"),
            }
            for r in runs[:5]
        ],
        "next_steps": next_steps,
        "recipes_hint": "aetherforge recipes   # list named presets",
    }


def format_status_text(report: Optional[dict[str, Any]] = None) -> str:
    r = report or gather_status()
    lines = [
        f"AetherForge v{r['aetherforge']}",
        f"  root:      {r['project_root']}",
        f"  disk free: {r['disk_gb']['free']} GB / {r['disk_gb']['total']} GB",
        "",
        "Dashboard:  "
        + (
            f"UP  {r['dashboard']['url']}"
            if r["dashboard"]["up"]
            else f"down — start with: aetherforge dashboard"
        ),
        "Compute:    "
        + (
            f"OK  {r['compute'].get('message')}"
            if r["compute"]["ok"]
            else f"not ready — {r['compute'].get('message')}"
        ),
        "LLM API:    "
        + (
            f"OK  {r['llm'].get('message')}"
            if r["llm"]["ok"]
            else f"optional — {r['llm'].get('message')}"
        ),
    ]
    fl = r.get("flash_local") or {}
    if fl.get("ok"):
        lines.append(
            f"Flash-0731: local {fl.get('shards')} shards @ {fl.get('path')}"
        )
    else:
        lines.append("Flash-0731: no local full checkpoint found (remote HF pull OK)")

    runs = r.get("recent_runs") or []
    lines.append("")
    lines.append(f"Recent runs ({len(runs)}):")
    if not runs:
        lines.append("  (none yet)")
    for run in runs:
        promo = "★ promoted" if run.get("promoted") else "·"
        lines.append(f"  {promo}  {run.get('name')}")

    lines.append("")
    lines.append("Next steps:")
    for i, s in enumerate(r.get("next_steps") or [], 1):
        lines.append(f"  {i}. {s}")
    return "\n".join(lines)
