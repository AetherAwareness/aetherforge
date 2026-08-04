"""
Named recipe presets — one flag instead of long -c chains.

  aetherforge train --recipe broad-flash --dry-run
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]

# id -> human card
RECIPE_PRESETS: dict[str, dict[str, Any]] = {
    "dryrun": {
        "label": "Smoke dry-run",
        "description": "Fast structure-only pipeline (no GPU, no model download)",
        "posture": "specialist",
        "configs": ["configs/base.yaml", "recipes/generic_dryrun.yaml"],
        "default_dry_run": True,
        "tags": ["beginner", "ci"],
    },
    "a3b-logistics": {
        "label": "A3B logistics flagship",
        "description": "Qwen A3B-class specialist on sample logistics corpus",
        "posture": "specialist",
        "configs": [
            "configs/base.yaml",
            "configs/qwen_a3b.yaml",
            "recipes/flagship_logistics_a3b.yaml",
        ],
        "tags": ["a3b", "specialist"],
    },
    "flash-domain": {
        "label": "Flash domain specialist",
        "description": "DeepSeek-V4-Flash-0731 single-domain ESFT-LoRA",
        "posture": "specialist",
        "configs": [
            "configs/base.yaml",
            "configs/deepseek_v4_flash.yaml",
            "recipes/flagship_flash_domain.yaml",
        ],
        "tags": ["flash", "specialist"],
    },
    "broad-flash": {
        "label": "Flash BROAD (2×96GB)",
        "description": "Multi-sector multi-domain PEFT — recommended general capability path",
        "posture": "broad",
        "configs": [
            "configs/base.yaml",
            "configs/deepseek_v4_flash.yaml",
            "recipes/broad_flash_192gb.yaml",
        ],
        "tags": ["flash", "broad", "recommended"],
        "recommended": True,
    },
    "wide-flash": {
        "label": "Flash WIDE lattice LoRA",
        "description": "Near-all experts via LoRA (still PEFT, heavier VRAM)",
        "posture": "wide",
        "configs": [
            "configs/base.yaml",
            "configs/deepseek_v4_flash.yaml",
            "recipes/wide_flash_192gb.yaml",
        ],
        "tags": ["flash", "wide"],
    },
}


def list_recipes() -> list[dict[str, Any]]:
    out = []
    for rid, meta in RECIPE_PRESETS.items():
        row = {"id": rid, **meta}
        row["configs_resolved"] = [str(ROOT / c) for c in meta["configs"]]
        row["missing"] = [c for c in meta["configs"] if not (ROOT / c).exists()]
        out.append(row)
    return out


def resolve_recipe(recipe_id: str) -> dict[str, Any]:
    rid = (recipe_id or "").strip().lower().replace("_", "-")
    aliases = {
        "generic": "dryrun",
        "smoke": "dryrun",
        "broad": "broad-flash",
        "wide": "wide-flash",
        "flash": "flash-domain",
        "a3b": "a3b-logistics",
        "logistics": "a3b-logistics",
    }
    rid = aliases.get(rid, rid)
    if rid not in RECIPE_PRESETS:
        known = ", ".join(sorted(RECIPE_PRESETS))
        raise KeyError(f"Unknown recipe '{recipe_id}'. Known: {known}")
    meta = dict(RECIPE_PRESETS[rid])
    paths = []
    for c in meta["configs"]:
        p = ROOT / c
        if not p.exists():
            raise FileNotFoundError(f"Recipe config missing: {p}")
        paths.append(str(p))
    meta["id"] = rid
    meta["config_paths"] = paths
    return meta


def recipe_help_text() -> str:
    lines = ["Named recipes (use: aetherforge train --recipe <id>):", ""]
    for r in list_recipes():
        star = " ★" if r.get("recommended") else ""
        lines.append(f"  {r['id']:<16} {r['label']}{star}")
        lines.append(f"  {'':16} {r['description']}")
        lines.append(f"  {'':16} posture={r.get('posture')}  configs={len(r['configs'])}")
        lines.append("")
    return "\n".join(lines)
