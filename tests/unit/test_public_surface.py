"""Public-release surface: no operator home paths, recipes load, samples exist."""

from pathlib import Path

from aetherforge.ux.recipes import list_recipes, resolve_recipe
from aetherforge.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]
_SKIP_SUFFIX = {".pyc", ".png", ".gif", ".webm", ".jpg"}
_SCAN = ("configs", "recipes", "scripts", "docs", "aetherforge")


def test_no_operator_home_paths_in_public_tree():
    leaks: list[str] = []
    for folder in _SCAN:
        root = ROOT / folder
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix in _SKIP_SUFFIX:
                continue
            if "__pycache__" in p.parts:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "/home/trinity" in text:
                leaks.append(str(p.relative_to(ROOT)))
    assert leaks == [], f"operator home path leaked in: {leaks}"


def test_shipped_sample_corpus_exists():
    assert (ROOT / "data" / "samples" / "logistics" / "train.jsonl").exists()
    assert (ROOT / "data" / "samples" / "logistics" / "eval.jsonl").exists()


def test_every_named_recipe_loads():
    for row in list_recipes():
        assert not row["missing"], row
        meta = resolve_recipe(row["id"])
        cfg = load_config(*meta["config_paths"])
        assert cfg.model.name
        assert cfg.data.domain
