"""UX recipes, init, status."""

from pathlib import Path

from aetherforge.ux.recipes import list_recipes, resolve_recipe, recipe_help_text
from aetherforge.ux.init_project import init_domain
from aetherforge.ux.status import gather_status, format_status_text


def test_recipe_presets():
    rows = list_recipes()
    ids = {r["id"] for r in rows}
    assert "dryrun" in ids
    assert "broad-flash" in ids
    assert "wide-flash" in ids
    assert "qwen38-27b" in ids
    meta = resolve_recipe("broad")
    assert meta["id"] == "broad-flash"
    assert meta["config_paths"]
    assert "broad-flash" in recipe_help_text()


def test_resolve_aliases():
    assert resolve_recipe("smoke")["id"] == "dryrun"
    assert resolve_recipe("a3b")["id"] == "a3b-logistics"
    assert resolve_recipe("qwen38")["id"] == "qwen38-27b"


def test_init_domain(tmp_path, monkeypatch):
    # write into real configs/domains so relative paths work; use unique name
    domain = "ux_test_field"
    pack = Path("configs/domains") / f"{domain}.yaml"
    if pack.exists():
        pack.unlink()
    result = init_domain(domain, posture="broad", recipe="broad-flash")
    assert result["ok"]
    assert Path(result["pack_path"]).exists()
    text = Path(result["pack_path"]).read_text()
    assert "posture: broad" in text
    assert domain in text
    # cleanup
    Path(result["pack_path"]).unlink(missing_ok=True)
    Path(result["card_path"]).unlink(missing_ok=True)


def test_status_shape():
    report = gather_status()
    assert "aetherforge" in report
    assert "next_steps" in report
    assert report["next_steps"]
    text = format_status_text(report)
    assert "AetherForge" in text
    assert "Next steps" in text
