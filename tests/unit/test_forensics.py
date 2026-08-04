"""Sector forensics unit tests."""

from aetherforge.groups.forensics import (
    DEFAULT_THEME_BANK,
    forensics_for_group,
    forensics_markdown,
    probe_texts_from_theme_bank,
    run_model_forensics,
)
from aetherforge.groups.studio import create_studio_plan, analyze_group


def test_theme_bank_and_probes():
    assert "code_software" in DEFAULT_THEME_BANK
    assert "logistics_ops" in DEFAULT_THEME_BANK
    probes = probe_texts_from_theme_bank(per_theme=2)
    assert len(probes) >= 10
    assert all("text" in p and "theme_id" in p for p in probes)


def test_a3b_forensics_fire_ratio():
    """Each A3B sector should report mass vs ~3B active fire."""
    plan = create_studio_plan(
        family="qwen_a3b",
        model_name="Qwen/Qwen3-30B-A3B",
        num_groups=8,
        strategy="active_slots",
    )
    report = run_model_forensics(plan)
    assert report.n_groups == 8
    assert report.capacity["active_params_b"] == 3.0
    assert report.inventory_table
    # sectors near one fire
    fires = [row["fire_x"] for row in report.inventory_table]
    assert min(fires) > 0
    assert max(fires) < 5  # not absurd
    md = forensics_markdown(report)
    assert "Sector" in md
    assert "A3B" in report.narrative or "3.0" in report.narrative


def test_flash_forensics_and_sector_dossier():
    plan = create_studio_plan(
        family="deepseek_v4_flash",
        model_name="deepseek-ai/DeepSeek-V4-Flash-0731",
        num_groups=12,
        strategy="active_slots",
    )
    # bind a domain on first group so theme scoring has signal
    plan.groups[0].domain = "logistics"
    plan.groups[0].topics = ["inventory", "warehouse", "lead time"]
    plan.groups[0].keywords = ["sla", "stockout", "carrier"]

    report = run_model_forensics(plan)
    assert report.n_groups == 12
    assert report.capacity["active_params_b"] == 13.0
    assert report.capacity["num_layers"] == 43

    gid = plan.groups[0].id
    dossier = forensics_for_group(plan, gid)
    assert dossier["mass"]["n_cells"] > 0
    assert dossier["content"]["assigned_domain"] == "logistics"
    # logistics theme should rank high
    themes = dossier["content"]["top_themes"]
    assert themes[0]["id"] == "logistics_ops" or themes[0]["score"] > 0.2
    assert dossier["edit_recommendations"]
    assert "summary" in dossier["content"]


def test_analyze_group_includes_forensics():
    plan = create_studio_plan(family="qwen_a3b", num_groups=4)
    plan.groups[0].domain = "code"
    plan.groups[0].keywords = ["python", "api", "debug"]
    out = analyze_group(plan, plan.groups[0].id, with_forensics=True)
    assert "forensics" in out
    assert out["forensics"]["content"]["summary"]
