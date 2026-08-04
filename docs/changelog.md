---
title: Changelog
layout: default
nav_order: 7
---

# Changelog

## 0.5.1 — Noncommercial license + complete guide

- **License:** PolyForm Noncommercial 1.0.0 (© 2026 AetherAwareness) — free noncommercial use; monetization not granted  
- **NOTICE** + **COMMERCIAL.md** for Required Notice and commercial inquiries  
- **docs/GUIDE.md** — comprehensive setup, roles, utilization matrix, hardware targets  
- GitHub Release artifacts for download  

## 0.5.0 — MoE training fidelity (no-weights)

- **Evidence tiers:** `structure_only` · `assignment` · `routing_probed` with calibrated non-saturating themes  
- **Plan freeze:** immutable membership fingerprint at sector wave start  
- **Pre/post sector probe** + keep/rollback + interference summary (fixture/synthetic affinity OK)  
- **Data contracts** per sector shard (real/synth fractions, uniqueness) with block|warn  
- **Scorecard kinds:** `ci_completeness` vs `moe_reliability`; dry-run never claims MoE readiness  
- **Synthetic affinity watermark** on dry-run affinity + promoted packages  
- Red-team suite: `tests/unit/test_redteam_fidelity.py`

## 0.4.1 — Deep visual Sector Forge

- **Training Console**: Sector Forge bay — pre-train timeline, readiness dossiers, dataset shards, orbital MoE map  
- **Theme AURORA** (ice/teal) + NEXUS / MATRIX / PLASMA  
- **Live telemetry** `live_status.v2`: `sectors.*` wave progress, per-sector forensics blurbs, visual hero labels  
- **Run bundle** includes `sector_forge` for dashboard (workflow + readiness + shards + inventory)  
- **CLI** aesthetic train card with sector wave table  
- Static: `/static/sector-forge.js`

## 0.4.0 — Sector workflow (forensics → data → ESFT)

- **Sequential sector training** (`training.sector_mode: sequential`, default)  
  - Pre-train forensic dossier per sector (`PRE_TRAIN_FORENSICS.md`)  
  - Readiness gate: `groups.forensics_gate_mode` = warn \| block \| skip  
  - Auto-bind unbound sectors from content signatures  
  - Per-sector dataset shards (match + shared mix + synthesize fill)  
  - ESFT only that sector’s experts; siblings frozen  
- **`aetherforge workflow`** — plan-only or full sector dry workflow  
- **`aetherforge data --sectors`** — partition corpus into sector shards  
- **`train --sector-mode sequential|joint`**  
- Modules: `groups/readiness.py`, `data/sector_datasets.py`, `training/sector_workflow.py`  
- `selection_for_group()` for single-sector SelectionPlan  

## 0.3.1 (UX)

- **`aetherforge status`** — dashboard / Vast / Flash weights / next steps  
- **`aetherforge recipes`** + **`train --recipe`** — one-flag presets (`broad-flash` ★)  
- **`aetherforge init <domain>`** — scaffold domain pack + first-run card  
- **`aetherforge quickstart`** — doctor + dry-run in one shot  
- Human train summary card; `doctor --human`  
- Dashboard empty-state onboarding; run list posture / dry / forensics tags  
- Makefile: `status`, `quickstart`, `broad`, `recipes`  

## 0.3.0

- **Rebrand: HiveForge → AetherForge**  
  - Python package / CLI: `aetherforge`  
  - Config home: `~/.aetherforge` (reads legacy `~/.hiveforge` if present)  
  - Desktop launcher: **⬢ AetherForge**  
  - Export unit: **AetherPackage**  
  - Docs / GitHub Pages / HF card updated  

## 0.2.6

- **Postures:** `specialist` · `broad` · `wide` with `apply_posture_defaults`  
- Multi-corpus `data.mix_paths` / `domains`  
- `groups.train_scope`: selected · top_n · all_enabled · all  
- Recipes: `broad_flash_192gb`, `wide_flash_192gb`  
- Desktop launcher default recipe → Flash BROAD  
- Docs: broad work guide  

## 0.2.5

- **Sector forensics** engine, CLI, dashboard panel, pipeline artifacts  
- Theme bank + edit recommendations  

## 0.2.4

- fused-expert MoE profile native PEFT (`target_parameters`, grad masks)  
- `aetherforge validate-flash`  
- Flash config + recipes  

## 0.2.3

- Neural Command UI themes  
- Flagship recipe, lattice paint, remote pull/logs  

## 0.2.x earlier

- Expert Group Studio, domain packs, scorecard, AetherPackage  
- Vast / RunPod / SSH providers  
- Industry bleed removed from trainer core  

---

## Upgrade notes

From specialist-only configs: add `training.posture: broad` or use `recipes/broad_flash_192gb.yaml`.  
Flash fused experts require **peft ≥ 0.15** and **lora_dropout: 0**.
