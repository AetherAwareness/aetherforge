---
title: Complete guide
layout: default
nav_order: 1
---

# AetherForge — Complete Guide
{: .no_toc }

What it is, the hole it fills in open-source MoE tooling, who it helps, how to install and run it successfully, every major utilization path, and what hardware/models it targets.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. What AetherForge is

**AetherForge** is a **MoE-native post-training factory** for open sparse Mixture-of-Experts models.

It is software you run (CLI + local Training Console) that:

1. **Carves** a large MoE into named **expert sectors** (groups of layer×expert cells sized relative to one “active fire”).  
2. **Forensically inventories** each sector — mass, depth role, content signature — with **honest evidence tiers** (so geometry is never sold as “knowledge”).  
3. **Builds datasets** per sector (match, mix, synthesize) under **data contracts**.  
4. **Trains adapters** (ESFT / LoRA / optional full-expert masks), by default **one sector at a time** with siblings frozen.  
5. **Probes** pre/post routing share, supports **keep/rollback**, and summarizes **interference**.  
6. **Scores and packages** results as gated **AetherPackages**, labeling dry-run CI completeness vs real MoE readiness.

It is **not**:

- A chat app or agent persona  
- A dense-model-only SFT script  
- A claim that dry-run equals a fully specialized Flash-class model  
- Free-to-monetize SaaS source (see [LICENSE](../LICENSE) / [COMMERCIAL.md](../COMMERCIAL.md))

**Copyright © 2026 AetherAwareness.** Source is public under **PolyForm Noncommercial 1.0.0** — free for noncommercial use; commercial monetization requires a separate grant.

---

## 2. The hole it fills (that open-source MoE tooling mostly lacked)

| Gap in prior open tooling | What AetherForge adds |
|---------------------------|------------------------|
| Fine-tuners treat MoEs like dense nets | Expert-aware selection, freeze, masks |
| Flash-class **fused** experts (`gate_up_proj`/`down_proj` 3D) ignored by plain PEFT `target_modules` | Native **target_parameters** + expert-index grad masks; `validate-flash` |
| No visual map of “which capacity am I editing?” | **Expert Group Studio** + lattice paint + Fire× |
| No pre-train inventory of sector contents | **Sector forensics** + theme bank + evidence tiers |
| One global dataset for all experts | **Per-sector datasets** + contracts + optional multi-pack mix |
| Train all selected experts in one blind pass | **Sequential sector workflow** with freeze, probe, rollback |
| Promote gates that over-claim dry-runs | **CI completeness vs MoE reliability** labels + watermarks |
| Industry knowledge hard-coded into trainers | **Domain packs** only — industry-agnostic core |
| Local-only or ad-hoc cloud scripts | First-class **Vast / RunPod / SSH** remote train path |
| No operator console for promote/reject | **Training Console** (dashboard) + audit trail |

In short: open dense fine-tuning matured years ago; **open sparse post-training as a factory with forensic control and honesty labels** did not. That is the product niche.

---

## 3. Roles AetherForge can play

| Role | Description |
|------|-------------|
| **Specialist factory** | Carve sectors, bind a domain pack, sequential ESFT on one or few sectors |
| **Broad skills factory** | Multi-corpus, multi-sector, broad posture on large multi-GPU boxes |
| **Lattice adapter factory** | Wide PEFT across most experts without full-weight retrain |
| **Forensic lab** | Offline/online inventory of sector mass and (when probed) routing |
| **CI harness** | Dry-run pipelines + red-team tests for orchestration correctness |
| **Remote job planner** | Sync + nohup train on rented GPUs, pull AetherPackages home |
| **Operator cockpit** | Dashboard for lattice, Sector Forge timeline, human promote gates |
| **Multi-domain pack workshop** | Scaffold packs for any industry without core code changes |

---

## 4. How it helps developers

1. **Stop guessing experts** — see sectors, Fire×, train/freeze flags.  
2. **Stop missing Flash experts** — fused-bank PEFT path is explicit.  
3. **Stop poisoning all experts with one blob of data** — sector-bound corpora + contracts.  
4. **Stop silent plan drift** — plan fingerprints freeze membership for a wave.  
5. **Stop lying to yourself after dry-runs** — synthetic affinity and CI packages are watermarked.  
6. **Ship reproducible runs** — config YAML, hashes, audit JSONL, AetherPackage manifests.  
7. **Move from laptop to Vast** without rewriting the pipeline.  
8. **Keep industry content out of the library** — packs are data, not forks of the trainer.

---

## 5. What you can run it on

### 5.1 Model families (primary)

| Family | Class | Active (approx) | Notes |
|--------|-------|-----------------|--------|
| **DeepSeek-V4-Flash** (e.g. 0731) | Sparse MoE | ~13B active / ~284B total | Fused experts; multi-GPU PEFT; `validate-flash` |
| **Qwen A3B-class** (e.g. Qwen3-30B-A3B) | Sparse MoE | ~3B active | Smaller fire; good specialist/dry-run target |
| **Generic MoE** (config-driven) | MoE-like | configurable | Studio capacity estimates; PEFT depends on architecture |

Dense-only models are not the product focus; you can still use domain packs and data tooling, but expert-sector value is for sparse lattices.

### 5.2 Hardware postures

| Hardware | Realistic use |
|----------|----------------|
| **CPU / no GPU** | Config validate, DataForge, forensics structure/assignment, dry-run full pipeline, dashboard, tests |
| **Single consumer GPU** | Small A3B-class PEFT experiments if VRAM fits; not Flash full lattice |
| **2× high-VRAM GPUs (e.g. 2×96GB)** | Flash **broad** PEFT sweet spot (product recipes) |
| **Larger multi-GPU / rented Vast/RunPod** | Wide or multi-sector sequential PEFT; production adapters |
| **Trinity / local APU** | Prefer dry-run + remote launch; keep heavy train off oversubscribed local VRAM |

### 5.3 Operating systems

- Linux (primary; tested path)  
- Any POSIX-like env with Python 3.10+, git, optional CUDA/ROCm/Vulkan torch builds as your stack allows  

### 5.4 What you utilize it *with*

- Open model weights (Hugging Face safetensors / local paths)  
- Domain packs + curated JSON/JSONL corpora  
- Optional OpenRouter / OpenAI-compatible LLM for consult/THD (not required for dry-run)  
- Optional Vast.ai / RunPod / SSH GPU box  

---

## 6. Installation

### 6.1 Download options

**A. Clone (developers)**

```bash
git clone https://github.com/AetherAwareness/aetherforge.git
cd aetherforge
```

**B. Release archive (downloadable)**

GitHub → Releases → latest → `Source code (zip)` / `tar.gz`  
or:

```bash
curl -sL https://github.com/AetherAwareness/aetherforge/archive/refs/heads/main.tar.gz | tar xz
cd aetherforge-main
```

**C. Specific tag**

```bash
git clone --branch v0.5.1 https://github.com/AetherAwareness/aetherforge.git
```

### 6.2 Install

```bash
bash scripts/install.sh
# or:
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### 6.3 Smoke

```bash
aetherforge doctor
aetherforge version
pytest tests/ -q
aetherforge train --recipe dryrun --dry-run
```

---

## 7. Mental model: factory stages

| Stage | Purpose |
|-------|---------|
| diagnostics | Load or dry-skip model summary |
| data | DataForge: curate, synth, gate, fingerprint |
| affinity | Routing probe / selection (synthetic on dry-run) |
| groups | Carve sectors, forensics, readiness, plan freeze |
| esft | Sequential sector wave **or** joint ESFT |
| router_hygiene | Light router calibration |
| preference | Preference / THD pairs |
| lifecycle | Elastic expert plan from utilization |
| scorecard | CI vs MoE labels |
| package | AetherPackage + optional promote/ |

Default train path is **sequential sectors**. Use `--sector-mode joint` for legacy single-pass.

---

## 8. How to run successfully (playbooks)

### 8.1 First success (no GPU)

```bash
source .venv/bin/activate
aetherforge quickstart
aetherforge train -c configs/base.yaml -c recipes/generic_dryrun.yaml --dry-run
aetherforge dashboard   # http://127.0.0.1:8765/
```

Expect:

- `artifacts/runs/<name>-<id>/`  
- `sector_forensics.json`, `plan_freeze.json`, `sector_workflow/`  
- `PROMOTION_LABEL.txt` mentioning dry-run / CI  
- optional `promoted/DRY_RUN_NOT_MOE_READY.txt`  

### 8.2 Your industry pack

```bash
aetherforge init logistics --posture broad
# edit configs/domains/logistics.yaml topics/keywords/actions/curated_path
aetherforge data -c configs/base.yaml -c configs/domains/logistics.yaml --sectors --dry-run
aetherforge train -c configs/base.yaml -c configs/domains/logistics.yaml --dry-run
```

### 8.3 Flash dry validation (still no full train)

```bash
aetherforge validate-flash
aetherforge train -c configs/base.yaml -c configs/deepseek_v4_flash.yaml \
  -c recipes/broad_flash_192gb.yaml --dry-run
```

### 8.4 Live PEFT on a GPU box (outline)

1. Place/open weights (HF id or local path) in model config.  
2. Connect compute:

```bash
export VAST_API_KEY=…
aetherforge connect key vast --from-env
aetherforge connect vast --host HOST --port PORT
```

3. Launch:

```bash
aetherforge remote plan --recipe broad-flash
aetherforge remote launch --exec --recipe broad-flash
aetherforge remote logs --tail 100
aetherforge remote pull
```

4. Inspect scorecard: require `moe_reliability` / `moe_ready` for real claims; use dashboard Approve on high-stakes.

### 8.5 Specialist single-sector

- Studio or CLI: set most groups `train=false`, one group `train=true`  
- Bind domain pack / curated path on that group  
- `training.posture: specialist`, `groups.train_scope: selected`  
- Sequential wave trains only that sector  

### 8.6 Broad multi-skill

- `training.posture: broad`  
- `groups.train_scope: top_n` or multi-domain `data.domains` + `mix_paths`  
- Recipe: `broad_flash_192gb`  

### 8.7 Wide lattice adapter

- `training.posture: wide`  
- Recipe: `wide_flash_192gb`  
- Heavier VRAM; still PEFT, not full bf16  

### 8.8 Forensics-only campaign

```bash
aetherforge groups --preview --family deepseek_v4_flash --num-groups 12
aetherforge forensics --family deepseek_v4_flash --num-groups 12 --markdown
aetherforge workflow -c configs/base.yaml -c recipes/generic_dryrun.yaml --plan-only --dry-run
```

### 8.9 High-stakes promote gate

```yaml
eval:
  scorecard_thresholds:
    high_stakes: true
    require_human_approval: true
    domain_depth_min: 0.65
```

Dashboard: Approve → promote; Reject blocks; Force promote is an explicit override.

---

## 9. Utilizing every major surface

| Surface | Command / path | Use |
|---------|----------------|-----|
| CLI train | `aetherforge train` | Full or partial pipeline |
| Recipes | `aetherforge recipes` / `--recipe` | Named presets |
| Data only | `aetherforge data [--sectors]` | Corpus + sector shards |
| Workflow | `aetherforge workflow [--plan-only]` | Forensics→datasets→sector plan |
| Forensics | `aetherforge forensics` | Inventory sectors |
| Groups | `aetherforge groups` | Carve / preview sectors |
| Dashboard | `aetherforge dashboard` | Visual console |
| Status | `aetherforge status` | Local readiness |
| Connect/Remote | `aetherforge connect` / `remote` | GPU boxes |
| Consult | `aetherforge consult` | Hive debate (optional LLM) |
| Package | `aetherforge package --run-dir …` | Export AetherPackage |
| Scorecard | `aetherforge scorecard` | Re-eval |

### Dashboard themes

NEXUS · MATRIX · PLASMA · AURORA — cosmetic; data is the same.

### Config layering

```bash
aetherforge train -c configs/base.yaml -c configs/deepseek_v4_flash.yaml \
  -c recipes/broad_flash_192gb.yaml -c configs/domains/my_field.yaml \
  -o training.max_steps=200
```

Later `-c` / `-o` overrides earlier.

---

## 10. Evidence tiers & data contracts (operator rules)

| Tier | Meaning | Theme content |
|------|---------|----------------|
| structure_only | Geometry only | Themes **zeroed** |
| assignment | Bound domain/topics/keywords | Calibrated peaks |
| routing_probed | Affinity matrix present | Calibrated; synthetic ≠ high confidence |

**Auto-bind:** may set domain slug from global domain; will **not** invent multi-theme keyword soup from zero-score structure_only dossiers.

**Contracts** (per sector shard): min samples, min real fraction, max synth fraction, uniqueness — `block` | `warn` | `off`.

---

## 11. Artifacts you should learn to read

| Path | Meaning |
|------|---------|
| `config.resolved.yaml` | Exact config for the run |
| `affinity.json` | Routing matrix (+ synthetic metadata) |
| `AFFINITY_SYNTHETIC.txt` | Dry-run watermark |
| `expert_groups.json` | Sector plan |
| `sector_forensics.json` / `.md` | Dossiers + tiers |
| `sector_readiness.json` | Gate verdicts |
| `sector_workflow/plan_freeze.json` | Immutable membership hash |
| `sector_workflow/checkpoints/<id>/PRE_TRAIN_FORENSICS.md` | Pre-train card |
| `sector_workflow/interference.json` | Sibling routing shifts |
| `scorecard.json` | Metrics + kind + labels |
| `PROMOTION_LABEL.txt` | Human one-liner |
| `aetherpackage/` | Export unit |
| `promoted/` | Only after gate (+ honesty stamps on dry-run) |

---

## 12. Failure modes & how to respond

| Symptom | Likely cause | Response |
|---------|--------------|----------|
| Empty sectors / 0 cells | Bad partition | Re-carve groups; check family capacity |
| All themes zero | structure_only | Bind pack or run real affinity probes |
| Multi-theme soup | Old auto-bind bug / manual keyword paste | Clear keywords; re-forensics; use peaked themes only |
| Data contract fail | All-synth / thin / dupes | Add curated real data; raise unique mass; or `contract_mode: warn` for dry demos |
| Plan fingerprint mismatch | Painted lattice mid-wave | Re-freeze / restart wave after paint |
| Sector rolled back | Post-probe regression | Inspect interference; reduce LR; more matched data |
| Scorecard CI but not MoE | Dry-run / synthetic affinity | Expected; train live for `moe_ready` |
| Flash LoRA misses experts | Used only target_modules | Use Flash config + validate-flash |

---

## 13. Possibilities matrix (comprehensive)

| Goal | Path |
|------|------|
| Learn MoE post-training safely | Dry-run + dashboard + forensics |
| Domain specialist adapter | Pack + specialist sequential train |
| Multi-skill generalist adapter | Broad recipe + mix_paths |
| Near-full lattice soft update | Wide recipe |
| Audit what a checkpoint’s sectors look like | Forensics + Studio on plan/affinity |
| CI for your fork of the factory | `pytest` + dry-run recipe in GH Actions |
| Research interference of sequential PEFT | Interference JSON + keep/rollback logs |
| Productize for customers | **Commercial license required** (COMMERCIAL.md) |
| Federated / privacy modes | `data.privacy_mode` + local-only paths |
| High-stakes regulated domain | high_stakes + human approve + depth min |
| Remote multi-GPU Flash | connect vast + broad-flash recipe |
| Local tiny experiments | A3B-class configs if VRAM allows |
| Export for deployment | AetherPackage / promoted/ |

---

## 14. License & protection summary

- **Copyright:** © 2026 AetherAwareness  
- **Public license:** PolyForm Noncommercial 1.0.0 (`LICENSE`)  
- **Required Notice:** see `NOTICE`  
- **Commercial monetization:** not granted — see `COMMERCIAL.md`  
- Free to study, run, modify, and share for **noncommercial** purposes with license intact  

This is **source-available / open for noncommercial use**. It is **not** MIT/Apache “do anything including sell.” That is intentional protection for AetherForge.

---

## 15. Next reading

- [Product explanation](product.md) — deeper product narrative  
- [Getting started](getting-started.md)  
- [Architecture](architecture.md)  
- [Studio & forensics](guides/studio.md)  
- [Flash-0731](guides/flash-0731.md)  
- [Safety](safety.md)  
- [Changelog](changelog.md)  

---

*AetherForge · © 2026 AetherAwareness · PolyForm Noncommercial 1.0.0*
