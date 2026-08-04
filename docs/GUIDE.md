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
2. **Forensically inventories** each sector — mass, depth role, content signature — with **honest evidence tiers**.  
3. **Builds datasets** per sector under **data contracts**.  
4. **Trains adapters** (ESFT / LoRA / optional full-expert masks), by default **one sector at a time** with siblings frozen.  
5. **Probes** pre/post routing share, supports **keep/rollback**, and summarizes **interference**.  
6. **Scores and packages** results as gated **AetherPackages**, labeling dry-run CI completeness vs real MoE readiness.

It is **not** a chat app, not a dense-only SFT script, and not a claim that dry-run equals a fully specialized production MoE.

**Copyright © 2026 AetherAwareness.** Source is public under **PolyForm Noncommercial 1.0.0** — free for noncommercial use; commercial monetization requires a separate grant.

---

## 2. The hole it fills

| Gap in prior open tooling | What AetherForge adds |
|---------------------------|------------------------|
| Fine-tuners treat MoEs like dense nets | Expert-aware selection, freeze, masks |
| Fused expert banks missed by plain PEFT `target_modules` | `target_parameters` + expert-index grad masks |
| No visual map of “which capacity am I editing?” | Expert Group Studio + lattice paint + Fire× |
| No pre-train inventory of sector contents | Sector forensics + evidence tiers |
| One global dataset for all experts | Per-sector datasets + contracts |
| Train all selected experts in one blind pass | Sequential sector workflow with freeze, probe, rollback |
| Promote gates that over-claim dry-runs | CI completeness vs MoE reliability labels |
| Industry knowledge hard-coded into trainers | Domain packs only |
| Local-only or ad-hoc cloud scripts | Vast / RunPod / SSH remote train path |
| No operator console for promote/reject | Training Console + audit trail |

Open dense fine-tuning matured years ago; **open sparse post-training as a factory with forensic control and honesty labels** did not. That is the product niche.

---

## 3. Roles AetherForge can play

| Role | Description |
|------|-------------|
| **Specialist factory** | Carve sectors, bind a domain pack, sequential ESFT on one or few sectors |
| **Broad skills factory** | Multi-corpus, multi-sector, broad posture on multi-GPU boxes |
| **Lattice adapter factory** | Wide PEFT across most experts without full-weight retrain |
| **Forensic lab** | Inventory sector mass and (when probed) routing |
| **CI harness** | Dry-run pipelines + tests for orchestration correctness |
| **Remote job planner** | Sync + train on rented GPUs, pull AetherPackages home |
| **Operator cockpit** | Dashboard for lattice, Sector Forge timeline, human promote gates |
| **Multi-domain pack workshop** | Scaffold packs for any industry without core code changes |

---

## 4. How it helps developers

1. Stop guessing which experts train — see sectors, Fire×, train/freeze flags.  
2. Stop missing fused routed experts — PEFT path is explicit.  
3. Stop poisoning all experts with one blob of data — sector-bound corpora + contracts.  
4. Stop silent plan drift — plan fingerprints freeze membership for a wave.  
5. Stop lying after dry-runs — synthetic affinity and CI packages are watermarked.  
6. Ship reproducible runs — config YAML, hashes, audit JSONL, AetherPackage manifests.  
7. Move from laptop to remote GPU without rewriting the pipeline.  
8. Keep industry content out of the library — packs are data, not forks of the trainer.

---

## 5. What you can run it on

### 5.1 Models

AetherForge is built for **open sparse MoE checkpoints** in general:

- ModuleList-style expert stacks  
- Fused expert-bank architectures (common in large open MoEs)  
- Configurable capacity (layers × experts × top-k) via family profiles / YAML  

You point `model.name` / `model.local_path` at **your** open MoE weights (Hugging Face id or local folder). Optional example family profiles ship under `configs/` for common shapes; they are **profiles**, not a closed list of supported vendors.

Dense-only models are not the product focus.

### 5.2 Hardware

| Hardware | Realistic use |
|----------|----------------|
| **CPU / no GPU** | Validate, DataForge, forensics, dry-run pipeline, dashboard, tests |
| **Single consumer GPU** | Compact MoE PEFT if VRAM fits |
| **Multi high-VRAM GPUs** | Large-MoE PEFT (broad / sequential sector waves) |
| **Rented Vast / RunPod / SSH** | Production adapters off-box |

### 5.3 OS & stack

Linux primary; Python 3.10+; torch/transformers/peft as in `pyproject.toml`.

---

## 6. Installation & download

```bash
# Clone
git clone https://github.com/AetherAwareness/aetherforge.git
cd aetherforge

# Or release archive
# https://github.com/AetherAwareness/aetherforge/releases

bash scripts/install.sh
source .venv/bin/activate
aetherforge doctor
pytest tests/ -q
aetherforge train --recipe dryrun --dry-run
```

---

## 7. Mental model: factory stages

diagnostics → data → affinity → groups (forensics + freeze) → esft (sequential or joint) → router_hygiene → preference → lifecycle → scorecard → package

Default train path is **sequential sectors**. Use `--sector-mode joint` for single-pass ESFT.

---

## 8. Playbooks

### First success (no GPU)

```bash
aetherforge quickstart
aetherforge train -c configs/base.yaml -c recipes/generic_dryrun.yaml --dry-run
aetherforge dashboard
```

### Your industry pack

```bash
aetherforge init my_field --posture broad
# edit configs/domains/my_field.yaml
aetherforge train -c configs/base.yaml -c configs/domains/my_field.yaml --dry-run
```

### Fused-expert PEFT validation

```bash
aetherforge validate-flash
aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c recipes/broad_flash_192gb.yaml --dry-run
```

### Live PEFT on a GPU box

```bash
aetherforge connect vast --host HOST --port PORT
aetherforge remote launch --exec --recipe broad-flash
aetherforge remote pull
```

### Specialist / broad / wide

- **Specialist:** one domain, few train-enabled sectors  
- **Broad:** multi-corpus, multi-sector, multi-GPU  
- **Wide:** lattice-scale LoRA, still PEFT  

### High-stakes promote

```yaml
eval:
  scorecard_thresholds:
    high_stakes: true
    require_human_approval: true
```

Dashboard: Approve / Reject / Force promote.

---

## 9. Surfaces

| Surface | Use |
|---------|-----|
| `train` | Full or partial pipeline |
| `data [--sectors]` | Corpus + sector shards |
| `workflow` | Forensics → datasets → sector plan |
| `forensics` / `groups` | Inventory & carve |
| `dashboard` | Visual console |
| `connect` / `remote` | GPU boxes |
| `scorecard` / `package` | Eval & export |

---

## 10. Evidence tiers & contracts

| Tier | Content claim |
|------|----------------|
| structure_only | Themes zeroed — geometry only |
| assignment | Bound domain/topics/keywords |
| routing_probed | Affinity present (synthetic ≠ high confidence) |

**Contracts:** min samples, real/synth fractions, uniqueness — `block` | `warn` | `off`.

---

## 11. Key artifacts

`config.resolved.yaml` · `affinity.json` · `AFFINITY_SYNTHETIC.txt` · `expert_groups.json` · `sector_forensics.*` · `plan_freeze.json` · `sector_workflow/` · `scorecard.json` · `PROMOTION_LABEL.txt` · `aetherpackage/` · `promoted/`

---

## 12. Possibilities matrix

| Goal | Path |
|------|------|
| Learn MoE post-training safely | Dry-run + dashboard + forensics |
| Domain specialist adapter | Pack + specialist sequential train |
| Multi-skill generalist adapter | Broad recipe + mix_paths |
| Near-full lattice soft update | Wide recipe |
| Audit sector layout | Forensics + Studio |
| CI for your fork | `pytest` + dry-run recipe |
| Research sequential PEFT interference | Interference JSON + keep/rollback |
| Productize for customers | **Commercial license required** |
| High-stakes domain | high_stakes + human approve |
| Remote multi-GPU train | connect + remote launch |

---

## 13. License & contact

- **Copyright:** © 2026 AetherAwareness  
- **Public license:** PolyForm Noncommercial 1.0.0  
- **Commercial monetization:** not granted — see COMMERCIAL.md  
- **Contact:** [admin@aetherawareness.com](mailto:admin@aetherawareness.com)  


---

## 14. Sister tools

No separate public sister repos are published with this release. AetherForge is standalone for MoE post-training.

## 15. Next reading

[HOWTO.md](HOWTO.md) · [product.md](product.md) · [getting-started.md](getting-started.md) · [architecture.md](architecture.md) · [guides/studio.md](guides/studio.md) · [safety.md](safety.md)

*AetherForge · © 2026 AetherAwareness · PolyForm Noncommercial 1.0.0 · admin@aetherawareness.com*
