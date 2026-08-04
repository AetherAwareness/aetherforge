---
title: Product explanation
layout: default
nav_order: 2
---

# AetherForge — Product Explanation
{: .no_toc }

A thorough description of what AetherForge is, who it is for, how the factory works, and what “done” honestly means without live weights.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. One-sentence product

**AetherForge is a MoE-native post-training factory:** it carves open sparse models into train-able expert *sectors*, forensically inventories what each sector is (with honest evidence tiers), builds sector-bound datasets under data contracts, trains adapters (ESFT/LoRA) with specialist or broad postures, and promotes gated **AetherPackages** with scorecards that distinguish CI completeness from MoE reliability.

It is **not** a chat UI, not a dense-model SFT script, and not a claim that dry-run equals a fully specialized production MoE.

---

## 2. Who it is for

| Audience | Why they care |
|----------|----------------|
| **Labs / operators** training open sparse MoEs | Expert-aware PEFT, not dense-style fine-tunes that miss routed capacity |
| **Product teams** building domain specialists | Domain packs supply industry content without welding it into the trainer |
| **Platform builders** | Reproducible config-as-code, audit logs, promote gates, remote GPU path |
| **Researchers** studying expert modularity | Sector forensics, affinity probes, interference summaries, keep/rollback |

**Not for:** people who only need a Hugging Face Trainer one-liner on a dense small model; or anyone expecting a single CLI flag to full-bf16-retrain a frontier-scale sparse model on a laptop.

---

## 3. Core problem

Sparse MoEs hide capacity:

- Only top‑k experts fire per token (thin active slice vs large total parameter count).  
- Many modern MoEs store **fused** expert banks — PEFT `target_modules` alone often only hits *shared* MLPs.  
- Operators cannot see which experts they are editing, so generic FT either wastes compute or destroys generality.

AetherForge’s answer is a **control plane** (sectors, forensics, contracts, scorecards) around a **training plane** (ESFT/LoRA + optional full-expert masks).

---

## 4. Product principles

1. **MoE-native** — amplify expert modularity; do not pretend dense.  
2. **Model-agnostic open MoEs** — works across open sparse architectures you can describe in config (module-list or fused banks).  
3. **Industry-agnostic core** — domain packs are fuel; no hard-coded field tables in Python.  
4. **Honesty over hype** — evidence tiers, synthetic watermarks, dry-run ≠ MoE ready.  
5. **Progressive & reversible** — freeze plans, staged unfreeze, sector keep/rollback.  
6. **Measurable gates** — quality, readiness, scorecard, human approve for high-stakes.  
7. **Reproducible** — config-as-code, seeds, fingerprints, audit JSONL.  
8. **Hardware-aware** — multi-GPU PEFT path; remote rentals; no fantasy full-bf16 of every parameter.

---

## 5. Architecture overview

```text
Domain pack + corpora
        │
        ▼
   DataForge  →  Affinity probe  →  Expert Group Studio (forensics + plan freeze)
        │
        ▼
   Sector wave (dataset · pre-probe · ESFT · post-probe · keep/rollback)
        │
        ▼
   Router hygiene · preference · lifecycle · scorecard · AetherPackage
```

Optional: **Training Console** visualizes runs, lattice paint, Sector Forge timeline, affinity, scorecard, operator approve/reject/force-promote.

---

## 6. Expert sectors

A **sector** is a named set of `(layer, expert_index)` cells with train/freeze flags, optional domain bindings, and capacity vs one **active fire** (Fire×).

**Postures:** specialist · broad · wide — how wide the update is, not which vendor built the base model.

---

## 7. Sector forensics & evidence tiers

| Tier | When | Content themes |
|------|------|----------------|
| `structure_only` | No assignment, no routing matrix | **Forced zero** — no content claim |
| `assignment` | Domain/topics/keywords/curated bound | Calibrated competitive scores |
| `routing_probed` | Affinity matrix present | Calibrated; **synthetic** matrices cannot claim high confidence |

Auto-bind may set a domain slug for dry-run convenience but **must not paint multi-theme keywords** from zero-score structure_only dossiers.

---

## 8. Sequential sector workflow

Default `training.sector_mode: sequential`:

1. Plan freeze (immutable fingerprint)  
2. Forensics + readiness gate  
3. Per-sector datasets + data contracts  
4. Pre-probe → ESFT → post-probe → keep/rollback  
5. Interference summary  

Joint mode remains for single-pass ESFT over all selected experts.

---

## 9. DataForge & domain packs

Industry-agnostic corpora: curated + synthetic + quality gates + fingerprints.  
**Domain packs** supply topics/keywords/actions/curated paths — never hard-coded industry tables in core Python.

---

## 10. Training methods

| Method | Role |
|--------|------|
| `esft_lora` | Default — LoRA on selected experts (fused banks use `target_parameters` + masks) |
| `qlora` | Quantized LoRA when configured |
| `full_esft` | Unfreeze selected expert params with masks |

`aetherforge validate-flash` proves fused-expert PEFT stack trainability without a full product train.

---

## 11. Scorecard & promotion

| Kind | Meaning |
|------|---------|
| `ci_completeness` | Pipeline + data + proxies OK (includes dry-run) |
| `moe_reliability` | Live model + non-synthetic affinity (+ healthy sector keep when wave ran) |

Dry-run **never** sets full MoE promoted readiness. Packages include honesty stamps under dry-run.

---

## 12. Remote training

```bash
aetherforge connect vast --host HOST --port PORT
aetherforge remote launch --exec --recipe broad-flash
aetherforge remote pull
```

Credentials live under `~/.aetherforge/` (gitignored).

---

## 13. What “success” means

**In-repo / dry-run:** full pipeline completes with fingerprints, forensics tiers, contracts, scorecard; tests green; dry-run promotion clearly labeled CI only.

**Production:** live affinity on a loaded open MoE, sequential PEFT with measured post-sector routing deltas, `moe_reliability` scorecard, human approve when high-stakes.

---

## 14. What we deliberately do not claim

- Dry-run synthetic affinity is not knowledge of what experts contain.  
- Theme banks are not a product taxonomy of the world.  
- Sequential sector training without live weights is orchestration + honesty, not proven specialization.  
- Full bf16 retrain of every parameter in a frontier-scale MoE is out of product scope.  
- No industry and no single vendor model is privileged in core code.

---

## 15. Getting started

```bash
git clone https://github.com/AetherAwareness/aetherforge.git
cd aetherforge && bash scripts/install.sh && source .venv/bin/activate
aetherforge quickstart
aetherforge train --recipe dryrun --dry-run
aetherforge dashboard
```

See also [GUIDE.md](GUIDE.md) · [getting-started.md](getting-started.md) · [architecture.md](architecture.md)

---

*AetherForge · © 2026 AetherAwareness · PolyForm Noncommercial 1.0.0 · MoE post-training factory for open sparse models. Not free to monetize — see COMMERCIAL.md.*
