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

It is **not** a chat UI, not a dense-model SFT script, and not a claim that dry-run equals a fully specialized Flash-class model.

---

## 2. Who it is for

| Audience | Why they care |
|----------|----------------|
| **Labs / operators** training open MoEs (DeepSeek-V4 Flash-class, Qwen A3B-class) | Need expert-aware PEFT, not dense-style fine-tunes that miss fused experts |
| **Product teams** building domain specialists (logistics, ops, code, …) | Domain packs supply industry content without welding it into the trainer |
| **Platform builders** (e.g. Aether Awareness) | Reproducible config-as-code, audit logs, promote gates, remote Vast/RunPod path |
| **Researchers** studying expert modularity | Sector forensics, affinity probes, interference summaries, keep/rollback |

**Not for:** people who only need a Hugging Face Trainer one-liner on a dense 7B; or anyone expecting a single CLI flag to fully retrain 284B in bf16 on a laptop.

---

## 3. Core problem

Sparse MoEs hide capacity:

- Only top‑k experts fire per token (~3B active on A3B, ~13B on Flash).  
- DeepSeek-V4 Flash stores **fused** expert banks (3D `gate_up_proj` / `down_proj`) — PEFT `target_modules` alone often only hits *shared* MLPs.  
- Operators cannot see which experts they are editing, so generic FT either wastes compute or destroys generality.

AetherForge’s answer is a **control plane** (sectors, forensics, contracts, scorecards) around a **training plane** (ESFT/LoRA + optional full-expert masks).

---

## 4. Product principles

1. **MoE-native** — amplify expert modularity; do not pretend dense.  
2. **Industry-agnostic core** — domain packs are fuel; no hard-coded field tables in Python.  
3. **Honesty over hype** — evidence tiers, synthetic watermarks, dry-run ≠ MoE ready.  
4. **Progressive & reversible** — freeze plans, staged unfreeze, sector keep/rollback.  
5. **Measurable gates** — quality, readiness, scorecard, human approve for high-stakes.  
6. **Reproducible** — config-as-code, seeds, dataset fingerprints, plan fingerprints, audit JSONL.  
7. **Hardware-aware** — multi-GPU PEFT path; remote Vast/RunPod; no fantasy full-bf16 284B product.

---

## 5. Architecture overview

```text
Domain pack + corpora
        │
        ▼
┌───────────────┐
│  DataForge    │  curate · synthesize · quality gates · fingerprint
└───────┬───────┘
        ▼
┌───────────────┐
│ Affinity probe│  routing / selection (live or synthetic dry-run)
└───────┬───────┘
        ▼
┌───────────────┐
│ Expert Group  │  carve lattice → sectors; forensics + readiness
│ Studio        │  plan freeze (immutable fingerprint)
└───────┬───────┘
        ▼
┌───────────────┐
│ Sector wave   │  per sector: dataset contract → pre-probe → ESFT → post-probe
│ (sequential)  │  keep/rollback · interference summary
└───────┬───────┘
        ▼
┌───────────────┐
│ Router hygiene│  preference / THD · lifecycle plan
└───────┬───────┘
        ▼
┌───────────────┐
│ Scorecard     │  CI completeness vs MoE reliability labels
└───────┬───────┘
        ▼
┌───────────────┐
│ AetherPackage │  export · optional promote/ (stamped honesty)
└───────────────┘
```

Optional: **Training Console** (`aetherforge dashboard`) visualizes runs, lattice paint, Sector Forge timeline, affinity, scorecard, operator approve/reject/force-promote.

---

## 6. Expert sectors

A **sector** is a named set of `(layer, expert_index)` cells with:

- train / freeze / enabled flags  
- optional domain, topics, keywords, curated_path, domain_pack  
- capacity estimate vs one **active fire** (Fire×)

Strategies for auto-carve include `active_slots`, `affinity`, `layer_bands`, `round_robin`.  
`groups.train_scope` selects who receives gradients: `selected`, `top_n`, `all_enabled`, `all`.

**Postures** set how wide the update is:

| Posture | Intent |
|---------|--------|
| specialist | Few sectors / experts, one domain |
| broad | Many experts + multi-sector + multi-corpus (default sweet spot on 2×96GB Flash) |
| wide | Near-lattice LoRA; still PEFT by default |

---

## 7. Sector forensics & evidence tiers

Before training a sector, AetherForge inventories:

- **Mass** — estimated expert params vs one active fire  
- **Structure** — layer span, early/mid/late role  
- **Content** — themes, assigned domain/topics/keywords  
- **Distinctiveness** — overlap with siblings  
- **Edit guide** — split/merge/bind/freeze recommendations  

### Evidence tiers (critical honesty layer)

| Tier | When | Content themes |
|------|------|----------------|
| `structure_only` | No assignment, no routing matrix | **Forced zero** — no content claim |
| `assignment` | Domain/topics/keywords/curated bound | Calibrated competitive scores |
| `routing_probed` | Affinity matrix present | Calibrated; **synthetic** matrices cannot claim high confidence |

**Anti-hallucination rules:**

- Unbound geometry must not saturate multi-theme scores to ~1.0.  
- Auto-bind may set a **domain slug** from a global domain for dry-run convenience, but **must not paint multi-theme keywords** from zero-score structure_only dossiers.  
- Theme topics/keywords only from themes with score ≥ threshold (default 0.2).

CLI:

```bash
aetherforge forensics --family deepseek_v4_flash --num-groups 12 --markdown
aetherforge forensics --plan artifacts/runs/RUN/expert_groups.json --affinity affinity.json
```

---

## 8. Sequential sector workflow

Default `training.sector_mode: sequential`:

1. **Plan freeze** — `plan_fingerprint` / `plan_freeze.json`; membership edits mid-wave fail closed  
2. Forensics + readiness gate (`warn` | `block` | `skip`)  
3. Per-sector datasets (soft-assign + optional shared mix + synth fill)  
4. **Data contracts** — min samples, min real fraction, max synth fraction, uniqueness  
5. Pre-probe sector routing mass  
6. ESFT on that sector only (`selection_for_group`)  
7. Post-probe → **keep** or **rollback**  
8. Interference summary for sibling regressions  

Joint mode (`sector_mode: joint`) remains for legacy single-pass ESFT over all selected experts.

Config knobs (see `configs/base.yaml`):

```yaml
training:
  sector_mode: sequential
  sector_min_samples: 8
  sector_shared_fraction: 0.15
  sector_probe_enabled: true
  sector_probe_min_delta: -0.02
  sector_keep_rollback: true
  sector_contract_mode: warn   # block | warn | off
  sector_min_real_fraction: 0.15
  sector_max_synth_fraction: 0.85
  sector_min_unique_ratio: 0.35
groups:
  require_forensics_gate: true
  forensics_gate_mode: warn
  auto_bind_from_forensics: true
```

---

## 9. DataForge & domain packs

**DataForge** builds industry-agnostic corpora:

- curated JSON/JSONL + optional `mix_paths`  
- synthetic self-instruct from **DomainPack** topics/keywords/actions  
- trajectory hive (stub or LLM) for preference pairs  
- quality gates (length, dedupe, toxicity, diversity)  
- content fingerprint for audit  

**Domain packs** (`configs/domains/_template.yaml`, `example_logistics.yaml`) supply:

- `domain`, `topics`, `keywords`, `actions`  
- optional specialists, populations, contexts, high_stakes  
- real `curated_path` when available  

The trainer never embeds medicine/finance/etc. keyword tables in core Python.

```bash
aetherforge init my_field --posture broad
aetherforge data -c configs/base.yaml -c configs/domains/my_field.yaml --sectors --dry-run
```

---

## 10. Training methods

| Method | Role |
|--------|------|
| `esft_lora` | Default — LoRA on selected experts (Flash: `target_parameters` + grad masks) |
| `qlora` | Quantized LoRA path when configured |
| `full_esft` | Unfreeze selected expert params (V4 slice masks) |
| `bar_merge` | Experimental merge posture |

**Flash-0731 note:** fused experts require peft ≥ 0.15 and careful dropout (often 0). `aetherforge validate-flash` proves config/PEFT/weight-map trainability without a full product train.

Specialization and load-balance losses are available as weighted auxiliaries. Router can start frozen and be lightly calibrated in the hygiene stage.

---

## 11. Scorecard & promotion

After training (or dry-run), the **Reliability Scorecard** evaluates proxies:

- domain / structure / general text proxies from pack keywords  
- routing entropy & load-balance CV from affinity  
- optional LM loss when a model is loaded  
- sector wave metrics when sequential workflow ran  

### Scorecard kinds

| Kind | Meaning |
|------|---------|
| `ci_completeness` | Pipeline + data + proxies OK (includes dry-run) |
| `moe_reliability` | Live model + non-synthetic affinity (+ healthy sector keep when wave ran) |

Dry-run **never** sets `full_moe_promoted_readiness`. Packages promoted under dry-run include:

- `promoted/DRY_RUN_NOT_MOE_READY.txt`  
- `promoted/PROMOTION_KIND.json` with `promoted_kind: ci_dry_run`  

High-stakes domains can require human approval via the dashboard before promote.

---

## 12. Remote training

Operators connect compute (Vast.ai, RunPod, SSH box) and LLM APIs (OpenRouter-style) via:

```bash
aetherforge connect vast --host HOST --port PORT
aetherforge remote plan -c … 
aetherforge remote launch --exec -c …
aetherforge remote pull && aetherforge remote logs --tail 80
```

Credentials live under `~/.aetherforge/` (gitignored). Desktop launcher can drive the same path for non-CLI users.

---

## 13. Training Console (dashboard)

`aetherforge dashboard` serves a local Neural Command UI:

- Mission run list + live stage spine  
- **Sector Forge** bay — timeline, readiness, dataset shards, orbit canvas  
- Expert Group Studio — lattice paint, forensics inventory  
- Affinity heatmap, scorecard meters, event stream  
- Operator controls: approve / reject / force promote  
- Themes: NEXUS · MATRIX · PLASMA · AURORA  

Live status schema includes sector wave telemetry (`sectors.*`, visual hero labels) for poll-based updates.

---

## 14. What “success” means

### In-repo / dry-run success (always available)

- Full pipeline completes with fingerprints, forensics tiers, contracts, scorecard  
- Tests green (`pytest tests/ -q`) including red-team fidelity suite  
- Dry-run promotion clearly labeled as CI only  

### Production success (requires GPU + weights)

- Live affinity probes on a loaded model  
- Sequential PEFT on Flash or A3B with measured post-sector routing deltas  
- `moe_reliability` scorecard pass and human-approved promote for high-stakes  

AetherForge is designed so the **factory** is correct before the **furnace** is lit.

---

## 15. What we deliberately do not claim

- Dry-run synthetic affinity is **not** knowledge of what experts contain.  
- Theme banks are **not** a product taxonomy of the world.  
- Sequential sector training without live weights is orchestration + honesty, not proven specialization.  
- Full 284B bf16 retrain is **out of product scope**.  
- No industry is privileged in core code — packs only.

---

## 16. Roadmap posture

Shipped toward:

1. Live multi-theme affinity forensics on real models  
2. Adapter composition policy across sequential PEFT passes  
3. Golden Vast multi-sector Flash run with published scorecard  
4. Pack↔sector binding recipes for multi-specialist hives  
5. Remote resume of interrupted sector waves  

See [BUILD_STATUS.md](BUILD_STATUS.md) and [changelog.md](changelog.md).

---

## 17. Getting started (pointer)

```bash
git clone https://github.com/AetherAwareness/aetherforge.git
cd aetherforge && bash scripts/install.sh && source .venv/bin/activate
aetherforge quickstart
aetherforge train --recipe dryrun --dry-run
aetherforge dashboard
```

Read next: [getting-started.md](getting-started.md) · [architecture.md](architecture.md) · [guides/studio.md](guides/studio.md) · [safety.md](safety.md)

---

*AetherForge · © 2026 AetherAwareness · PolyForm Noncommercial 1.0.0 · MoE post-training factory for open sparse models. Not free to monetize — see COMMERCIAL.md.*
