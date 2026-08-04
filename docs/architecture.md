---
title: Architecture
layout: default
nav_order: 3
---

# Architecture
{: .no_toc }

How AetherForge thinks about sparse MoE post-training.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Sparse MoE in one page

Dense models update (roughly) all parameters each step.  
**Mixture-of-Experts** models route each token through a small **active set**:

| Family | Total (approx) | Active fire | Routed experts | Top‑k |
|--------|----------------|-------------|----------------|-------|
| fused-expert MoE profile | frontier-scale total parameter counts | ~**13B** | 256 × 43 layers | 6 |
| Compact open MoE | medium | ~active fire | family-dependent | top-k |

AetherForge sizes **sectors** relative to one active fire so operators can reason:

> “This sector is ~1.0× one Flash fire (~13B of expert mass).”

---

## Design principles

1. **Industry-agnostic core** — domain content only via YAML packs / corpora  
2. **Expert-aware PEFT** — fused 3D expert banks need `target_parameters`, not only `target_modules`  
3. **Visible control** — lattice, forensics, scorecard before promote  
4. **Reversible remote train** — plan → sync → explicit `--exec` / YES confirm  
5. **Auditability** — `audit.jsonl`, data fingerprints, resolved config per run  

---

## Pipeline stages

| Stage | Module | Artifact |
|-------|--------|----------|
| diagnostics | `models.loaders` | `model_summary.json` |
| data | `data.forge` | `data/`, fingerprint |
| affinity | `affinity.probe` + selector | `affinity.json`, `selection_plan.json` |
| groups | `groups.*` + forensics | `expert_groups.json`, `sector_forensics.*` |
| esft | `training.esft_trainer` | `checkpoints/esft/` |
| router_hygiene | `training.router_hygiene` | router-calibrated ckpt |
| preference | THD / DPO export | preference pairs |
| lifecycle | elastic expert plan | mitosis / rebirth plan |
| scorecard | multi-axis gates | `scorecard.json` |
| package | AetherPackage | `aetherpackage/`, optional `promoted/` |

Stage aliases: `sft`→`esft`, `probe`→`affinity`, `studio`→`groups`, etc.  
See CLI `--stages` and `training.stages` in config.

---

## Postures (update width)

```
specialist  →  few experts / selected sectors / one domain
broad       →  ~28% experts + top-N sectors + multi-corpus
wide        →  lattice LoRA, masks off, all sectors train
```

Implemented in `utils.config.apply_posture_defaults`, `affinity.expert_selector`, and `groups.train_scope`.

---

## Fused expert banks

Disk weights use paths like `layers.N.ffn.experts.E.w{1,2,3}`.  
Runtime modules are `model.layers.N.mlp.experts` with:

- `gate_up_proj` shape `[n_experts, …]`  
- `down_proj` shape `[n_experts, …]`  

PEFT **ParamWrapper** attaches multi-expert LoRA; AetherForge installs **grad masks** on expert dim 0 for ESFT (unless `wide`).

---

## Package layout (monorepo)

```
aetherforge/
├── configs/          # base, family profiles, domain templates
├── recipes/          # flagship, broad, wide, dryrun
├── packs/            # optional domain packs
├── aetherforge/        # Python package
│   ├── affinity/
│   ├── data/
│   ├── groups/       # studio + forensics
│   ├── models/       # loaders, family adapters, moe_utils
│   ├── providers/    # vast, runpod, ssh, remote_train
│   ├── training/     # pipeline, esft, scorecard hooks
│   └── viz/          # dashboard
├── docs/             # this site
├── scripts/
└── tests/
```

---

## Run directory contract

Each run under `artifacts/runs/<name>-<id>/` includes:

| File | Purpose |
|------|---------|
| `config.resolved.yaml` | Merged config |
| `live_status.json` | Dashboard live panel |
| `audit.jsonl` | Stage events |
| `affinity.json` | Probe matrix |
| `expert_groups.json` | Studio plan |
| `sector_forensics.json` / `.md` | Content inventory |
| `scorecard.json` | Promote decision |
| `aetherpackage/` | Export unit |
| `promoted/` | Only if gates pass (or force promote) |
