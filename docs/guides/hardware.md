---
title: Hardware
layout: default
parent: Guides
nav_order: 4
---

# Hardware guidance
{: .no_toc }

What fits specialist, broad, and wide PEFT — vs full-weight train.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Large fused-expert MoE memory intuition

| Object | Order of magnitude |
|--------|-------------------|
| Released safetensors (quant-style) | ~**160 GB** disk |
| bf16 full weights | ~**568 GB** |
| + grads + Adam (full) | multi‑**TB** |
| PEFT adapters | **MB–GB** |

**Load full checkpoint + train adapters** ≠ **full Adam on all params in bf16**.

---

## Recommended targets

### 2×96 GB (192 GB total) — **green for AetherForge**

| Job | Fit |
|-----|-----|
| Flash ESFT/LoRA specialist | Excellent |
| Flash **broad** multi-sector | **Sweet spot** |
| Flash **wide** lattice LoRA | Feasible (batch 1, checkpointing) |
| Full bf16 retrain of frontier-scale MoEs | **No** |

Settings:

- `per_device_train_batch_size: 1`  
- `max_seq_length: 4096` (drop to 2048 if OOM)  
- `gradient_checkpointing: true`  
- Prefer recipes `broad_flash_192gb` / `wide_flash_192gb`  

### 2×48 GB (96 GB total)

| Job | Fit |
|-----|-----|
| PEFT + quant load | Possible, tight |
| Broad | Possible with shorter seq / fewer sectors |
| Wide | Risky |

### Consumer single 24 GB

Not for Flash full load. Use dry-run, forensics, dashboard, and remote train on Vast.

---

## Compact MoE class

Compact MoEs often fit **1–2×24–48 GB** with QLoRA/Unsloth.  
Use `configs/<moe_family_profile>.yaml` + `recipes/flagship specialist recipe.yaml`.

---

## Full-weight training (selective)

`method: full_esft` unfreezes fused banks with **slice masks** on selected experts only.

| Scope | Hardware |
|-------|----------|
| Few sectors / experts | 2×96 GB possible |
| Model-wide full updates | Multi-node / many×80 GB+ |

---

## Disk on the train host

| Item | Budget |
|------|--------|
| Flash weights | ~160 GB+ |
| Env + cache | ~20–40 GB |
| Checkpoints / runs | tens of GB |
| **Comfortable free** | **≥ 400 GB** for Flash jobs |

---

## Cost intuition (adapters)

| Goal | GPU-hours (order) |
|------|-------------------|
| First Flash adapter (hours of data) | ~2–8 h on multi-GPU |
| Broad multi-cycle product | ~10–30 h over a week of iteration |
| Full bf16 foundation retrain | weeks / $$$$ |
