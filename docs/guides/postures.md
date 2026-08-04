---
title: Training postures
layout: default
parent: Guides
nav_order: 3
---

# Training postures
{: .no_toc }

**specialist** · **broad** · **wide** — how much of the lattice and data you update.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Comparison

| | specialist | broad | wide |
|--|------------|-------|------|
| Expert fraction | top‑k (small) | ~25–35% | ~100% (LoRA) |
| Sectors | `train_scope: selected` | `top_n` (e.g. 6/12) | `all` |
| Data | one domain | multi-domain + `mix_paths` | large multi-mix |
| Attention LoRA | optional | on | on |
| Expert grad masks | on | on | **off** |
| Steps (typical) | 100–500 | 800–1500 | 1500–3000 |
| multi-GPU large-MoE PEFT | easy | **sweet spot** | heavier |

All three still default to **PEFT/ESFT-LoRA** on the full safetensors checkpoint.

---

## Config

```yaml
training:
  posture: broad   # specialist | broad | wide
  include_attention: true
  mask_unselected_experts: true

affinity:
  top_k_fraction: 0.28
  multi_theme_probes: true

groups:
  train_scope: top_n
  train_top_n: 6

data:
  domains: [aether_public, logistics, tools_agents]
  mix_paths:
    - /data/corpus_a.jsonl
    - /data/corpus_b.chat.jsonl
  mix_max_per_source: 2000
```

Setting only `training.posture` applies sensible defaults via `apply_posture_defaults()`.

---

## Recipes

```bash
# Broad (recommended general capability on 192GB)
aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c recipes/broad_flash_192gb.yaml --dry-run

# Wide lattice LoRA
aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c recipes/wide_flash_192gb.yaml --dry-run
```

---

## Choosing

| Goal | Posture |
|------|---------|
| One industry / product surface | specialist |
| Multi-skill generalist adapter | **broad** |
| Lattice-wide behavior shift (still adapters) | wide |
| Full-parameter bf16 of whole MoE | not these postures — see hardware |

---

## Implementation map

| Concern | Module |
|---------|--------|
| Defaults | `aetherforge/utils/config.py` → `apply_posture_defaults` |
| Expert ranking | `aetherforge/affinity/expert_selector.py` |
| Sector train flags | `pipeline._stage_groups` + `groups.train_scope` |
| Data mix | `aetherforge/data/forge.py` → `mix_paths` |
| Masks | `loaders.apply_expert_lora` + `mask_unselected_experts` |
