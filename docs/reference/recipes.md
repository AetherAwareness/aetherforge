---
title: Recipes
layout: default
parent: Reference
nav_order: 3
---

# Recipe catalog

Recipes are thin YAML overlays on top of `configs/base.yaml` (+ model config).

| Recipe | Posture | Model | Purpose |
|--------|---------|-------|---------|
| `recipes/generic_dryrun.yaml` | specialist | generic | CI / smoke dry-run |
| `recipes/flagship specialist recipe.yaml` | specialist | compact MoE | Example specialist flagship |
| `recipes/flagship_flash_domain.yaml` | specialist | Flash | Domain specialist |
| `recipes/broad_flash_192gb.yaml` | **broad** | Flash | Multi-skill on 2×96 GB |
| `recipes/wide_flash_192gb.yaml` | wide | Flash | Lattice LoRA |

## Domain configs

| Path | Purpose |
|------|---------|
| `configs/domains/_template.yaml` | Copy for any industry |
| `configs/domains/example_logistics.yaml` | Example only |

## Model configs

| Path | Purpose |
|------|---------|
| `configs/base.yaml` | Defaults |
| `configs/<moe_family_profile>.yaml` | example fused-expert PEFT family profile |
| `configs/<moe_family_profile>.yaml` | optional compact-MoE / QLoRA profile |

## Examples

```bash
# Dry-run smoke
aetherforge train -c configs/base.yaml -c recipes/generic_dryrun.yaml --dry-run

# Broad Flash
aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c recipes/broad_flash_192gb.yaml --dry-run

# Custom industry
cp configs/domains/_template.yaml configs/domains/my_field.yaml
aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c configs/domains/my_field.yaml -o data.curated_path=/data/corpus.jsonl
```
