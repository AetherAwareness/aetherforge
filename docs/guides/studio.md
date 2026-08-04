---
title: Studio & forensics
layout: default
parent: Guides
nav_order: 1
---

# Expert Group Studio & sector forensics
{: .no_toc }

See the lattice, carve sectors, and know **what each sector contains** before you train.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Concepts

| Term | Meaning |
|------|---------|
| **Cell** | One `(layer, expert_index)` slot |
| **Sector / group** | Named set of cells with train/freeze + data binding |
| **Active fire** | Params that typically fire in one forward (~13B Flash / ~3B A3B) |
| **Fire×** | Sector expert mass ÷ one active fire |

---

## CLI preview

```bash
# Flash-scale: how many ~13B sectors?
aetherforge groups --preview --family deepseek_v4_flash --num-groups 12

# A3B-scale: ~3B sectors
aetherforge groups --preview --family qwen_a3b --num-groups 8
```

---

## Training Console

```bash
aetherforge dashboard --port 8765
# open http://127.0.0.1:8765/
```

| Control | Action |
|---------|--------|
| **Re-carve** | Repartition N groups (active_slots / affinity / layer_bands / round_robin) |
| **Paint** | Assign lattice cells to the selected sector |
| **Eraser** | Unassign cells |
| **Forensics** | Full inventory table of sectors |
| Sector detail | Domain, data path, topics, train/freeze, content signature, edit guide |

Themes: **NEXUS** · **MATRIX** · **PLASMA** (header toggle).

![Studio]({{ site.baseurl }}/demo/studio.png)

---

## Sequential sector workflow (default training path)

When `training.sector_mode: sequential` (default), each train-enabled sector is handled as its own mini-pipeline:

1. **Forensic assess** — mass, depth role, content signature (`sector_forensics.json`)  
2. **Readiness gate** — `groups.forensics_gate_mode`: `warn` | `block` | `skip`  
3. **Auto-bind** — empty domain/topics/keywords filled from themes (operator bindings never overwritten)  
4. **Sector dataset** — soft-assign global corpus + synthesize fill → `sector_datasets/<id>/train.jsonl`  
5. **ESFT** — only that sector’s experts train; siblings frozen; card at `PRE_TRAIN_FORENSICS.md`

```bash
# Full dry pipeline with per-sector forensics + datasets + ESFT dry-run
aetherforge train -c configs/base.yaml -c recipes/generic_dryrun.yaml --dry-run

# Forensics gate + sector datasets only (no ESFT cards)
aetherforge workflow -c configs/base.yaml -c recipes/generic_dryrun.yaml --plan-only --dry-run

# Partition an existing DataForge corpus into sector shards
aetherforge data -c configs/base.yaml -c recipes/generic_dryrun.yaml --sectors --dry-run

# Legacy joint ESFT (one pass over all selected experts)
aetherforge train --recipe dryrun --sector-mode joint --dry-run
```

Artifacts under each run:

| Path | Content |
|------|---------|
| `sector_forensics.md` | Full lattice inventory |
| `sector_readiness.md` | Pre-train gate verdicts |
| `sector_workflow/` | Sequential workflow root |
| `sector_workflow/sector_datasets/<id>/` | Per-sector train/eval shards |
| `sector_workflow/checkpoints/<id>/PRE_TRAIN_FORENSICS.md` | What was assessed before that sector’s train |

---

## Sector forensics

Before binding data, inventory each sector:

```bash
# Markdown inventory
aetherforge forensics --family deepseek_v4_flash \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --num-groups 12 --markdown

# After a run (routing-aware if affinity.json exists)
aetherforge forensics --plan artifacts/runs/RUN/expert_groups.json \
  --affinity artifacts/runs/RUN/affinity.json --label

# Theme probe bank for live multi-pass probing
aetherforge forensics --probes artifacts/theme_probes.jsonl
```

Each dossier includes:

- Mass (~B) and fire×  
- Layer span / structural role (early → full stack → late)  
- Content signature (theme bank: code, logistics, tools, product_ops, …)  
- Uniqueness vs siblings  
- **Edit recommendations** (split, merge, bind domain, freeze, LR notes)  
- Confidence (higher with affinity matrix + domain binding)

Pipeline `groups` stage writes:

- `sector_forensics.json`  
- `sector_forensics.md`  

API (dashboard):

- `GET /api/runs/<run>/forensics`  
- `GET /api/runs/<run>/groups/<id>` → includes `forensics`  
- `GET /api/studio/forensics?family=…&num_groups=…`  

---

## Config

```yaml
groups:
  enabled: true
  target_num_groups: 12
  target_active_fire_ratio: 1.0
  strategy: active_slots   # affinity | layer_bands | round_robin
  use_for_training: true
  train_scope: top_n       # selected | all_enabled | top_n | all
  train_top_n: 6
  total_params_b: 284
  active_params_b: 13
```

---

## Workflow

1. Dry-run or probe → groups + forensics  
2. Open dashboard → read inventory  
3. Bind domain/data only on intended sectors  
4. Freeze the rest  
5. ESFT / broad / wide train  
6. Scorecard → promote AetherPackage  
