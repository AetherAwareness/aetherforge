---
title: How-to cookbook
layout: default
nav_order: 3
---

# AetherForge — How-to Cookbook
{: .no_toc }

Task-oriented recipes. For the full story see [GUIDE.md](GUIDE.md) and [product.md](product.md).
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Install from release

```bash
curl -sL https://github.com/AetherAwareness/aetherforge/archive/refs/tags/v0.5.1.tar.gz | tar xz
cd aetherforge-0.5.1
bash scripts/install.sh
source .venv/bin/activate
aetherforge doctor
```

## First dry-run (prove the factory)

```bash
aetherforge train --recipe dryrun --dry-run
aetherforge dashboard   # http://127.0.0.1:8765/
```

Expect under `artifacts/runs/…`: `sector_forensics.json`, `plan_freeze.json`, `sector_workflow/`, `PROMOTION_LABEL.txt`.

## Scaffold a domain pack

```bash
aetherforge init my_field --posture broad
# edit configs/domains/my_field.yaml
aetherforge train -c configs/base.yaml -c configs/domains/my_field.yaml --dry-run
```

## Sector forensics only

```bash
aetherforge groups --preview --family generic_moe --num-groups 12
aetherforge forensics --family generic_moe --num-groups 12 --markdown
aetherforge workflow -c configs/base.yaml -c recipes/generic_dryrun.yaml --plan-only --dry-run
```

## Partition data into sector shards

```bash
aetherforge data -c configs/base.yaml -c recipes/generic_dryrun.yaml --sectors --dry-run
```

## Specialist vs broad vs wide

```bash
aetherforge train --recipe dryrun --dry-run
aetherforge train --recipe broad-flash --dry-run
aetherforge train --recipe wide-flash --dry-run
```

## Fused-expert PEFT stack check

```bash
aetherforge validate-flash
aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c recipes/flagship_flash_domain.yaml --dry-run
```

## Remote GPU train (outline)

```bash
aetherforge connect key vast --from-env   # after export VAST_API_KEY
aetherforge connect vast --host HOST --port PORT
aetherforge remote plan --recipe broad-flash
aetherforge remote launch --exec --recipe broad-flash
aetherforge remote logs --tail 100
aetherforge remote pull
```

## High-stakes human promote

```yaml
# in domain or run config
eval:
  scorecard_thresholds:
    high_stakes: true
    require_human_approval: true
```

Then open `aetherforge dashboard` → Approve / Reject / Force promote.

## Read run artifacts

| File | Meaning |
|------|---------|
| `sector_forensics.md` | What each sector contains + evidence tier |
| `plan_freeze.json` | Immutable membership hash for the wave |
| `sector_workflow/` | Per-sector datasets, probes, keep/rollback |
| `scorecard.json` | CI vs MoE labels |
| `promoted/DRY_RUN_NOT_MOE_READY.txt` | Dry-run honesty stamp |

## After training: serve with Switchboard

1. Convert/quantize adapter to your serving format (e.g. GGUF) as you prefer  
2. Register with [Aether Switchboard](https://github.com/AetherAwareness/aether-switchboard)  
3. Map ports on [Aether Constellation](https://github.com/AetherAwareness/aether-constellation)  

## Tests

```bash
pytest tests/ -q
```

## Contact

admin@aetherawareness.com · PolyForm Noncommercial · © 2026 AetherAwareness
