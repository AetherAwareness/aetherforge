# AetherForge Build Status

## Version

**0.5.0** — **MoE fidelity** (evidence tiers, plan freeze, probe keep/rollback, data contracts, CI vs MoE scorecard) · Sector Forge visuals · sequential workflow · Flash-0731 PEFT

## 0.4.0 — Sector-perfect training loop (2026-08-04)

Every train-enabled MoE sector is now handled as a first-class unit:

1. **Forensic inventory** — what mass, depth role, and content signature the sector holds  
2. **Readiness gate** — `warn` / `block` / `skip` before gradients touch that sector  
3. **Auto-bind** — empty domain/topics/keywords filled from forensic themes (never overwrites operator bindings)  
4. **Per-sector datasets** — soft-assign global corpus + synthesize fill from sector signature  
5. **Sequential ESFT** — train only that sector’s experts; siblings frozen; `PRE_TRAIN_FORENSICS.md` per sector  

```bash
# Full dry pipeline (sequential sectors by default)
aetherforge train -c configs/base.yaml -c recipes/generic_dryrun.yaml --dry-run

# Forensics + sector datasets only
aetherforge workflow -c configs/base.yaml -c recipes/generic_dryrun.yaml --plan-only --dry-run

# Partition corpus into sector shards
aetherforge data -c configs/base.yaml -c recipes/generic_dryrun.yaml --sectors --dry-run

# Joint (legacy single-pass) ESFT
aetherforge train --recipe dryrun --sector-mode joint --dry-run
```

Config knobs (`configs/base.yaml`):

| Key | Default | Meaning |
|-----|---------|---------|
| `training.sector_mode` | `sequential` | `sequential` \| `joint` |
| `groups.require_forensics_gate` | `true` | Run readiness before train |
| `groups.forensics_gate_mode` | `warn` | `warn` \| `block` \| `skip` |
| `groups.auto_bind_from_forensics` | `true` | Fill empty bindings from themes |
| `training.sector_min_samples` | `8` | Min shard size (synth fills) |
| `training.sector_shared_fraction` | `0.15` | General pool mix into each shard |

## Correctness fix (2026-07-31)

Removed industry bleed from the trainer core:

- Deleted hard-coded medical / oncology / neurology keyword tables  
- Removed `medical_score_min` → generic `domain_depth_min` + `high_stakes`  
- Synthetic data filled only from **DomainPack** (config/YAML), not Python field maps  
- Domain recipes: `_template.yaml` + `example_logistics.yaml` only  
- Dry-run: `recipes/generic_dryrun.yaml` (`demo_field`)  
- Tests assert no medical strings in resolved packs  

## Verified

```bash
pytest tests/ -q
aetherforge train -c configs/base.yaml -c recipes/generic_dryrun.yaml --dry-run
aetherforge train -c configs/base.yaml -c configs/domains/example_logistics.yaml --dry-run
aetherforge workflow -c configs/base.yaml -c recipes/generic_dryrun.yaml --plan-only --dry-run
```

## How to train any industry

```bash
cp configs/domains/_template.yaml configs/domains/my_field.yaml
# edit topics, keywords, actions, curated_path
aetherforge train -c configs/base.yaml -c configs/domains/my_field.yaml
```

## Training Console

```bash
aetherforge dashboard   # http://127.0.0.1:8765/
aetherforge runs        # JSON list of runs
```

- Live `live_status.json` per run (stage states, %, metrics, events)  
- Dashboard: pipeline, scorecard meters, affinity heatmap, util bars, log tail  
- Controls: human approve / reject / force promote  

## Next cuts

1. Live A3B / Flash FT on Vast with real domain pack + corpus (sector workflow end-to-end)  
2. Real eval harness driven by pack-defined benchmarks  
3. Live multi-theme affinity probes feeding forensic content scores  
4. Live THD via OpenAI-compat LLM  
5. Hermes skill wrapping CLI  
6. Optional WebSocket push (currently 2s poll)  
