# AetherForge Build Status

## Version

**0.5.2** — pack eval · live THD · Qwen3.8 dense PEFT · dashboard SSE · Hermes skill · **0.5.1** PolyForm Noncommercial · MoE fidelity · Sector Forge

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

## 0.5.2 shipped (2026-08-14)

| Cut | Status |
|-----|--------|
| Pack-defined eval harness | **done** — `aetherforge eval`, `pack_eval.json`, scorecard `pack_eval_score` |
| Multi-theme affinity probes | **done** — offline always; live when a bundle is loaded + `affinity.multi_theme_probes` |
| Live THD via OpenAI-compat | **done** — `aetherforge thd --live`, `providers.use_llm_for_thd` (skip dry-run) |
| Hermes skill wrapping CLI | **done** — `~/.hermes/skills/aetherforge` + `skills/hermes/SKILL.md` |
| Live dashboard events | **done** — SSE `GET /api/stream`, poll fallback |
| Qwen3.8-27B train profile | **done** — `configs/qwen38_27b.yaml` + `--recipe qwen38-27b` (dense QLoRA, Vast) |

```bash
aetherforge eval --recipe dryrun --dry-run
aetherforge thd --recipe dryrun --dry-run
aetherforge train --recipe qwen38-27b --dry-run
aetherforge train --recipe dryrun --dry-run
```

## Next cuts

1. Live MoE PEFT on Vast with a real domain pack + corpus (sector workflow end-to-end — operator GPU $)  
2. Live Qwen3.8-27B QLoRA on Vast against a downloaded HF tree  
3. Optional richer dashboard (WebSocket instead of SSE) if SSE proves chatty on long runs  

**Public readiness:** beta for clone / install / dry-run / dashboard / pack eval. Not a turnkey live-train guarantee.  
