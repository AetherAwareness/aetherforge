---
name: aetherforge
description: >
  Operate AetherForge (MoE / dense post-training factory) via the aetherforge CLI:
  dry-run, recipes, pack eval, THD, dashboard, Vast remote.
when_to_use: >
  When the operator mentions AetherForge, MoE train, ESFT, sector forge, domain
  packs, Qwen3.8 PEFT, pack eval, or THD.
---

# AetherForge

Project root is the clone (desktop launcher **⬢ AetherForge** if installed).
CLI: `aetherforge` after `source .venv/bin/activate` or `scripts/install.sh`.

## Hard lines

- Live ESFT / QLoRA belongs on rented GPU (`aetherforge remote launch --exec`). Local default is `--dry-run`.
- Official **Qwen3.8-27B HF** is **dense VL**, not sparse MoE. Recipe `qwen38-27b` is QLoRA, `groups.enabled: false`.
- Dry-run scorecards are **CI completeness**, never MoE weight-level readiness.
- Credentials stay in `~/.aetherforge/` — never commit keys.

## Everyday commands

```bash
aetherforge doctor --human
aetherforge recipes
aetherforge train --recipe dryrun --dry-run
aetherforge eval --recipe dryrun --dry-run
aetherforge thd --recipe dryrun --dry-run
aetherforge dashboard          # http://127.0.0.1:8765/
aetherforge status
```

Named recipes: `dryrun` · `a3b-logistics` · `flash-domain` · `broad-flash` · `wide-flash` · `qwen38-27b`.

## Domain pack (any industry)

```bash
aetherforge init my_field --posture specialist --recipe dryrun
# edit configs/domains/my_field.yaml  (topics, keywords, actions, benchmarks)
aetherforge train -c configs/base.yaml -c configs/domains/my_field.yaml --dry-run
aetherforge eval -c configs/base.yaml -c configs/domains/my_field.yaml --dry-run
```

## Live THD

Needs an OpenAI-compat teacher (`AETHERFORGE_LLM_BASE` or `aetherforge connect openrouter`).
Never on `--dry-run`. Pipeline flag: `providers.use_llm_for_thd: true`.

## Vast

```bash
aetherforge connect vast --host HOST --port PORT
aetherforge remote plan --recipe broad-flash
aetherforge remote launch --exec --recipe broad-flash     # costs GPU $
aetherforge remote pull
```

The operator picks the instance. Do not search/rent unless they pasted a machine id.
