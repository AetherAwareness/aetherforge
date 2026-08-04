---
title: CLI reference
layout: default
parent: Reference
nav_order: 1
---

# CLI reference
{: .no_toc }

```text
aetherforge [-h] [--version] <command> …
```

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Global

| Flag | Description |
|------|-------------|
| `--version` | Print package version |
| `-h` / `--help` | Help |

Config-bearing commands accept:

| Flag | Description |
|------|-------------|
| `-c` / `--config` | YAML path (repeatable; later overrides earlier) |
| `-o` / `--override` | Dotlist e.g. `data.domain=logistics` `training.max_steps=100` |
| `--dry-run` | Skip heavy model load; synthetic affinity |

---

## `train`

Run the full or partial pipeline.

```bash
aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c recipes/broad_flash_192gb.yaml --dry-run

aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  --stages data,affinity,groups,esft,scorecard,package
```

| Flag | Description |
|------|-------------|
| `--stages` | Comma list of stages (aliases allowed) |

Exit codes: `0` success/promoted or dry-run complete; `2` completed but not promoted.

---

## `validate` / `validate-flash`

```bash
aetherforge validate -c configs/base.yaml -c configs/<moe_family_profile>.yaml
aetherforge validate-flash
aetherforge validate-flash --model <your-open-moe-checkpoint> --out report.json
```

| Flag (`validate-flash`) | Description |
|-------------------------|-------------|
| `--model` | HF id (default fused-expert profile) |
| `--skip-weights` | Skip weight index fetch |
| `--skip-meta` | Skip from_config probe |
| `--skip-peft-smoke` | Skip tiny PEFT attach |
| `--out` | Write JSON report |

---

## `doctor` / `version`

```bash
aetherforge doctor
aetherforge version
```

---

## `data` / `probe` / `scorecard` / `package`

```bash
aetherforge data -c configs/base.yaml -c configs/domains/example_logistics.yaml --output artifacts/data_out
aetherforge probe -c configs/base.yaml --texts probe.txt --output artifacts/affinity_probe.json
aetherforge scorecard -c configs/base.yaml --affinity artifacts/affinity.json
aetherforge package -c configs/base.yaml --run-dir artifacts/runs/RUN
```

---

## `groups`

```bash
aetherforge groups --preview --family generic_moe --num-groups 12
aetherforge groups --plan path/expert_groups.json --analyze GROUP_ID
aetherforge groups --preview --family generic_moe --forensics --markdown
```

| Flag | Description |
|------|-------------|
| `--family` | `generic_moe` \| `generic_moe` \| `generic_moe` |
| `--num-groups` | Sector count |
| `--strategy` | `active_slots` \| `affinity` \| `round_robin` \| `layer_bands` |
| `--forensics` | Full sector inventory |
| `--affinity` | affinity.json for routing-aware forensics |
| `--label` | Stamp theme tags onto groups |
| `--markdown` | Print MD inventory |
| `--out` | Write plan JSON |

---

## `forensics`

```bash
aetherforge forensics --family generic_moe --num-groups 12 --markdown
aetherforge forensics --plan expert_groups.json --affinity affinity.json --label
aetherforge forensics --sector GROUP_ID --plan expert_groups.json
aetherforge forensics --probes artifacts/theme_probes.jsonl
aetherforge forensics --out-report artifacts/forensics.md
```

---

## `connect`

```bash
aetherforge connect list
aetherforge connect status
aetherforge connect key vast --from-env
aetherforge connect vast --host IP --port PORT
aetherforge connect runpod --host IP --port PORT
aetherforge connect ssh --host IP --port 22 --user ubuntu
aetherforge connect openrouter --model openrouter/auto
```

---

## `remote`

```bash
aetherforge remote plan -c configs/base.yaml -c recipes/broad_flash_192gb.yaml
aetherforge remote sync
aetherforge remote launch -c …                 # print plan
aetherforge remote launch --exec -c …          # sync + nohup train
aetherforge remote launch --exec --foreground -c …
aetherforge remote pull [--dest artifacts/remote]
aetherforge remote logs --tail 100
```

---

## `dashboard` / `runs` / `consult`

```bash
aetherforge dashboard --host 127.0.0.1 --port 8765
aetherforge runs --limit 30
aetherforge consult "Question" --specialists a,b,c --protocol debate --llm
```
