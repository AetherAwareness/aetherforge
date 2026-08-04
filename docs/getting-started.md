---
title: Getting started
layout: default
nav_order: 2
---

# Getting started
{: .no_toc }

Quick path from zero to a dry-run pipeline and the Training Console.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Requirements

| Item | Notes |
|------|--------|
| Python | **3.10+** (3.11 / 3.12 tested in CI) |
| OS | Linux recommended (macOS often works for dry-run / dashboard) |
| GPU | Optional for dry-run; **required** for live train |
| Disk | Models vary — Flash-0731 full safetensors ≈ **160 GB+** |
| Network | Hugging Face for configs / optional weight download |

---

## Install

```bash
git clone https://github.com/AetherAwareness/aetherforge.git
cd aetherforge
bash scripts/install.sh
source .venv/bin/activate
aetherforge doctor --human
aetherforge version
```

## One-command path

```bash
aetherforge quickstart                 # doctor + smoke dry-run + status
aetherforge status                     # what to do next
aetherforge recipes                    # named presets
aetherforge train --recipe broad-flash --dry-run
aetherforge init my_field --posture broad
```

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e ".[dev]"
```

Optional Unsloth path (A3B speed; Flash uses PEFT):

```bash
pip install -e ".[unsloth]"
```

---

## First dry-run (no GPU)

```bash
aetherforge train \
  -c configs/base.yaml \
  -c recipes/generic_dryrun.yaml \
  --dry-run
```

This exercises **DataForge → affinity → groups → forensics → scorecard → package** with synthetic data.

Inspect:

```bash
ls artifacts/runs/
aetherforge runs
aetherforge dashboard   # http://127.0.0.1:8765/
```

---

## Domain pack for any industry

```bash
cp configs/domains/_template.yaml configs/domains/my_field.yaml
# edit: data.domain, topics, keywords, curated_path
aetherforge train -c configs/base.yaml -c configs/domains/my_field.yaml --dry-run
```

Nothing industry-specific is hard-coded in the trainer core — only **domain packs** supply content.

---

## Flash-0731 stack check

```bash
aetherforge validate-flash
# optional report
aetherforge validate-flash --out artifacts/flash_validate.json
```

Proves transformers config, weight index layout, PEFT `target_parameters`, and meta structure **without** downloading full weights.

---

## Broad dry-run (multi-sector)

```bash
aetherforge train \
  -c configs/base.yaml \
  -c configs/deepseek_v4_flash.yaml \
  -c recipes/broad_flash_192gb.yaml \
  --dry-run
```

Expect: 12 sectors, **top-N (6)** train-enabled, `sector_forensics.md` in the run dir.

---

## Desktop launcher (optional)

If you install the desktop shortcut (see [Remote / Vast]({% link guides/remote-vast.md %}))):

- Double-click **⬢ AetherForge**
- Opens Training Console + interactive menu for Vast connect / train

CLI:

```bash
./scripts/aetherforge-desktop.sh
# or: ~/bin/aetherforge-desktop
```

---

## Next steps

- [Architecture]({% link architecture.md %}) — stages and modules  
- [Studio & forensics]({% link guides/studio.md %}) — edit sectors  
- [Flash-0731]({% link guides/flash-0731.md %}) — live multi-GPU train  
- [Postures]({% link guides/postures.md %}) — specialist / broad / wide  
