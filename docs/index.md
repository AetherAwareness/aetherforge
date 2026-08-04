---
title: Home
layout: home
nav_order: 1
description: "AetherForge — reliable post-training for open sparse MoE models"
permalink: /
---

# AetherForge
{: .fs-9 }

**Reliable post-training for open sparse MoE models.**  
Carve large fused-expert MoE (~13B active) and compact MoE (~3B active) into visual expert sectors, forensically inspect what each sector contains, train specialist or **broad** adapters, and promote gated AetherPackages.
{: .fs-6 .fw-300 }

[Get started]({% link getting-started.md %}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[fused-expert profile guide]({% link guides/flash-0731.md %}){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[GitHub](https://github.com/AetherAwareness/aetherforge){: .btn .fs-5 .mb-4 .mb-md-0 }
[Product explanation]({{ site.baseurl }}/product/){: .btn .fs-5 .mb-4 .mb-md-0 .ms-2 }

---

![Neural Command console]({{ site.baseurl }}/demo/hero.png)

![Demo tour]({{ site.baseurl }}/demo/demo.gif)

*Training Console themes: **NEXUS** · **MATRIX** · **PLASMA***

---

## Why AetherForge?

Mixture-of-Experts models only **fire a thin slice** of parameters per token. Classic fine-tuning tools treat them like dense transformers and either:

- waste compute updating the wrong experts, or  
- miss fused expert banks entirely (many modern MoEs), or  
- give no visual map of *which* capacity you are editing.

AetherForge is built for **open sparse MoE post-training**:

| Capability | What you get |
|------------|----------------|
| **Expert Group Studio** | Carve the lattice into ~active-fire sectors (~one active fire for the model family) |
| **Sector forensics** | Inventory mass, depth role, content signature, edit recommendations |
| **Postures** | `specialist` · **`broad`** · `wide` lattice LoRA |
| **fused-expert profile native** | Fused `gate_up_proj`/`down_proj` via PEFT `target_parameters` |
| **Reliability scorecard** | Promote gates, audit trail, high-stakes human approval |
| **Remote train** | Vast.ai / RunPod / SSH — sync, nohup launch, pull artifacts |
| **Industry-agnostic** | Domain packs only — no hard-coded medical/field bleed |

{: .note }
Full bf16 retrain of frontier-scale total parameter counts is **not** the product goal. AetherForge trains **adapters (ESFT/LoRA)** on the full safetensors checkpoint, with optional selective full-expert updates.

---

## 60-second install

```bash
git clone https://github.com/AetherAwareness/aetherforge.git
cd aetherforge
bash scripts/install.sh
source .venv/bin/activate
aetherforge doctor
aetherforge train -c configs/base.yaml -c recipes/generic_dryrun.yaml --dry-run
aetherforge dashboard   # http://127.0.0.1:8765/
```

---

## Postures at a glance

| Posture | Expert coverage | Best for | Hardware hint |
|---------|-----------------|----------|---------------|
| **specialist** | Few experts / selected sectors | One domain or product surface | Single multi-GPU box |
| **broad** | ~28% slots, top‑N sectors, multi-domain mix | Generalist adapters | **2×96 GB sweet spot** |
| **wide** | Near-all experts (LoRA, no masks) | Lattice-scale behavior shift | 2×96 GB+ |

→ [Broad work guide]({% link guides/postures.md %})

---

## Flagship paths

```bash
# Prove fused-expert profile train stack (no full weight download)
aetherforge validate-flash

# Broad multi-skill dry-run
aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c recipes/broad_flash_192gb.yaml --dry-run

# Connect Vast and launch (costs GPU $)
aetherforge connect vast --host HOST --port PORT
aetherforge remote launch --exec -c configs/base.yaml \
  -c configs/<moe_family_profile>.yaml -c recipes/broad_flash_192gb.yaml
```

→ [Remote / Vast]({% link guides/remote-vast.md %}) · [Recipes]({% link reference/recipes.md %}) · [CLI]({% link reference/cli.md %})

---

## Documentation map

| Section | Contents |
|---------|----------|
| [Getting started]({% link getting-started.md %}) | Install, doctor, first dry-run |
| [Architecture]({% link architecture.md %}) | Pipeline stages, MoE concepts |
| [Studio & forensics]({% link guides/studio.md %}) | Lattice, paint, sector dossiers |
| [fused-expert profile]({% link guides/flash-0731.md %}) | Fused experts, validate-flash, PEFT |
| [Postures]({% link guides/postures.md %}) | specialist / broad / wide |
| [Hardware]({% link guides/hardware.md %}) | VRAM planning, 192 GB guidance |
| [Remote train]({% link guides/remote-vast.md %}) | Vast, RunPod, SSH, desktop launcher |
| [CLI reference]({% link reference/cli.md %}) | Every subcommand |
| [Config reference]({% link reference/config.md %}) | YAML knobs |
| [Safety]({% link safety.md %}) | High-stakes gates, privacy, limits |
| [Changelog]({% link changelog.md %}) | Version history |

---

## License

[PolyForm Noncommercial](https://github.com/AetherAwareness/aetherforge/blob/main/LICENSE) · © 2026 AetherAwareness
