---
title: FAQ
layout: default
nav_order: 8
---

# FAQ
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

### Does AetherForge train the full Flash safetensors model?

It **loads** the full open checkpoint (or shards) and trains **adapters** (ESFT/LoRA) by default — including fused routed experts. That is the supported “full safetensors” product path. It does **not** mean full bf16 Adam on all ~284B parameters.

### Will 2× RTX 6000 (192 GB) work?

**Yes** for specialist, **broad**, and (with care) **wide** PEFT on Flash-0731. See [Hardware]({% link guides/hardware.md %}).

### Why is my LoRA missing experts on Flash?

You must target fused parameters (`gate_up_proj`, `down_proj`) via PEFT `target_parameters`. `target_modules` alone often only hits **shared** MLPs. Use `configs/deepseek_v4_flash.yaml` or `aetherforge validate-flash`.

### Can I train any industry?

Yes. Copy `configs/domains/_template.yaml`, fill topics/keywords/data paths. The core never hard-codes a vertical.

### Does remote train auto-rent Vast GPUs?

**No.** You rent the instance; AetherForge connects via SSH and only launches when you pass `--exec` or type YES in the desktop menu.

### Where do API keys live?

`~/.aetherforge/credentials.yaml` and connections in `~/.aetherforge/connections.yaml` — not in the git tree.

### Dry-run scorecard failed — is that bad?

Dry-run uses proxy metrics without a real model; failures are common. Live train + real eval matter for promotion.

### How do I publish GitHub Pages?

1. Push `docs/` and `.github/workflows/pages.yml`  
2. Settings → Pages → **GitHub Actions**  
3. Replace `AetherAwareness` in `_config.yml` and README badges  
