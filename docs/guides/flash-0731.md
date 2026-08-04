---
title: DeepSeek-V4-Flash-0731
layout: default
parent: Guides
nav_order: 2
---

# DeepSeek-V4-Flash-0731
{: .no_toc }

Native support for the open Flash-0731 safetensors checkpoint.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Architecture (verified)

| Field | Value |
|-------|-------|
| HF id | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| model_type | `deepseek_v4` |
| Class | `DeepseekV4ForCausalLM` |
| Layers | **43** |
| Routed experts | **256** |
| Top‑k | **6** |
| Shared | **1** |
| Hidden / MoE intermediate | 4096 / 2048 |
| Total / active | ~**284B** / ~**13B** |

### Disk vs runtime

- **Disk:** `layers.N.ffn.experts.E.w{1,2,3}` (+ scales for FP4/FP8)  
- **Runtime:** `model.layers.N.mlp.{gate,experts,shared_experts}`  
- **Fused banks:** `gate_up_proj` / `down_proj` 3D — **not** ModuleList of MLPs  

{: .warning }
Classic PEFT `target_modules=['gate_proj','up_proj','down_proj']` only hits **shared** experts.  
AetherForge uses `target_parameters=['gate_up_proj','down_proj']` for routed banks.  
`lora_dropout` must be **0** for PEFT ParamWrapper.

---

## Prove the stack

```bash
aetherforge validate-flash
aetherforge validate-flash --out artifacts/flash_validate.json
```

Checks: AutoConfig, weight index, PEFT support, meta `DeepseekV4Experts` shapes, synthetic 43×256 refs, LoRA smoke + grad masks.

---

## Config & recipes

| File | Role |
|------|------|
| `configs/deepseek_v4_flash.yaml` | Model + PEFT targets + groups capacity |
| `recipes/flagship_flash_domain.yaml` | Specialist domain |
| `recipes/broad_flash_192gb.yaml` | Multi-sector multi-domain (**2×96 GB**) |
| `recipes/wide_flash_192gb.yaml` | Lattice-scale LoRA |

```bash
# Structure dry-run
aetherforge train -c configs/base.yaml -c configs/deepseek_v4_flash.yaml \
  -c recipes/broad_flash_192gb.yaml --dry-run

# Live multi-GPU (adapters)
aetherforge connect vast --host HOST --port PORT
aetherforge remote launch --exec -c configs/base.yaml \
  -c configs/deepseek_v4_flash.yaml -c recipes/broad_flash_192gb.yaml
```

Local weights (optional):

```yaml
model:
  name: deepseek-ai/DeepSeek-V4-Flash-0731
  local_path: "/path/to/DeepSeek-V4-Flash-0731"  # full safetensors folder
```

---

## Training methods

| Method | Behavior |
|--------|----------|
| `esft_lora` (default) | PEFT on fused banks + optional expert grad masks |
| `full_esft` | Unfreeze fused banks + slice masks (selective full-weight) |
| `wide` posture | LoRA on essentially all expert slots (masks off) |

See [Hardware]({% link guides/hardware.md %}) for VRAM expectations.

---

## Further reading

Internal deep dive: [FLASH_0731_TRAIN.md](https://github.com/AetherAwareness/aetherforge/blob/main/docs/FLASH_0731_TRAIN.md) in the repo.
