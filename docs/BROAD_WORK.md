# Broad work postures (specialist → broad → wide)

AetherForge is not only a one-sector specialist tool. Three **postures** control how much of the MoE lattice and how much data you update.

| Posture | Expert coverage | Sectors | Data | multi-GPU large-MoE PEFT |
|---------|-----------------|---------|------|----------------|
| **specialist** | top‑k / one domain | selected groups | single domain | easy |
| **broad** | ~25–35% slots | top‑N sectors (e.g. 6/12) | multi-domain + mix_paths | **sweet spot** |
| **wide** | ~all slots (LoRA, no expert masks) | all groups train | large multi-mix | feasible, heavier |

All three still default to **PEFT/ESFT-LoRA** on the full safetensors checkpoint — not full bf16 retrain of 284 B.

## Config knobs

```yaml
training:
  posture: broad   # specialist | broad | wide
  include_attention: true
  mask_unselected_experts: true   # false for wide lattice LoRA

affinity:
  top_k_fraction: 0.28            # broad
  multi_theme_probes: true

groups:
  train_scope: top_n              # selected | all_enabled | top_n | all
  train_top_n: 6

data:
  domains: [aether_public, logistics, tools_agents]
  mix_paths:
    - /path/to/corpus_a.jsonl
    - /path/to/corpus_b.chat.jsonl
  mix_max_per_source: 2000
```

`apply_posture_defaults()` fills sensible values when you only set `training.posture`.

## Recipes

```bash
# Broad (recommended for “general capability” on 192GB)
aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c recipes/broad_flash_192gb.yaml --dry-run

# Wide lattice LoRA (still PEFT)
aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c recipes/wide_flash_192gb.yaml --dry-run

# Vast
aetherforge remote launch --exec -c configs/base.yaml \
  -c configs/<moe_family_profile>.yaml -c recipes/broad_flash_192gb.yaml
```

Desktop **⬢ AetherForge** menu includes these recipes under training recipe pick.

## What broad does under the hood

1. **DataForge** loads `mix_paths` + multi-domain synthetic shells  
2. **Affinity** ranks a large fraction of experts  
3. **Groups** marks top‑N (or all) sectors `train=true`  
4. **ESFT** applies LoRA; optional **attention** modules; expert grad masks unless `wide`  
5. **Router hygiene** + scorecard still gate promotion  

## What broad is *not*

- Not full-parameter Adam on 284 B bf16  
- Not a substitute for pretraining  
- Wide still needs multi-GPU VRAM discipline (batch 1, checkpointing)  

## Choosing a posture

| Goal | Posture |
|------|---------|
| One industry / product surface | specialist |
| Multi-skill generalist adapter (Aether + tools + ops) | **broad** |
| Lattice-wide behavior shift, still adapters | wide |
| Research full-weight SFT | full_esft + much more hardware |
