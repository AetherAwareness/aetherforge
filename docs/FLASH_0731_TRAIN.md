# Training fused-expert MoE profile with AetherForge

**Model:** `<your-open-moe-checkpoint>`  
**Prove command:** `aetherforge validate-flash`  
**Config:** `configs/<moe_family_profile>.yaml`  
**Recipe:** `recipes/flagship_flash_domain.yaml`

## Architecture (verified)

| Field | Value |
|-------|-------|
| model_type | family-specific |
| class | `the model’s causal LM class` |
| layers | 43 |
| routed experts | 256 |
| top-k | 6 |
| shared experts | 1 |
| hidden | 4096 |
| moe intermediate | 2048 |
| total / active | frontier-scale total parameter counts / ~13B |

### Disk vs runtime

- **Disk weight map:** `layers.N.ffn.experts.E.w{1,2,3}` (+ scales for FP4/FP8)
- **Runtime modules:** `model.layers.N.mlp.{gate,experts,shared_experts}`
- **Routed experts:** fused `fused expert modules` with 3D params:
  - `gate_up_proj` shape `(256, 4096, 4096)`
  - `down_proj` shape `(256, 4096, 2048)`
  - `is_transposed=False`
- **Shared:** family shared MLP modules with standard `gate_proj` / `up_proj` / `down_proj` Linears

**Critical:** classic PEFT `target_modules=['gate_proj','up_proj','down_proj']` only hits **shared** experts. AetherForge uses `target_parameters=['gate_up_proj','down_proj']` for routed banks.

## What AetherForge does

1. Detect MoE family from config / name
2. Synthesize 43×256 expert refs (virtual slices of fused banks)
3. Apply PEFT LoRA via `apply_v4_expert_lora`:
   - `target_parameters` on fused banks
   - optional shared Linear modules
   - **`lora_dropout=0`** (PEFT ParamWrapper requirement)
4. Install expert-index **grad masks** on multi-expert LoRA A/B (and base 3D params for `full_esft`) so ESFT only trains selected (layer, expert) slots
5. Expert Group Studio carves ~active-fire sectors (~13B) for data binding

## Commands

```bash
# Hard validation (no full weight download)
aetherforge validate-flash
aetherforge validate-flash --out artifacts/flash_validate.json

# Config check
aetherforge validate -c configs/base.yaml -c configs/<moe_family_profile>.yaml

# Dry pipeline (structure only)
aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c recipes/flagship_flash_domain.yaml --dry-run

# Live on multi-GPU box
aetherforge connect vast --host HOST --port PORT
aetherforge remote launch --exec \
  -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c recipes/flagship_flash_domain.yaml
aetherforge remote pull && aetherforge remote logs
```

## Hardware honesty

| Mode | Feasible? |
|------|-----------|
| Full bf16 train of a frontier-scale MoE on one consumer GPU | **No** |
| PEFT/ESFT LoRA adapters on multi-GPU 4–8×80GB or Vast | **Yes** (recommended) |
| This mini-PC 16GB VRAM alone | **No** for full Flash weight train |
| Dry-run / validate-flash / Group Studio | **Yes** anywhere |

Always train **adapters** (ESFT-LoRA), not full weights. Prefer the released FP4/FP8 checkpoint layout when loading for inference/train if your stack supports it.

## Confidence checklist

- [x] HF id resolves; `AutoConfig` → family config
- [x] Weight index has `ffn.experts` × 43 layers × 256 experts
- [x] Meta `from_config` exposes `fused expert modules` + shapes
- [x] PEFT `target_parameters` accepted (peft≥0.15; tested 0.20)
- [x] LoRA attach smoke + grad-mask hooks
- [x] Wired into `loaders.apply_expert_lora` + `ESFTTrainer`
- [x] Config + recipe + CLI `validate-flash`
- [x] Unit tests + full suite green
