---
title: Config reference
layout: default
parent: Reference
nav_order: 2
---

# Config reference
{: .no_toc }

Hierarchical YAML merged left→right; `-o` dotlist overrides last.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Merge order

```bash
aetherforge train \
  -c configs/base.yaml \
  -c configs/deepseek_v4_flash.yaml \
  -c recipes/broad_flash_192gb.yaml \
  -o training.max_steps=500 \
  -o data.domain=logistics
```

Later files and `-o` win. `load_config` then applies `apply_posture_defaults()`.

---

## `model`

| Key | Type | Notes |
|-----|------|-------|
| `name` | str | HF id |
| `family` | auto \| deepseek_v4_flash \| qwen_a3b \| generic_moe | |
| `local_path` | str? | Full safetensors directory |
| `load_in_4bit` / `load_in_8bit` | bool | |
| `max_seq_length` | int | |
| `dtype` | str | e.g. bfloat16 |
| `num_experts` / `num_experts_per_tok` / `num_shared_experts` | int? | Hints |
| `trust_remote_code` | bool | default true |
| `expert_parallel` | bool | Flash multi-GPU hint |

---

## `data`

| Key | Type | Notes |
|-----|------|-------|
| `domain` | str | Primary slug |
| `domains` | list[str] | Extra domains for broad synthetic |
| `domain_pack` | path? | YAML/JSON pack |
| `topics` / `keywords` / `actions` | list | Pack overrides |
| `curated_path` | path? | json/jsonl/txt |
| `mix_paths` | list[path] | Broad multi-corpus |
| `mix_max_per_source` | int? | Cap per mix file |
| `eval_path` / `probe_path` | path? | |
| `synthetic.enabled` / `num_samples` | | |
| `synthetic.trajectory_hive` | bool | THD pairs |
| `quality_gates.*` | | diversity, length, toxicity, … |
| `max_train_samples` | int? | |
| `privacy_mode` | local \| federated \| open | |
| `curriculum` | bool | Sort short→long |

Chat JSONL (`messages`) and instruction/output records are normalized to `text`.

---

## `affinity`

| Key | Notes |
|-----|-------|
| `probe_size` | Probe tokens |
| `top_k_experts` | Fixed k |
| `top_k_fraction` | Fraction of lattice (broad/wide) |
| `progressive_unfreeze` | Tiered expand |
| `freeze_router_initially` | |
| `freeze_low_affinity` | |
| `multi_theme_probes` | Broad forensics/selection |

---

## `groups`

| Key | Notes |
|-----|-------|
| `enabled` | Studio stage |
| `target_num_groups` | Auto carve count |
| `target_active_fire_ratio` | Size vs one fire |
| `strategy` | active_slots \| affinity \| layer_bands \| round_robin |
| `use_for_training` | Drive ESFT from groups |
| `train_scope` | selected \| all_enabled \| top_n \| all |
| `train_top_n` | For top_n scope |
| `total_params_b` / `active_params_b` | Capacity overrides |
| `plan_path` | Pre-edited plan |

---

## `training`

| Key | Notes |
|-----|-------|
| `method` | esft_lora \| full_esft \| qlora \| bar_merge |
| `posture` | specialist \| broad \| wide |
| `backend` | auto \| peft \| unsloth \| deepspeed |
| `lora_r` / `lora_alpha` / `lora_dropout` | Flash fused → dropout **0** |
| `target_modules` | Shared / Linear names |
| `target_parameters` | Fused 3D expert names |
| `include_attention` | q/k/v/o LoRA |
| `attention_modules` | list |
| `mask_unselected_experts` | false for wide |
| `learning_rate` / `router_learning_rate` | |
| `max_steps` / `num_epochs` | |
| `per_device_train_batch_size` | usually 1 |
| `gradient_accumulation_steps` | |
| `specialization_loss_weight` | lower for broad/wide |
| `load_balance_loss_weight` | higher for broad/wide |
| `stages` | list of stage intents |
| `router_hygiene_steps` | |
| `gradient_checkpointing` | |
| `output_dir` | default artifacts/runs |

---

## `eval.scorecard_thresholds`

| Key | Notes |
|-----|-------|
| `domain_score` | Min domain metric |
| `general_delta_max` | Max allowed general drop (often negative) |
| `routing_entropy_min` | |
| `load_balance_cv_max` | |
| `hallucination_max` | |
| `domain_depth_min` | Optional competence axis |
| `high_stakes` | |
| `require_human_approval` | Blocks auto promote |

---

## `run` / `providers`

| Key | Notes |
|-----|-------|
| `run.name` | Run folder prefix |
| `run.dry_run` | |
| `providers.compute_profile` | Prefer named ~/.aetherforge connection |
| `providers.llm_profile` | |
| `providers.use_llm_for_synthetic` | Teacher via OpenRouter etc. |
