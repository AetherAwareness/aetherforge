---
title: Safety & limits
layout: default
nav_order: 6
---

# Safety, privacy, and limits
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Decision-support only

Specialists produced by AetherForge are **decision-support tools**, not substitutes for licensed practice in regulated fields (health, law, finance, safety-critical engineering, etc.).

---

## High-stakes promote gates

```yaml
eval:
  scorecard_thresholds:
    high_stakes: true
    require_human_approval: true
    domain_depth_min: 0.65
```

With `require_human_approval`, auto-promote is blocked until a human uses the dashboard **Approve** / **Force promote** controls.

---

## Audit trail

Every run writes:

- `audit.jsonl` — stage events  
- `config.resolved.yaml` — exact config  
- Data fingerprint in DataForge / AetherPackage  

---

## Privacy modes

```yaml
data:
  privacy_mode: local   # local | federated | open
```

- **local** — do not ship raw corpora off-box by default workflows  
- **federated** — reserved for multi-site update patterns  
- **open** — public-safe corpora  

Remote train only syncs **code** by default (rsync excludes artifacts); you control what data paths exist on the GPU box.

---

## Spend gates (remote)

- `aetherforge remote plan` — no SSH train  
- `aetherforge remote launch` without `--exec` — print only  
- `aetherforge remote launch --exec` — real GPU work  
- Desktop menu requires typing **YES**  

AetherForge does **not** auto-rent Vast instances from the UI.

---

## Technical limits (honest)

| Claim | Reality |
|-------|---------|
| Train Flash full safetensors with PEFT | Yes on multi-GPU (e.g. 2×96 GB) |
| Full bf16 Adam over every parameter of a frontier-scale MoE | Not the product path |
| Single 24 GB consumer full Flash load | No |
| Domain packs hard-code medicine | No — industry-agnostic core |

---

## Security reporting

See [SECURITY.md](https://github.com/AetherAwareness/aetherforge/blob/main/SECURITY.md) in the repository root.
