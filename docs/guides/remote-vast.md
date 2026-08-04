---
title: Remote & Vast.ai
layout: default
parent: Guides
nav_order: 5
---

# Remote training (Vast / RunPod / SSH)
{: .no_toc }

Connect a GPU box you already rent — AetherForge does not silently create billable instances.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Connect once

Credentials live in `~/.aetherforge/` (chmod 600).

```bash
# Optional: list instances via API
export VAST_API_KEY=…
aetherforge connect key vast --from-env

# Required for train: SSH from Vast dashboard
aetherforge connect vast --host 123.45.67.89 --port 22
# RunPod / generic SSH
aetherforge connect runpod --host … --port …
aetherforge connect ssh --host … --port 22 --user ubuntu

aetherforge connect status
```

---

## Plan → sync → launch → pull

```bash
# 1) Preview (no spend)
aetherforge remote plan -c configs/base.yaml \
  -c configs/<moe_family_profile>.yaml -c recipes/broad_flash_192gb.yaml

# 2) Rsync code (excludes .venv, artifacts, .git)
aetherforge remote sync

# 3) Launch — default: nohup background on the box
aetherforge remote launch --exec -c configs/base.yaml \
  -c configs/<moe_family_profile>.yaml -c recipes/broad_flash_192gb.yaml

# Foreground (blocks SSH for the whole job)
aetherforge remote launch --exec --foreground …

# 4) Pull artifacts + logs
aetherforge remote pull
aetherforge remote logs --tail 100
```

Background train writes on the remote host:

- `artifacts/remote_train.nohup.log`  
- `artifacts/remote_train.pid`  

{: .important }
`--exec` and the desktop **YES** confirm are intentional spend gates.

---

## Desktop launcher

```bash
./scripts/aetherforge-desktop.sh
# installs as ⬢ AetherForge on Desktop when using the project install path
```

Menu flow:

1. Open Training Console  
2. Save Vast API key  
3–5. List / connect instance  
6. Choose recipe (**Flash BROAD** default)  
7. Remote plan  
8. Sync  
9. **START TRAINING** (type YES)  
10–11. Pull / logs  

---

## LLM APIs (teachers / consult)

```bash
export OPENROUTER_API_KEY=…
aetherforge connect key openrouter --from-env
aetherforge connect openrouter --model "openrouter/auto"
aetherforge consult "Tradeoffs of multi-sector PEFT" --llm --specialists a,b,c
```

Supported: OpenRouter, OpenAI, Together, Fireworks, Groq, and other OpenAI-compatible APIs.

---

## Troubleshooting

| Symptom | Check |
|---------|--------|
| `No compute connection` | `aetherforge connect vast --host … --port …` |
| SSH failed | Instance running? Port from Vast dashboard? Key? |
| OOM | Shorter seq, fewer train sectors, broad not wide |
| Empty pull | Job still running? Check `remote logs` / nohup log |
