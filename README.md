# AetherForge

[![CI](https://github.com/AetherAwareness/aetherforge/actions/workflows/ci.yml/badge.svg)](https://github.com/AetherAwareness/aetherforge/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://AetherAwareness.github.io/aetherforge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.5.0-informational.svg)](docs/changelog.md)

**Reliable post-training for open sparse MoE models.**

Carve DeepSeek-V4-Flash-class (~**13B active**) and Qwen A3B-class (~**3B active**) models into visual **expert sectors**, forensically inspect what each sector contains (with honest evidence tiers), train with **specialist / broad / wide** postures, enforce **per-sector data contracts**, promote gated **AetherPackages** — and never confuse dry-run CI completeness with weight-level MoE readiness.

> **v0.5.0** — Evidence tiers · plan freeze · sector probe keep/rollback · data contracts · CI vs MoE scorecard · sequential sector workflow · Flash-0731 PEFT · Training Console

| | |
|---|---|
| **Code** | [github.com/AetherAwareness/aetherforge](https://github.com/AetherAwareness/aetherforge) |
| **Product deep-dive** | [docs/product.md](docs/product.md) — full product explanation |
| **User docs** | [AetherAwareness.github.io/aetherforge](https://AetherAwareness.github.io/aetherforge/) |
| **HF card template** | [`HF_README.md`](HF_README.md) |

---

## Why AetherForge?

Sparse MoE models only fire a thin slice of parameters per token. Generic fine-tuning tools:

- treat them like dense models, or  
- miss **fused** expert banks (DeepSeek-V4 Flash), or  
- give no map of which capacity you are editing.

| Capability | Benefit |
|------------|---------|
| **Expert Group Studio** | Carve the lattice into ~active-fire sectors |
| **Sector forensics** | Mass, depth role, content signature — with **evidence tiers** |
| **Sequential sector workflow** | Forensics → dataset → ESFT per sector (siblings frozen) |
| **Data contracts** | Min real mass, max synth fraction, uniqueness gates |
| **Plan freeze** | Immutable membership fingerprint for each train wave |
| **Probe keep/rollback** | Pre/post routing share; rollback on regression |
| **Postures** | `specialist` · **`broad`** · `wide` lattice LoRA |
| **Flash-0731** | PEFT `target_parameters` + expert-index grad masks |
| **Scorecard** | **CI completeness** vs **MoE reliability** (honest labels) |
| **Remote train** | Vast / RunPod / SSH — plan → sync → nohup → pull |
| **Domain packs** | Any industry — no hard-coded field bleed |

> Full bf16 retrain of ~284B is **not** the product goal. AetherForge trains **adapters (ESFT/LoRA)** on the full open safetensors checkpoint (optional selective full-expert updates).

---

## Quick start

```bash
git clone https://github.com/AetherAwareness/aetherforge.git
cd aetherforge
bash scripts/install.sh
source .venv/bin/activate

aetherforge quickstart                 # doctor + smoke dry-run + next steps
aetherforge status                     # Vast / Flash / runs / what to do next
aetherforge recipes                    # named presets (no long -c chains)
aetherforge dashboard                  # http://127.0.0.1:8765/
```

```bash
# One-flag recipes
aetherforge train --recipe dryrun --dry-run
aetherforge train --recipe broad-flash --dry-run     # multi-sector (2×96GB class)
aetherforge train --recipe wide-flash --dry-run

# Scaffold any industry pack
aetherforge init logistics --posture broad
aetherforge train --recipe broad-flash -c configs/domains/logistics.yaml --dry-run

# Prove Flash-0731 train stack (no full weight download)
aetherforge validate-flash

# Sector workflow plan only (forensics + datasets)
aetherforge workflow -c configs/base.yaml -c recipes/generic_dryrun.yaml --plan-only --dry-run
```

---

## Sequential sector training (default)

When `training.sector_mode: sequential` (default), each train-enabled sector is its own mini-pipeline:

1. **Freeze plan fingerprint** — membership cannot silently change mid-wave  
2. **Forensic assess** — evidence tier + calibrated themes (no multi-theme soup from unbound geometry)  
3. **Readiness gate** — warn / block / skip  
4. **Sector dataset + data contract** — match corpus, synth fill, enforce quality  
5. **Pre-probe** routing mass on the sector’s cells  
6. **ESFT** only that sector’s experts; siblings frozen  
7. **Post-probe → keep or rollback**  
8. **Interference summary** across sectors  

Artifacts under each run include `sector_forensics.json`, `plan_freeze.json`, `sector_workflow/`, `PROMOTION_LABEL.txt`, and (on dry-run) `AFFINITY_SYNTHETIC.txt` + `promoted/DRY_RUN_NOT_MOE_READY.txt`.

```bash
aetherforge train -c configs/base.yaml -c recipes/generic_dryrun.yaml --dry-run
aetherforge train --recipe dryrun --sector-mode joint --dry-run   # legacy single-pass
```

---

## Evidence tiers (honest forensics)

| Tier | Meaning |
|------|---------|
| **structure_only** | Geometry / mass / depth — **no content claim** (themes zeroed) |
| **assignment** | Operator or pack bound domain/topics/keywords |
| **routing_probed** | Affinity/routing matrix present (may still be **synthetic** fixture) |

High-confidence content claims are refused without real routing evidence. Synthetic dry-run affinity is watermarked and never presented as weight-level forensics.

---

## Training postures

| Posture | Coverage | Recipe | 2×96 GB Flash |
|---------|----------|--------|----------------|
| **specialist** | Few experts / selected sectors | `flagship_flash_domain` | easy |
| **broad** | ~28% experts, top‑N sectors, multi-domain | **`broad_flash_192gb`** | **sweet spot** |
| **wide** | Lattice LoRA (still PEFT) | `wide_flash_192gb` | heavier |

Details: [docs/guides/postures.md](docs/guides/postures.md) · [docs/BROAD_WORK.md](docs/BROAD_WORK.md)

---

## Scorecard: CI vs MoE readiness

| Label | Meaning |
|-------|---------|
| **ci_completeness** | Pipeline, data, proxies passed (includes dry-run) |
| **moe_reliability** | Live model + non-synthetic routing (+ sector keep when wave ran) |

Dry-run packages may land under `promoted/` for CI packaging, but they are stamped **`DRY_RUN_NOT_MOE_READY`**. Do not treat dry-run promotion as weight-level specialization.

---

## Expert Group Studio & forensics

```bash
aetherforge groups --preview --family deepseek_v4_flash --num-groups 12
aetherforge forensics --family deepseek_v4_flash --num-groups 12 --markdown
aetherforge dashboard   # themes: NEXUS · MATRIX · PLASMA · AURORA
```

Industry content lives only in **domain packs** (never hard-coded in the trainer).

---

## DeepSeek-V4-Flash-0731

```bash
aetherforge validate-flash
aetherforge train -c configs/base.yaml -c configs/deepseek_v4_flash.yaml \
  -c recipes/flagship_flash_domain.yaml --dry-run
```

Flash uses **fused** expert banks (`gate_up_proj` / `down_proj` 3D). AetherForge applies PEFT `target_parameters` + expert-index grad masks. Plain `target_modules` alone only hits **shared** experts. See [docs/guides/flash-0731.md](docs/guides/flash-0731.md).

**Hardware:** multi-GPU PEFT is the path (e.g. **2×96 GB = 192 GB** is a strong target). Full bf16 Adam on all ~284B params is not.

---

## Pipeline stages

| Stage | Output |
|-------|--------|
| diagnostics | model summary |
| data | train/eval + fingerprint (+ mix_paths) |
| affinity | routing probe + selection (**synthetic watermark** on dry-run) |
| groups | sectors + **sector_forensics** + readiness |
| esft | sequential sector wave or joint adapter checkpoint |
| router_hygiene | router calibration |
| preference | THD / pairs |
| lifecycle | elastic expert plan |
| scorecard | CI vs MoE labels |
| package | AetherPackage (+ promoted/ with honesty stamps) |

---

## CLI map

```text
aetherforge doctor | validate | validate-flash | train | data | workflow | probe
aetherforge groups | forensics | dashboard | runs | status | recipes | init | quickstart
aetherforge connect | remote
aetherforge scorecard | package | consult
```

Full reference: [docs/reference/cli.md](docs/reference/cli.md)

---

## Connect remote GPU

Keys stay in `~/.aetherforge/` (not the git tree).

```bash
export VAST_API_KEY=…
aetherforge connect key vast --from-env
aetherforge connect vast --host HOST --port PORT
aetherforge remote launch --exec --recipe broad-flash
aetherforge remote pull && aetherforge remote logs --tail 80
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| **[Product explanation](docs/product.md)** | Thorough product deep-dive |
| **[GitHub Pages](https://AetherAwareness.github.io/aetherforge/)** | User guide site |
| [Getting started](docs/getting-started.md) | Install & first dry-run |
| [Architecture](docs/architecture.md) | Pipeline & MoE concepts |
| [Studio & forensics](docs/guides/studio.md) | Lattice + sector dossiers |
| [Flash-0731](docs/guides/flash-0731.md) | Fused experts & PEFT |
| [Postures](docs/guides/postures.md) | specialist / broad / wide |
| [Safety](docs/safety.md) | High-stakes & privacy |
| [Changelog](docs/changelog.md) | Versions |

---

## Repository layout

```text
aetherforge/
├── configs/           # base, flash, a3b, domain templates
├── recipes/           # flagship, broad, wide, dryrun
├── packs/             # optional domain packs
├── aetherforge/       # Python package
├── docs/              # GitHub Pages + product.md
├── scripts/           # install, desktop, demo capture
├── tests/             # unit + integration + red-team fidelity
├── README.md
├── HF_README.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

---

## High-stakes fields

```yaml
eval:
  scorecard_thresholds:
    high_stakes: true
    require_human_approval: true
    domain_depth_min: 0.65
```

Generic promote gate — not a field-specific code path. See [docs/safety.md](docs/safety.md).

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Limits (read this)

- **Dry-run ≠ trained MoE.** Synthetic affinity and CI scorecards prove the factory, not weight-level specialization.  
- **Live Flash / multi-GPU PEFT** is the real train path; full 284B bf16 is not.  
- **Domain packs** supply industry content — the core never hard-codes medicine/finance/etc.  
- Secrets and credentials belong in `~/.aetherforge/`, never in the repo.

---

## License

[MIT](LICENSE) © 2026 AetherForge contributors · published by [AetherAwareness](https://github.com/AetherAwareness)
