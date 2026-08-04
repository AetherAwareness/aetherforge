# AetherForge

[![CI](https://github.com/AetherAwareness/aetherforge/actions/workflows/ci.yml/badge.svg)](https://github.com/AetherAwareness/aetherforge/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://AetherAwareness.github.io/aetherforge/)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm%20Noncommercial-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.5.1-informational.svg)](docs/changelog.md)
[![Download](https://img.shields.io/badge/download-GitHub%20Release-success.svg)](https://github.com/AetherAwareness/aetherforge/releases)

**Reliable post-training for open sparse MoE models.**

Carve open **Mixture-of-Experts** checkpoints into visual **expert sectors**, forensically inspect what each sector contains (with honest evidence tiers), train with **specialist / broad / wide** postures, enforce **per-sector data contracts**, promote gated **AetherPackages** — and never confuse dry-run CI completeness with weight-level MoE readiness.

> **v0.5.1** — PolyForm Noncommercial · complete operator guide · evidence tiers · sequential sector workflow · fused-expert PEFT · Training Console  
> **© 2026 AetherAwareness** — free for noncommercial use; **not free to monetize** ([COMMERCIAL.md](COMMERCIAL.md)).

| | |
|---|---|
| **Code** | [github.com/AetherAwareness/aetherforge](https://github.com/AetherAwareness/aetherforge) |
| **Download** | [Releases](https://github.com/AetherAwareness/aetherforge/releases) (source zip/tarball) |
| **Related** | [Aether Switchboard](https://github.com/AetherAwareness/aether-switchboard) · [Aether Constellation](https://github.com/AetherAwareness/aether-constellation) |
| **Complete guide** | [docs/GUIDE.md](docs/GUIDE.md) — setup, run, utilize, all postures |
| **Product deep-dive** | [docs/product.md](docs/product.md) |
| **User docs site** | [AetherAwareness.github.io/aetherforge](https://AetherAwareness.github.io/aetherforge/) |

---

## Why AetherForge?

Sparse MoE models only fire a thin slice of parameters per token. Generic fine-tuning tools:

- treat them like dense models, or  
- miss **fused** expert banks used by many modern MoEs, or  
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
| **Fused-expert PEFT** | `target_parameters` + expert-index grad masks where needed |
| **Scorecard** | **CI completeness** vs **MoE reliability** (honest labels) |
| **Remote train** | Vast / RunPod / SSH — plan → sync → nohup → pull |
| **Domain packs** | Any industry — no hard-coded field bleed |

> Full dense-style retrain of entire multi-hundred-billion MoE lattices is **not** the product goal. AetherForge trains **adapters (ESFT/LoRA)** on open checkpoints (optional selective full-expert updates).

**Model coverage:** AetherForge targets **open sparse MoEs in general** (module-list experts *and* fused expert banks). Built-in family profiles and recipes cover common shapes; you can add capacity/config for other MoEs via YAML.

---

## Quick start

```bash
git clone https://github.com/AetherAwareness/aetherforge.git
cd aetherforge
bash scripts/install.sh
source .venv/bin/activate

aetherforge quickstart                 # doctor + smoke dry-run + next steps
aetherforge status
aetherforge recipes
aetherforge dashboard                  # http://127.0.0.1:8765/
```

```bash
aetherforge train --recipe dryrun --dry-run
aetherforge train --recipe broad-flash --dry-run
aetherforge init logistics --posture broad
aetherforge train --recipe broad-flash -c configs/domains/logistics.yaml --dry-run
aetherforge validate-flash             # prove fused-expert PEFT stack
aetherforge workflow -c configs/base.yaml -c recipes/generic_dryrun.yaml --plan-only --dry-run
```

---

## Sequential sector training (default)

When `training.sector_mode: sequential` (default), each train-enabled sector is its own mini-pipeline:

1. **Freeze plan fingerprint** — membership cannot silently change mid-wave  
2. **Forensic assess** — evidence tier + calibrated themes  
3. **Readiness gate** — warn / block / skip  
4. **Sector dataset + data contract**  
5. **Pre-probe** routing mass  
6. **ESFT** only that sector’s experts; siblings frozen  
7. **Post-probe → keep or rollback**  
8. **Interference summary** across sectors  

```bash
aetherforge train -c configs/base.yaml -c recipes/generic_dryrun.yaml --dry-run
aetherforge train --recipe dryrun --sector-mode joint --dry-run   # single-pass ESFT
```

---

## Evidence tiers (honest forensics)

| Tier | Meaning |
|------|---------|
| **structure_only** | Geometry / mass / depth — **no content claim** |
| **assignment** | Operator or pack bound domain/topics/keywords |
| **routing_probed** | Affinity/routing matrix present (may be synthetic in dry-run) |

---

## Training postures

| Posture | Coverage |
|---------|----------|
| **specialist** | Few experts / selected sectors, one domain |
| **broad** | Many experts + multi-sector + multi-corpus |
| **wide** | Lattice-scale LoRA (still PEFT) |

Details: [docs/guides/postures.md](docs/guides/postures.md)

---

## Scorecard: CI vs MoE readiness

| Label | Meaning |
|-------|---------|
| **ci_completeness** | Pipeline, data, proxies passed (includes dry-run) |
| **moe_reliability** | Live model + non-synthetic routing (+ sector keep when wave ran) |

Dry-run packages are stamped **`DRY_RUN_NOT_MOE_READY`**. Do not treat dry-run promotion as weight-level specialization.

---

## Expert Group Studio & forensics

```bash
aetherforge groups --preview --family generic_moe --num-groups 12
aetherforge forensics --family generic_moe --num-groups 12 --markdown
aetherforge dashboard   # themes: NEXUS · MATRIX · PLASMA · AURORA
```

Industry content lives only in **domain packs** (never hard-coded in the trainer).

---

## Fused-expert MoEs

Many modern MoEs store routed experts as **fused parameter banks** rather than a simple ModuleList of MLPs. AetherForge applies PEFT `target_parameters` plus expert-index grad masks so routed capacity is trainable—not only shared experts.

```bash
aetherforge validate-flash
aetherforge train -c configs/base.yaml -c configs/<moe_family_profile>.yaml \
  -c recipes/flagship_flash_domain.yaml --dry-run
```

Optional YAML family *profiles* under `configs/` illustrate common shapes; the product itself is **any open MoE** you can describe in capacity + PEFT config. See [docs/guides/flash-0731.md](docs/guides/flash-0731.md) for the fused-expert PEFT pattern.

**Hardware:** multi-GPU PEFT is the realistic path for large MoEs. Full bf16 Adam over every parameter of a frontier-scale sparse model is not the product goal.

---

## Pipeline stages

| Stage | Output |
|-------|--------|
| diagnostics | model summary |
| data | train/eval + fingerprint |
| affinity | routing probe + selection |
| groups | sectors + forensics + readiness |
| esft | sequential sector wave or joint adapter |
| router_hygiene | router calibration |
| preference | preference / THD pairs |
| lifecycle | elastic expert plan |
| scorecard | CI vs MoE labels |
| package | AetherPackage (+ promote stamps) |

---

## CLI map

```text
aetherforge doctor | validate | validate-flash | train | data | workflow | probe
aetherforge groups | forensics | dashboard | runs | status | recipes | init | quickstart
aetherforge connect | remote
aetherforge scorecard | package | consult
```

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
| **[Complete guide](docs/GUIDE.md)** | What / why / setup / run / utilize / hardware |
| **[Product explanation](docs/product.md)** | Thorough product deep-dive |
| **[GitHub Pages](https://AetherAwareness.github.io/aetherforge/)** | User guide site |
| [Getting started](docs/getting-started.md) | Install & first dry-run |
| [Architecture](docs/architecture.md) | Pipeline & MoE concepts |
| [Studio & forensics](docs/guides/studio.md) | Lattice + sector dossiers |
| [Fused-expert PEFT](docs/guides/flash-0731.md) | Fused banks & PEFT pattern |
| [Postures](docs/guides/postures.md) | specialist / broad / wide |
| [Safety](docs/safety.md) | High-stakes & privacy |
| [Commercial use](COMMERCIAL.md) | What is / is not monetizable |
| [Changelog](docs/changelog.md) | Versions |

---

## High-stakes fields

```yaml
eval:
  scorecard_thresholds:
    high_stakes: true
    require_human_approval: true
    domain_depth_min: 0.65
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

---

## Limits (read this)

- **Dry-run ≠ trained MoE.** Synthetic affinity and CI scorecards prove the factory, not weight-level specialization.  
- **Live multi-GPU PEFT** is the real train path for large sparse models.  
- **Domain packs** supply industry content — the core never hard-codes a field.  
- Secrets belong in `~/.aetherforge/`, never in the repo.

---

## License & copyright

**Copyright © 2026 AetherAwareness.**

AetherForge is licensed under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**.

| You may | You may not (without a commercial grant) |
|---------|------------------------------------------|
| Use, study, modify for **noncommercial** purposes | Sell AetherForge or a paid fork |
| Share with license + Required Notice | Offer paid SaaS / managed train service built on it |
| Research, education, hobby, public knowledge | Bundle into a commercial product for sale |

See [NOTICE](NOTICE) and [COMMERCIAL.md](COMMERCIAL.md).  
Trademark/names “AetherForge”, “AetherPackage”, “AetherAwareness” remain with AetherAwareness.

**Contact:** [admin@aetherawareness.com](mailto:admin@aetherawareness.com)
