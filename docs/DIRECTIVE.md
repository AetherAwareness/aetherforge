# AetherForge Directive v1.1 — Industry-Agnostic

**Reliable post-training system for open sparse MoE models, for ANY industry or field.**

Primary model targets:

- DeepSeek-V4-Flash-class (284B total / ~13B active)
- Qwen3.x / A3B-series (~30–35B total / ~3B active)

## Core principle

> The trainer is a **factory**. Industry knowledge is **fuel** (domain packs + corpora), never welded into the machinery.

Hypothetical examples (medicine, markets, threat intel, logistics) are only illustrations of *usage*. They must not appear as hard-coded tables, keyword maps, or special code paths in the core.

## Design principles

1. Reliability over raw iteration speed — measurable gates every stage  
2. MoE-native — amplify expert modularity  
3. Progressive & reversible — AGPS, staged unfreeze, auto-rollback  
4. Compounding by design — trajectories and hive loops first-class  
5. Privacy-preserving — Openveil-ready isolation  
6. Hardware-agnostic, efficiency-first  
7. Reproducible — config-as-code, seeds, hashes, audit  
8. **Industry-agnostic** — domain packs supply topics/keywords/actions/specialists  

## Domain pack contract

Every run resolves a `DomainPack`:

- `domain` (slug)  
- `topics`, `keywords`, `actions`  
- optional `specialists`, `populations`, `contexts`, `angles`, `hints`  
- optional `high_stakes`  
- plus real `curated_path` corpora when available  

Source order: pack file → inline config → generic scaffolds (never field-specific hardcode).

## Pipeline stages

0 Diagnostics → 1 DataForge → 2 Affinity → 3 ESFT → 4 Router hygiene →  
5 Preference/THD → lifecycle plan → 6 Scorecard → 7 AetherPackage → ∞ continuous  

## Failure modes & mitigations

| Failure | Mitigation |
|---------|------------|
| Expert collapse | Entropy + specialization loss + rebirth |
| Catastrophic forgetting | Progressive freeze + general probes |
| Routing drift | Router hygiene |
| Over-specialization | General capability gates |
| Data poisoning | Quality gates + lineage hashes |
| Industry bleed into core | Domain packs only; no field tables in Python |

## Roadmap

1. MVP generic dry-run + pack-driven A3B specialist *(current)*  
2. Live A3B on vast with real corpora  
3. Flash multi-node  
4. Multi-specialist hive across packs for one industry  
5. Openveil federated continuous updates  
