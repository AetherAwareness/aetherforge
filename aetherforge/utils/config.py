"""Hierarchical YAML config with Pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen3-30B-A3B"
    family: Literal[
        "auto", "deepseek_v4_flash", "qwen_a3b", "qwen38_dense", "generic_moe"
    ] = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    expert_parallel: bool = False
    tensor_parallel: int = 1
    max_seq_length: int = 8192
    dtype: str = "bfloat16"
    # Architecture hints (overridden by auto-detect when possible)
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    num_shared_experts: Optional[int] = None
    local_path: Optional[str] = None
    revision: Optional[str] = None


class SyntheticConfig(BaseModel):
    enabled: bool = True
    num_samples: int = 10_000
    trajectory_hive: bool = True
    self_instruct: bool = True
    rag_grounded: bool = False
    teacher_model: Optional[str] = None


class QualityGatesConfig(BaseModel):
    min_diversity: float = 0.70
    max_toxicity: float = 0.05
    factuality_threshold: float = 0.80
    min_length: int = 32
    max_length: int = 16_384
    dedupe_threshold: float = 0.92
    require_language: Optional[str] = "en"


class DataConfig(BaseModel):
    """
    Industry-agnostic data config.

    Domain content (topics, keywords, actions, specialists) comes from:
      - domain: slug name for the field (any industry)
      - domain_pack: optional YAML/JSON path with topics/keywords/actions
      - topics / keywords / actions: inline lists (override pack)
      - curated_path: real corpus for that industry
    Nothing industry-specific is hard-coded in the trainer.
    """

    domain: str = "general"
    domain_pack: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    curated_path: Optional[str] = None
    eval_path: Optional[str] = None
    probe_path: Optional[str] = None
    # Broad work: mix multiple corpora (json/jsonl paths) + optional multi-domain labels
    mix_paths: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    # Cap per mix source before concat (None = all)
    mix_max_per_source: Optional[int] = None
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    quality_gates: QualityGatesConfig = Field(default_factory=QualityGatesConfig)
    curriculum: bool = True
    privacy_mode: Literal["local", "federated", "open"] = "local"
    max_train_samples: Optional[int] = None
    seed: int = 42
    # Build per-sector dataset shards (also used by sequential sector workflow)
    sector_datasets: bool = True
    # Soft-assign threshold when partitioning global corpus → sectors
    sector_min_match: float = 0.18
    # Pack-owned eval cases (dicts matching PackBenchmark). Inline override of pack file.
    benchmarks: list[dict[str, Any]] = Field(default_factory=list)


class AffinityConfig(BaseModel):
    probe_size: int = 2048
    top_k_experts: int = 16
    top_k_fraction: Optional[float] = None  # alternative to fixed k
    progressive_unfreeze: bool = True
    use_gradient_contribution: bool = True
    freeze_router_initially: bool = True
    freeze_low_affinity: bool = True
    mitosis_overload_threshold: float = 3.0  # relative to mean util
    min_affinity_score: float = 0.0
    # Broad work: multi-theme probe labels (forensics + selection)
    multi_theme_probes: bool = False


class GroupsConfig(BaseModel):
    """
    Expert Group Studio — carve MoE lattice into train-able sectors.

    For Flash-class models (~13B active), target_num_groups controls how many
    ~active-sized sectors the user can call up and assign data to.
    """

    enabled: bool = True
    # How many sectors to auto-create (clamped by capacity)
    target_num_groups: int = 8
    # 1.0 ≈ size each sector near one routing fire (~13B experts on Flash)
    target_active_fire_ratio: float = 1.0
    strategy: Literal["active_slots", "affinity", "round_robin", "layer_bands"] = (
        "active_slots"
    )
    # Use studio groups as ESFT selection (instead of pure top-k affinity)
    use_for_training: bool = True
    # Which sectors receive gradients:
    #   selected     — only groups with train=true (default specialist)
    #   all_enabled  — all enabled groups
    #   top_n        — top train_top_n groups by mean affinity
    #   all          — every group (broad / wide lattice)
    train_scope: Literal["selected", "all_enabled", "top_n", "all"] = "selected"
    train_top_n: int = 4
    # Optional pre-saved plan path
    plan_path: Optional[str] = None
    # Capacity overrides (optional)
    total_params_b: Optional[float] = None
    active_params_b: Optional[float] = None
    # Pre-sector forensic readiness gate
    #   warn  — inventory + soft fail; still train (default)
    #   block — refuse sectors that fail readiness (empty / unbound / thin data)
    #   skip  — forensics only, never fail
    require_forensics_gate: bool = True
    forensics_gate_mode: Literal["block", "warn", "skip"] = "warn"
    # Fill empty domain/topics/keywords from forensic theme signature
    auto_bind_from_forensics: bool = True
    # Only auto-bind when evidence tier is assignment/routing (never invent from structure)
    auto_bind_requires_theme_peak: float = 0.25


class TrainingConfig(BaseModel):
    """
    posture controls how *wide* the update is (not just method):

      specialist — few experts / few sectors, one domain (default)
      broad      — many experts + multi-sector + multi-corpus (generalist skills)
      wide       — near-lattice LoRA / most experts; still PEFT by default
                   (full-weight whole-model remains method=full_esft + huge hardware)
    """

    method: Literal["esft_lora", "full_esft", "qlora", "bar_merge"] = "esft_lora"
    posture: Literal["specialist", "broad", "wide"] = "specialist"
    backend: Literal["unsloth", "peft", "deepspeed", "auto"] = "auto"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "gate_proj",
            "up_proj",
            "down_proj",
            "gate_up_proj",
        ]
    )
    # Fused MoE expert banks (DeepSeek-V4 Flash gate_up_proj/down_proj 3D params).
    # PEFT LoraConfig(target_parameters=...); empty = family default.
    target_parameters: list[str] = Field(default_factory=list)
    # Broad/wide: also attach LoRA to attention projections when present
    include_attention: bool = False
    attention_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # When posture=wide and method=esft_lora: skip expert grad masks (all experts train)
    mask_unselected_experts: bool = True
    learning_rate: float = 2e-4
    router_learning_rate: float = 5e-5
    num_epochs: float = 1.0
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    specialization_loss_weight: float = 0.1
    load_balance_loss_weight: float = 0.01
    ewc_weight: float = 0.0
    kd_weight: float = 0.0
    stages: list[str] = Field(
        default_factory=lambda: ["sft", "router_hygiene", "preference"]
    )
    router_hygiene_steps: int = 200
    output_dir: str = "artifacts/runs"
    save_steps: int = 200
    logging_steps: int = 10
    seed: int = 42
    gradient_checkpointing: bool = True
    # How ESFT is applied across expert sectors:
    #   joint      — one ESFT pass over all selected experts (legacy default)
    #   sequential — forensics → sector dataset → ESFT per sector (recommended)
    sector_mode: Literal["joint", "sequential"] = "sequential"
    # Minimum train samples per sector after match+synthesize (sequential)
    sector_min_samples: int = 8
    # Fraction of unmatched/global pool mixed into each sector shard
    sector_shared_fraction: float = 0.15
    # If a sector is blocked by forensics gate, continue with the rest
    sector_continue_on_block: bool = True
    # Pre/post sector probe keep/rollback
    sector_probe_enabled: bool = True
    sector_probe_min_delta: float = -0.02
    sector_probe_dry_boost: float = 0.08
    sector_keep_rollback: bool = True
    # Data contracts (per-sector shards)
    # block = refuse train; warn = train with hard-warn (default so pure-synth dry-runs still exercise ESFT)
    sector_contract_mode: Literal["block", "warn", "off"] = "warn"
    sector_min_real_fraction: float = 0.15
    sector_max_synth_fraction: float = 0.85
    sector_min_unique_ratio: float = 0.35


class ScorecardThresholds(BaseModel):
    domain_score: float = 0.75
    general_delta_max: float = -0.05  # max allowed drop vs baseline
    routing_entropy_min: float = 1.2
    load_balance_cv_max: float = 1.5  # coefficient of variation
    hallucination_max: float = 0.12
    safety_pass: bool = True
    # Optional deeper domain-competence axis (any industry — not field-specific)
    domain_depth_min: Optional[float] = None
    # High-stakes industries: require human promote (finance, health, legal, etc.)
    require_human_approval: bool = False
    high_stakes: bool = False
    # Optional pack-eval gate (None = record score, do not fail)
    pack_eval_min: Optional[float] = None


class EvalConfig(BaseModel):
    scorecard_thresholds: ScorecardThresholds = Field(default_factory=ScorecardThresholds)
    domain_benchmarks: list[str] = Field(default_factory=lambda: ["domain_heldout"])
    general_benchmarks: list[str] = Field(default_factory=lambda: ["mmlu_sample", "hellaswag_sample"])
    num_eval_samples: int = 256
    stress_test: bool = True
    auto_rollback: bool = True


class ContinuousConfig(BaseModel):
    enabled: bool = False
    update_frequency: str = "daily"
    federated: bool = False
    error_cluster_min: int = 20
    max_experts_per_update: int = 8


class LifecycleConfig(BaseModel):
    enabled: bool = True
    util_low_threshold: float = 0.15  # relative to mean
    util_high_threshold: float = 3.0
    allow_mitosis: bool = True
    allow_rebirth: bool = True
    allow_prune: bool = False  # dangerous; opt-in


class HiveConfig(BaseModel):
    specialists: list[str] = Field(default_factory=list)
    consult_protocol: Literal["round_robin", "router", "debate"] = "debate"
    max_consult_rounds: int = 3
    distill_after_consult: bool = True


class RunConfig(BaseModel):
    name: str = "aetherforge-run"
    run_id: Optional[str] = None
    project: str = "aetherforge"
    tracking: Literal["none", "wandb", "mlflow", "json"] = "json"
    dry_run: bool = False
    resume_from: Optional[str] = None
    device: str = "cuda"
    num_workers: int = 2


class ProvidersConfig(BaseModel):
    """
    Remote connections.

    Actual host keys live in ~/.aetherforge/ (via `aetherforge connect`).
    This block only sets defaults for which active provider to prefer
    and optional teacher-model usage during data synthesis.
    """

    # Prefer named connection from ~/.aetherforge/connections.yaml
    compute_profile: Optional[str] = None
    llm_profile: Optional[str] = None
    # Use connected LLM as synthetic teacher / THD when available
    use_llm_for_synthetic: bool = False
    use_llm_for_thd: bool = False
    # Remote train defaults
    remote_dir: str = "/workspace/aetherforge"


class AetherForgeConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    affinity: AffinityConfig = Field(default_factory=AffinityConfig)
    groups: GroupsConfig = Field(default_factory=GroupsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    continuous: ContinuousConfig = Field(default_factory=ContinuousConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    hive: HiveConfig = Field(default_factory=HiveConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @field_validator("data")
    @classmethod
    def _domain_slug(cls, v: DataConfig) -> DataConfig:
        v.domain = v.domain.strip().lower().replace(" ", "_")
        return v


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def merge_configs(*paths: str | Path, overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, raw)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged


def apply_posture_defaults(cfg: AetherForgeConfig) -> AetherForgeConfig:
    """
    Expand specialist → broad → wide knobs when the user only set training.posture.

    Explicit YAML values always win (we only fill when still at specialist defaults
    for the fields posture is meant to own).
    """
    posture = (cfg.training.posture or "specialist").lower()
    if posture == "specialist":
        return cfg

    # --- affinity / selection ---
    if posture == "broad":
        if cfg.affinity.top_k_experts <= 16 and cfg.affinity.top_k_fraction is None:
            cfg.affinity.top_k_fraction = 0.28  # ~28% of expert slots
            cfg.affinity.top_k_experts = 0  # fraction wins in ranker when set
        cfg.affinity.freeze_low_affinity = True
        cfg.affinity.freeze_router_initially = True
        cfg.affinity.progressive_unfreeze = True
        cfg.affinity.multi_theme_probes = True
        cfg.affinity.probe_size = max(cfg.affinity.probe_size, 1024)
        if cfg.groups.train_scope == "selected":
            cfg.groups.train_scope = "top_n"
            cfg.groups.train_top_n = max(cfg.groups.train_top_n, 6)
        cfg.groups.target_num_groups = max(cfg.groups.target_num_groups, 12)
        cfg.training.specialization_loss_weight = min(
            cfg.training.specialization_loss_weight, 0.05
        )
        cfg.training.load_balance_loss_weight = max(
            cfg.training.load_balance_loss_weight, 0.02
        )
        if cfg.training.max_steps is not None and cfg.training.max_steps < 400:
            cfg.training.max_steps = 800
        cfg.training.include_attention = True
        cfg.training.mask_unselected_experts = True
        cfg.training.router_hygiene_steps = max(cfg.training.router_hygiene_steps, 300)
    elif posture == "wide":
        cfg.affinity.top_k_fraction = 1.0
        cfg.affinity.top_k_experts = 0
        cfg.affinity.freeze_low_affinity = False
        cfg.affinity.freeze_router_initially = False
        cfg.affinity.progressive_unfreeze = True
        cfg.affinity.multi_theme_probes = True
        cfg.affinity.probe_size = max(cfg.affinity.probe_size, 1536)
        cfg.groups.train_scope = "all"
        cfg.groups.target_num_groups = max(cfg.groups.target_num_groups, 12)
        cfg.training.specialization_loss_weight = 0.02
        cfg.training.load_balance_loss_weight = max(
            cfg.training.load_balance_loss_weight, 0.03
        )
        if cfg.training.max_steps is not None and cfg.training.max_steps < 800:
            cfg.training.max_steps = 1500
        cfg.training.include_attention = True
        # wide LoRA = update all expert slots in fused tensors
        cfg.training.mask_unselected_experts = False
        cfg.training.router_hygiene_steps = max(cfg.training.router_hygiene_steps, 400)
        cfg.training.learning_rate = min(cfg.training.learning_rate, 8e-5)

    return cfg


def load_config(
    *paths: str | Path,
    overrides: Optional[dict[str, Any]] = None,
    validate: bool = True,
) -> AetherForgeConfig | dict[str, Any]:
    """Load one or more YAML configs (later files override earlier)."""
    raw = merge_configs(*paths, overrides=overrides) if paths else (overrides or {})
    # Resolve OmegaConf interpolations if present
    try:
        conf = OmegaConf.create(raw)
        raw = OmegaConf.to_container(conf, resolve=True)  # type: ignore[assignment]
    except Exception:
        pass
    if not validate:
        return raw  # type: ignore[return-value]
    cfg = AetherForgeConfig.model_validate(raw)
    return apply_posture_defaults(cfg)


def dump_config(cfg: AetherForgeConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(), f, sort_keys=False, default_flow_style=False)
