"""Build provider instances from saved connections + env."""

from __future__ import annotations

from typing import Any, Optional

from aetherforge.providers.base import ComputeProvider, HealthReport, LLMProvider
from aetherforge.providers.compute.runpod import RunPodComputeProvider
from aetherforge.providers.compute.ssh_box import SSHComputeProvider
from aetherforge.providers.compute.vast import VastComputeProvider
from aetherforge.providers.credentials import get_active_connection, list_connections
from aetherforge.providers.llm.openai_compat import PRESETS, OpenAICompatLLMProvider
from aetherforge.utils.logging import get_logger

log = get_logger("providers.registry")


def list_providers() -> dict[str, Any]:
    return {
        "compute": [
            {
                "id": "vast",
                "label": "Vast.ai",
                "kind": "compute",
                "description": "Rent GPUs; connect via SSH host:port from dashboard or API",
            },
            {
                "id": "runpod",
                "label": "RunPod",
                "kind": "compute",
                "description": "GPU pods; connect via SSH",
            },
            {
                "id": "ssh",
                "label": "Generic SSH",
                "kind": "compute",
                "description": "Any GPU box (Lambda, CoreWeave, home server, …)",
            },
        ],
        "llm": [
            {"id": k, "label": v["label"], "kind": "llm", "base_url": v["base_url"]}
            for k, v in PRESETS.items()
        ],
        "connections": list_connections(),
    }


def build_compute_from_profile(profile: dict[str, Any]) -> ComputeProvider:
    provider = (profile.get("provider") or profile.get("type") or "ssh").lower()
    common = {
        "host": profile.get("host") or "",
        "port": int(profile.get("port") or 22),
        "user": profile.get("user") or "root",
        "identity_file": profile.get("identity_file"),
        "remote_dir": profile.get("remote_dir") or "/workspace/aetherforge",
        "label": profile.get("name") or provider,
        "extra": profile.get("extra") or {},
    }
    if provider == "vast":
        return VastComputeProvider(
            **common,
            instance_id=profile.get("instance_id"),
        )
    if provider == "runpod":
        return RunPodComputeProvider(pod_id=profile.get("pod_id"), **common)
    return SSHComputeProvider(**common)


def build_llm_from_profile(profile: dict[str, Any]) -> LLMProvider:
    provider = (profile.get("provider") or profile.get("type") or "openrouter").lower()
    return OpenAICompatLLMProvider(
        provider_id=provider,
        base_url=profile.get("base_url"),
        model=profile.get("model"),
        api_key=profile.get("api_key"),  # discouraged; prefer env/store
        api_key_env=profile.get("api_key_env"),
        timeout_sec=float(profile.get("timeout_sec") or 120),
    )


def get_active_compute() -> Optional[ComputeProvider]:
    prof = get_active_connection("compute")
    if not prof:
        # fall back to vast API-only if key present
        vast = VastComputeProvider()
        if vast.api_key():
            return vast
        return None
    return build_compute_from_profile(prof)


def get_active_llm() -> Optional[LLMProvider]:
    prof = get_active_connection("llm")
    if not prof:
        # env-based openrouter fallback
        llm = OpenAICompatLLMProvider("openrouter")
        if llm.cfg.api_key and llm.cfg.api_key != "missing":
            return llm
        return None
    return build_llm_from_profile(prof)


def provider_status() -> dict[str, Any]:
    out: dict[str, Any] = {
        "compute": None,
        "llm": None,
        "connections": list_connections(),
        "catalog": list_providers(),
    }
    try:
        comp = get_active_compute()
        if comp:
            out["compute"] = comp.health().to_dict()
            out["compute"]["info"] = comp.connection_info()
    except Exception as e:
        out["compute"] = HealthReport(False, "compute", "compute", str(e)).to_dict()
    try:
        llm = get_active_llm()
        if llm:
            out["llm"] = llm.health().to_dict()
            out["llm"]["info"] = llm.connection_info()
    except Exception as e:
        out["llm"] = HealthReport(False, "llm", "llm", str(e)).to_dict()
    return out
