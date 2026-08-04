"""High-level connect helpers used by CLI and dashboard API."""

from __future__ import annotations

from typing import Any, Optional

from aetherforge.providers.credentials import (
    get_secret,
    list_connections,
    mask_secret,
    set_active,
    set_secret,
    upsert_connection,
)
from aetherforge.providers.llm.openai_compat import PRESETS
from aetherforge.providers.registry import (
    build_compute_from_profile,
    build_llm_from_profile,
    list_providers,
    provider_status,
)
from aetherforge.utils.logging import get_logger

log = get_logger("providers.connect")


def save_api_key(provider: str, api_key: str) -> dict[str, Any]:
    set_secret(provider, "api_key", api_key.strip())
    return {
        "ok": True,
        "provider": provider,
        "api_key": mask_secret(api_key),
        "message": f"Saved {provider} API key to ~/.aetherforge/credentials.yaml",
    }


def connect_compute(
    provider: str,
    *,
    name: Optional[str] = None,
    host: str = "",
    port: int = 22,
    user: str = "root",
    identity_file: Optional[str] = None,
    remote_dir: str = "/workspace/aetherforge",
    instance_id: Optional[str] = None,
    pod_id: Optional[str] = None,
    test: bool = True,
) -> dict[str, Any]:
    provider = provider.lower().strip()
    if provider not in ("vast", "runpod", "ssh"):
        return {"ok": False, "error": f"Unknown compute provider: {provider}"}
    profile_name = name or provider
    profile: dict[str, Any] = {
        "provider": provider,
        "host": host,
        "port": int(port),
        "user": user,
        "identity_file": identity_file,
        "remote_dir": remote_dir,
        "instance_id": instance_id,
        "pod_id": pod_id,
    }
    upsert_connection("compute", profile_name, profile)
    result: dict[str, Any] = {
        "ok": True,
        "name": profile_name,
        "profile": {k: v for k, v in profile.items() if k != "identity_file" or v},
    }
    if test:
        comp = build_compute_from_profile({"name": profile_name, **profile})
        health = comp.health()
        result["health"] = health.to_dict()
        result["ok"] = health.ok or bool(host is None)  # API-only vast may be ok without host
        # For explicit host, require SSH ok
        if host:
            result["ok"] = health.ok
    return result


def connect_llm(
    provider: str,
    *,
    name: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    test: bool = True,
) -> dict[str, Any]:
    provider = provider.lower().strip()
    if provider not in PRESETS:
        return {
            "ok": False,
            "error": f"Unknown LLM provider: {provider}",
            "known": list(PRESETS.keys()),
        }
    if api_key:
        set_secret(provider, "api_key", api_key.strip())
    preset = PRESETS[provider]
    profile_name = name or provider
    profile = {
        "provider": provider,
        "model": model or preset["default_model"],
        "base_url": base_url or preset["base_url"],
        "api_key_env": preset["key_env"],
    }
    upsert_connection("llm", profile_name, profile)
    result: dict[str, Any] = {"ok": True, "name": profile_name, "profile": profile}
    if test:
        llm = build_llm_from_profile({"name": profile_name, **profile})
        health = llm.health()
        result["health"] = health.to_dict()
        # don't hard-fail connect if network blocked — still saved
        result["saved"] = True
    return result


def status() -> dict[str, Any]:
    return provider_status()


def catalog() -> dict[str, Any]:
    return list_providers()
