"""
Credential store for remote providers.

Secrets live in:
  1. Environment variables (preferred for CI / servers)
  2. ~/.aetherforge/credentials.yaml  (chmod 600) — never committed

Never log raw keys. Never put keys in run artifacts.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, Optional

import yaml

from aetherforge.utils.logging import get_logger

log = get_logger("providers.credentials")

def _resolve_home() -> Path:
    """Prefer AETHERFORGE_HOME, then legacy HIVEFORGE_HOME / ~/.hiveforge."""
    env = os.environ.get("AETHERFORGE_HOME") or os.environ.get("HIVEFORGE_HOME")
    if env:
        return Path(env)
    modern = Path.home() / ".aetherforge"
    legacy = Path.home() / ".hiveforge"
    if modern.exists() or not legacy.exists():
        return modern
    return legacy


AETHERFORGE_HOME = _resolve_home()
CREDS_PATH = AETHERFORGE_HOME / "credentials.yaml"
CONNECTIONS_PATH = AETHERFORGE_HOME / "connections.yaml"


def ensure_home() -> Path:
    AETHERFORGE_HOME.mkdir(parents=True, exist_ok=True)
    return AETHERFORGE_HOME


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
        return {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    ensure_home()
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
    except OSError:
        pass


def set_secret(provider: str, key: str, value: str) -> None:
    data = _load_yaml(CREDS_PATH)
    data.setdefault(provider, {})[key] = value
    _save_yaml(CREDS_PATH, data)
    log.info("Saved credential %s.%s → %s", provider, key, CREDS_PATH)


def get_secret(
    provider: str,
    key: str,
    *,
    env_names: Optional[list[str]] = None,
    default: Optional[str] = None,
) -> Optional[str]:
    """Resolve secret: env first, then credentials file."""
    for env in env_names or []:
        v = os.environ.get(env, "").strip()
        if v:
            return v
    data = _load_yaml(CREDS_PATH)
    v = (data.get(provider) or {}).get(key)
    if v:
        return str(v).strip()
    return default


def mask_secret(value: Optional[str]) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return "****"
    return value[:4] + "…" + value[-4:]


def has_secret(provider: str, key: str, env_names: Optional[list[str]] = None) -> bool:
    return bool(get_secret(provider, key, env_names=env_names))


# ── Connection profiles (non-secret host/instance config) ────────────


def load_connections() -> dict[str, Any]:
    return _load_yaml(CONNECTIONS_PATH)


def save_connections(data: dict[str, Any]) -> None:
    _save_yaml(CONNECTIONS_PATH, data)


def upsert_connection(kind: str, name: str, profile: dict[str, Any]) -> dict[str, Any]:
    """kind: compute | llm"""
    data = load_connections()
    data.setdefault(kind, {})
    data[kind][name] = profile
    # mark active
    data.setdefault("active", {})
    data["active"][kind] = name
    save_connections(data)
    return profile


def get_active_connection(kind: str) -> Optional[dict[str, Any]]:
    data = load_connections()
    name = (data.get("active") or {}).get(kind)
    if not name:
        return None
    prof = (data.get(kind) or {}).get(name)
    if not prof:
        return None
    return {"name": name, **prof}


def set_active(kind: str, name: str) -> None:
    data = load_connections()
    if name not in (data.get(kind) or {}):
        raise KeyError(f"No {kind} connection named {name!r}")
    data.setdefault("active", {})[kind] = name
    save_connections(data)


def list_connections() -> dict[str, Any]:
    data = load_connections()
    return {
        "active": data.get("active") or {},
        "compute": list((data.get("compute") or {}).keys()),
        "llm": list((data.get("llm") or {}).keys()),
        "compute_profiles": data.get("compute") or {},
        "llm_profiles": data.get("llm") or {},
        "creds_path": str(CREDS_PATH),
        "connections_path": str(CONNECTIONS_PATH),
    }
