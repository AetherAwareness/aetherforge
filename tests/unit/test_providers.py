from pathlib import Path

import yaml

from aetherforge.providers.credentials import (
    set_secret,
    get_secret,
    mask_secret,
    upsert_connection,
    get_active_connection,
    list_connections,
)
from aetherforge.providers import connect as conn
from aetherforge.providers.llm.openai_compat import PRESETS, OpenAICompatLLMProvider
from aetherforge.providers.compute.ssh_box import SSHComputeProvider
from aetherforge.providers.remote_train import build_remote_bundle


def test_mask_secret():
    assert mask_secret(None) == "(not set)"
    assert "…" in mask_secret("sk-abcdefghijklmnop")


def test_credentials_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHERFORGE_HOME", str(tmp_path))
    # reload paths — credentials module caches Path at import; patch attributes
    import aetherforge.providers.credentials as creds

    monkeypatch.setattr(creds, "AETHERFORGE_HOME", tmp_path)
    monkeypatch.setattr(creds, "CREDS_PATH", tmp_path / "credentials.yaml")
    monkeypatch.setattr(creds, "CONNECTIONS_PATH", tmp_path / "connections.yaml")

    set_secret("openrouter", "api_key", "sk-test-1234567890")
    assert get_secret("openrouter", "api_key") == "sk-test-1234567890"
    # env wins
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-priority")
    assert (
        get_secret("openrouter", "api_key", env_names=["OPENROUTER_API_KEY"])
        == "sk-env-priority"
    )


def test_upsert_compute_connection(tmp_path, monkeypatch):
    import aetherforge.providers.credentials as creds

    monkeypatch.setattr(creds, "AETHERFORGE_HOME", tmp_path)
    monkeypatch.setattr(creds, "CREDS_PATH", tmp_path / "credentials.yaml")
    monkeypatch.setattr(creds, "CONNECTIONS_PATH", tmp_path / "connections.yaml")

    upsert_connection(
        "compute",
        "vast",
        {"provider": "vast", "host": "1.2.3.4", "port": 22, "user": "root"},
    )
    active = get_active_connection("compute")
    assert active is not None
    assert active["host"] == "1.2.3.4"
    assert "vast" in list_connections()["compute"]


def test_connect_llm_saves_profile(tmp_path, monkeypatch):
    import aetherforge.providers.credentials as creds

    monkeypatch.setattr(creds, "AETHERFORGE_HOME", tmp_path)
    monkeypatch.setattr(creds, "CREDS_PATH", tmp_path / "credentials.yaml")
    monkeypatch.setattr(creds, "CONNECTIONS_PATH", tmp_path / "connections.yaml")

    r = conn.connect_llm("openrouter", model="test/model", api_key="sk-abc", test=False)
    assert r["ok"] is True
    assert r["profile"]["model"] == "test/model"
    assert get_secret("openrouter", "api_key") == "sk-abc"


def test_ssh_sync_command():
    p = SSHComputeProvider(host="10.0.0.1", port=2222, user="root")
    cmd = p.sync_command("/tmp/aetherforge")
    assert "rsync" in cmd
    assert "10.0.0.1" in cmd
    assert "2222" in cmd


def test_llm_presets_complete():
    for name in ("openrouter", "openai", "together", "fireworks", "groq", "deepseek"):
        assert name in PRESETS
        assert "base_url" in PRESETS[name]


def test_remote_bundle_without_connection(tmp_path, monkeypatch):
    import aetherforge.providers.credentials as creds
    import aetherforge.providers.registry as reg

    monkeypatch.setattr(creds, "AETHERFORGE_HOME", tmp_path)
    monkeypatch.setattr(creds, "CONNECTIONS_PATH", tmp_path / "connections.yaml")
    monkeypatch.setattr(reg, "get_active_compute", lambda: None)
    b = build_remote_bundle(config_paths=["configs/base.yaml"])
    assert b["ok"] is False


def test_catalog():
    cat = conn.catalog()
    assert "compute" in cat
    assert "llm" in cat
    assert any(p["id"] == "vast" for p in cat["compute"])
    assert any(p["id"] == "openrouter" for p in cat["llm"])


def test_pull_without_connection(tmp_path, monkeypatch):
    import aetherforge.providers.registry as reg
    from aetherforge.providers.remote_train import exec_pull_artifacts, tail_remote_logs

    monkeypatch.setattr(reg, "get_active_compute", lambda: None)
    assert exec_pull_artifacts().get("ok") is False
    assert tail_remote_logs().get("ok") is False
