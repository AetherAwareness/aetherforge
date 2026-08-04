"""
OpenAI-compatible chat client for THD / hive consult.

Works with:
  - local llama-server / Hermes OpenAI API (:8095, :8642, …)
  - vLLM OpenAI server
  - any cloud OpenAI-compatible endpoint
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from aetherforge.utils.logging import get_logger

log = get_logger("utils.llm_client")


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:8095/v1"
    api_key: str = "local"
    model: str = "trinity-active"
    timeout_sec: float = 120.0
    temperature: float = 0.4
    max_tokens: int = 1024


class OpenAICompatClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig(
            base_url=os.environ.get("AETHERFORGE_LLM_BASE", "http://127.0.0.1:8095/v1"),
            api_key=os.environ.get("AETHERFORGE_LLM_KEY", "local"),
            model=os.environ.get("AETHERFORGE_LLM_MODEL", "trinity-active"),
        )

    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        cfg = self.config
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": kwargs.get("model", cfg.model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": kwargs.get("temperature", cfg.temperature),
            "max_tokens": kwargs.get("max_tokens", cfg.max_tokens),
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            log.error("LLM HTTP %s: %s", e.code, err[:300])
            raise
        except Exception as e:
            log.error("LLM request failed: %s", e)
            raise

    def available(self) -> bool:
        try:
            url = self.config.base_url.rstrip("/") + "/models"
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


def make_llm_fn(config: Optional[LLMConfig] = None):
    """Return (system, user) -> str callable for HiveOrchestrator / THD."""
    client = OpenAICompatClient(config)

    def _fn(system: str, user: str) -> str:
        return client.chat(system, user)

    return _fn


def make_llm_fn_from_providers():
    """
    Prefer active OpenRouter/Together/… connection from `aetherforge connect`.
    Falls back to AETHERFORGE_LLM_* env / local :8095.
    """
    try:
        from aetherforge.providers.registry import get_active_llm

        llm = get_active_llm()
        if llm is not None:
            return llm.chat
    except Exception as e:
        log.debug("No active LLM provider: %s", e)
    return make_llm_fn()
