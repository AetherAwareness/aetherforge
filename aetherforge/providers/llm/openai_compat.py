"""Generic OpenAI-compatible LLM provider (OpenRouter, Together, Fireworks, Groq, …)."""

from __future__ import annotations

from typing import Any, Optional

from aetherforge.providers.base import HealthReport, LLMProvider
from aetherforge.providers.credentials import get_secret, mask_secret
from aetherforge.utils.llm_client import LLMConfig, OpenAICompatClient
from aetherforge.utils.logging import get_logger

log = get_logger("providers.llm.openai_compat")

# Known presets — users can still pass custom base_url
PRESETS: dict[str, dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openrouter/auto",
        "key_env": "OPENROUTER_API_KEY",
        "label": "OpenRouter",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
        "label": "OpenAI",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "key_env": "TOGETHER_API_KEY",
        "label": "Together AI",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "default_model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "key_env": "FIREWORKS_API_KEY",
        "label": "Fireworks",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "label": "Groq",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
        "label": "DeepSeek API",
    },
    "custom": {
        "base_url": "http://127.0.0.1:8095/v1",
        "default_model": "local",
        "key_env": "AETHERFORGE_LLM_KEY",
        "label": "Custom OpenAI-compatible",
    },
}


class OpenAICompatLLMProvider(LLMProvider):
    def __init__(
        self,
        provider_id: str = "openrouter",
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        timeout_sec: float = 120.0,
    ):
        preset = PRESETS.get(provider_id, PRESETS["custom"])
        self.provider_id = provider_id if provider_id in PRESETS else "custom"
        self.name = self.provider_id
        env_name = api_key_env or preset["key_env"]
        key = api_key or get_secret(
            self.provider_id,
            "api_key",
            env_names=[env_name, "AETHERFORGE_LLM_KEY"],
        )
        self.cfg = LLMConfig(
            base_url=(base_url or preset["base_url"]).rstrip("/"),
            api_key=key or "missing",
            model=model or preset["default_model"],
            timeout_sec=timeout_sec,
        )
        self._client = OpenAICompatClient(self.cfg)
        self._env_name = env_name
        self._label = preset.get("label", provider_id)

    def health(self) -> HealthReport:
        key_ok = self.cfg.api_key and self.cfg.api_key != "missing"
        if not key_ok and self.provider_id != "custom":
            return HealthReport(
                ok=False,
                provider=self.name,
                kind="llm",
                message=f"API key not set ({self._env_name})",
                details={
                    "base_url": self.cfg.base_url,
                    "model": self.cfg.model,
                    "api_key": mask_secret(None),
                },
            )
        try:
            ok = self._client.available()
            # Some providers block /models — try a tiny chat as fallback signal
            if not ok and key_ok:
                try:
                    self.chat("ping", "Reply with OK only.")
                    ok = True
                except Exception:
                    ok = False
            return HealthReport(
                ok=ok,
                provider=self.name,
                kind="llm",
                message=f"{self._label} reachable" if ok else f"{self._label} not reachable",
                details={
                    "base_url": self.cfg.base_url,
                    "model": self.cfg.model,
                    "api_key": mask_secret(self.cfg.api_key if key_ok else None),
                    "label": self._label,
                },
            )
        except Exception as e:
            return HealthReport(
                ok=False,
                provider=self.name,
                kind="llm",
                message=str(e),
                details={"base_url": self.cfg.base_url, "model": self.cfg.model},
            )

    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        return self._client.chat(system, user, **kwargs)

    def connection_info(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "label": self._label,
            "base_url": self.cfg.base_url,
            "model": self.cfg.model,
            "api_key_set": bool(self.cfg.api_key and self.cfg.api_key != "missing"),
            "api_key_env": self._env_name,
        }
