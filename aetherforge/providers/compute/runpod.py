"""RunPod connection — SSH profile + optional API key for instance list."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Optional

from aetherforge.providers.base import HealthReport
from aetherforge.providers.compute.ssh_box import SSHComputeProvider
from aetherforge.providers.credentials import get_secret, mask_secret
from aetherforge.utils.logging import get_logger

log = get_logger("providers.compute.runpod")


class RunPodComputeProvider(SSHComputeProvider):
    name = "runpod"

    def __init__(self, pod_id: Optional[str] = None, **kwargs: Any):
        kwargs.setdefault("label", "runpod")
        kwargs.setdefault("remote_dir", "/workspace/aetherforge")
        extra = dict(kwargs.get("extra") or {})
        if pod_id:
            extra["pod_id"] = pod_id
        kwargs["extra"] = extra
        super().__init__(**kwargs)
        self.pod_id = pod_id or extra.get("pod_id")

    @staticmethod
    def api_key() -> Optional[str]:
        return get_secret(
            "runpod",
            "api_key",
            env_names=["RUNPOD_API_KEY", "RUNPOD_APIKEY"],
        )

    def health(self) -> HealthReport:
        if self.host:
            report = super().health()
            report.provider = "runpod"
            report.details["pod_id"] = self.pod_id
            report.details["api_key"] = mask_secret(self.api_key())
            return report
        key = self.api_key()
        if not key:
            return HealthReport(
                ok=False,
                provider="runpod",
                kind="compute",
                message="Set RUNPOD_API_KEY or connect with SSH host from RunPod UI",
            )
        # lightweight: key present
        return HealthReport(
            ok=True,
            provider="runpod",
            kind="compute",
            message="RunPod API key present (connect SSH host for training)",
            details={"api_key": mask_secret(key)},
        )

    def connection_info(self) -> dict[str, Any]:
        info = super().connection_info()
        info["provider"] = "runpod"
        info["pod_id"] = self.pod_id
        info["api_key_set"] = bool(self.api_key())
        return info
