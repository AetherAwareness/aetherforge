"""
Vast.ai compute connection.

Two modes:
  1. **Connect to a rented instance** (primary): paste SSH host/port from Vast dashboard
     or instance id + fetch SSH via API when VAST_API_KEY is set.
  2. **List instances** (optional API): helps pick an already-running box.

We do not force-rent GPUs from the UI (billing-sensitive); we make connecting
an existing rental one click, and document the offer ID → rent flow.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from aetherforge.providers.base import HealthReport
from aetherforge.providers.compute.ssh_box import SSHComputeProvider
from aetherforge.providers.credentials import get_secret, mask_secret
from aetherforge.utils.logging import get_logger

log = get_logger("providers.compute.vast")

VAST_API = "https://console.vast.ai/api/v0"


class VastComputeProvider(SSHComputeProvider):
    name = "vast"

    def __init__(
        self,
        host: str = "",
        port: int = 22,
        user: str = "root",
        identity_file: Optional[str] = None,
        remote_dir: str = "/workspace/aetherforge",
        instance_id: Optional[str] = None,
        label: str = "vast",
        extra: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            host=host,
            port=port,
            user=user,
            identity_file=identity_file,
            remote_dir=remote_dir,
            label=label,
            extra=extra or {},
        )
        self.instance_id = instance_id
        self.extra["instance_id"] = instance_id

    @staticmethod
    def api_key() -> Optional[str]:
        return get_secret(
            "vast",
            "api_key",
            env_names=["VAST_API_KEY", "VASTAI_API_KEY", "VAST_APIKEY"],
        )

    def _api_get(self, path: str) -> Any:
        key = self.api_key()
        if not key:
            raise RuntimeError("Vast API key not set (VAST_API_KEY or aetherforge connect key vast)")
        url = VAST_API.rstrip("/") + "/" + path.lstrip("/")
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_instances(self) -> list[dict[str, Any]]:
        """Best-effort list of current instances (API shape may vary)."""
        try:
            data = self._api_get("instances/")
        except Exception as e:
            log.warning("Vast list instances failed: %s", e)
            return []
        # Normalize common shapes
        rows = data if isinstance(data, list) else data.get("instances") or data.get("results") or []
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "id": row.get("id") or row.get("instance_id"),
                    "status": row.get("actual_status") or row.get("status") or row.get("cur_state"),
                    "gpu": row.get("gpu_name") or row.get("gpu"),
                    "ssh_host": row.get("ssh_host") or row.get("public_ipaddr"),
                    "ssh_port": row.get("ssh_port") or row.get("ports", {}).get("22/tcp"),
                    "label": row.get("label") or row.get("machine_id"),
                    "raw_keys": list(row.keys())[:20],
                }
            )
        return out

    def health(self) -> HealthReport:
        # Prefer SSH if configured
        if self.host:
            report = super().health()
            report.provider = "vast"
            report.details["instance_id"] = self.instance_id
            report.details["api_key"] = mask_secret(self.api_key())
            return report
        # API-only check
        key = self.api_key()
        if not key:
            return HealthReport(
                ok=False,
                provider="vast",
                kind="compute",
                message="Set VAST_API_KEY or connect with SSH host:port from Vast dashboard",
                details={"api_key": "(not set)", "hint": "aetherforge connect vast --host … --port …"},
            )
        try:
            instances = self.list_instances()
            return HealthReport(
                ok=True,
                provider="vast",
                kind="compute",
                message=f"Vast API OK · {len(instances)} instance(s) visible",
                details={
                    "api_key": mask_secret(key),
                    "instances": instances[:10],
                    "hint": "Pick an instance and: aetherforge connect vast --host IP --port PORT",
                },
            )
        except Exception as e:
            return HealthReport(
                ok=False,
                provider="vast",
                kind="compute",
                message=f"Vast API error: {e}",
                details={"api_key": mask_secret(key)},
            )

    def connection_info(self) -> dict[str, Any]:
        info = super().connection_info()
        info["provider"] = "vast"
        info["instance_id"] = self.instance_id
        info["api_key_set"] = bool(self.api_key())
        return info
