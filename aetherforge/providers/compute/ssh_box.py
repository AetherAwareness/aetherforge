"""Generic SSH GPU box — also the transport for Vast / RunPod once you have host:port."""

from __future__ import annotations

import shlex
import subprocess
from typing import Any, Optional

from aetherforge.providers.base import ComputeProvider, HealthReport
from aetherforge.utils.logging import get_logger

log = get_logger("providers.compute.ssh")


class SSHComputeProvider(ComputeProvider):
    name = "ssh"

    def __init__(
        self,
        host: str,
        port: int = 22,
        user: str = "root",
        identity_file: Optional[str] = None,
        remote_dir: str = "/workspace/aetherforge",
        label: str = "ssh",
        extra: Optional[dict[str, Any]] = None,
    ):
        self.host = host
        self.port = int(port)
        self.user = user
        self.identity_file = identity_file
        self.remote_dir = remote_dir
        self.label = label
        self.extra = extra or {}

    def _ssh_base(self) -> list[str]:
        cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=8",
            "-p",
            str(self.port),
        ]
        if self.identity_file:
            cmd += ["-i", self.identity_file]
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    def health(self) -> HealthReport:
        if not self.host:
            return HealthReport(
                ok=False,
                provider=self.name,
                kind="compute",
                message="No SSH host configured",
            )
        try:
            r = subprocess.run(
                self._ssh_base() + ["echo", "aetherforge-ok && nvidia-smi -L 2>/dev/null | head -3"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            out = (r.stdout or "") + (r.stderr or "")
            ok = r.returncode == 0 and "aetherforge-ok" in out
            return HealthReport(
                ok=ok,
                provider=self.name,
                kind="compute",
                message="SSH reachable" if ok else f"SSH failed (rc={r.returncode})",
                details={
                    "host": self.host,
                    "port": self.port,
                    "user": self.user,
                    "output_head": out[:500],
                },
            )
        except Exception as e:
            return HealthReport(
                ok=False,
                provider=self.name,
                kind="compute",
                message=str(e),
                details={"host": self.host, "port": self.port},
            )

    def connection_info(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "label": self.label,
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "remote_dir": self.remote_dir,
            "identity_file": self.identity_file,
            "extra": self.extra,
        }

    def sync_command(self, local_dir: str, remote_dir: Optional[str] = None) -> str:
        rd = remote_dir or self.remote_dir
        ssh_opt = f"-e 'ssh -p {self.port}"
        if self.identity_file:
            ssh_opt += f" -i {shlex.quote(self.identity_file)}"
        ssh_opt += "'"
        # exclude heavy local junk
        excludes = [
            "--exclude", ".venv",
            "--exclude", "artifacts",
            "--exclude", ".git",
            "--exclude", "__pycache__",
            "--exclude", "*.pyc",
            "--exclude", "dist",
            "--exclude", "build",
        ]
        excl = " ".join(excludes)
        return (
            f"rsync -avz {ssh_opt} {excl} "
            f"{shlex.quote(local_dir.rstrip('/'))}/ "
            f"{self.user}@{self.host}:{shlex.quote(rd)}/"
        )

    def remote_train_command(self, train_args: str, *, background: bool = False) -> str:
        rd = shlex.quote(self.remote_dir)
        # train_args already a shell fragment after `aetherforge train`
        setup = (
            f"cd {rd} && "
            f"(test -d .venv && . .venv/bin/activate || "
            f"(python3 -m venv .venv && . .venv/bin/activate && pip install -q -U pip)) && "
            f"pip install -q -e '.[dev]' 2>/dev/null || pip install -q -e . 2>/dev/null || true && "
            f"mkdir -p artifacts/runs"
        )
        train = f"aetherforge train {train_args}"
        if not background:
            return f"{setup} && {train}"
        # Detached: survive SSH disconnect; logs under artifacts/
        return (
            f"{setup} && "
            f"nohup env PYTHONUNBUFFERED=1 {train} "
            f"> artifacts/remote_train.nohup.log 2>&1 & echo $! > artifacts/remote_train.pid && "
            f"echo STARTED_PID=$(cat artifacts/remote_train.pid) && "
            f"sleep 1 && tail -n 20 artifacts/remote_train.nohup.log || true"
        )

    def run_remote(self, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._ssh_base() + [command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
