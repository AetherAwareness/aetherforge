"""Provider interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class HealthReport:
    ok: bool
    provider: str
    kind: str  # compute | llm
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "provider": self.provider,
            "kind": self.kind,
            "message": self.message,
            "details": self.details,
        }


class ComputeProvider(ABC):
    """Remote GPU box used for training (Vast, RunPod, generic SSH, …)."""

    name: str = "compute"

    @abstractmethod
    def health(self) -> HealthReport: ...

    @abstractmethod
    def connection_info(self) -> dict[str, Any]: ...

    def sync_command(self, local_dir: str, remote_dir: Optional[str] = None) -> str:
        """Shell command to rsync project to remote (for user to run or we execute)."""
        raise NotImplementedError

    def remote_train_command(self, train_args: str) -> str:
        """Command string to run on the remote host."""
        raise NotImplementedError


class LLMProvider(ABC):
    """OpenAI-compatible HTTP API (OpenRouter, Together, Fireworks, …)."""

    name: str = "llm"

    @abstractmethod
    def health(self) -> HealthReport: ...

    @abstractmethod
    def chat(self, system: str, user: str, **kwargs: Any) -> str: ...

    def connection_info(self) -> dict[str, Any]:
        return {"provider": self.name}
