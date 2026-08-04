"""Append-only audit log for training runs (any industry; critical for high-stakes domains)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from aetherforge.utils.logging import get_logger

log = get_logger("audit")


@dataclass
class AuditEvent:
    event_id: str
    ts: float
    stage: str
    action: str
    actor: str = "aetherforge"
    details: dict[str, Any] = field(default_factory=dict)
    data_hash: Optional[str] = None
    checkpoint: Optional[str] = None
    gate_result: Optional[str] = None


class AuditLog:
    """JSONL audit trail written next to run artifacts."""

    def __init__(self, path: str | Path, run_id: Optional[str] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._fh = open(self.path, "a", encoding="utf-8")
        self.record("system", "audit_open", {"run_id": self.run_id, "path": str(self.path)})

    def record(
        self,
        stage: str,
        action: str,
        details: Optional[dict[str, Any]] = None,
        *,
        data_hash: Optional[str] = None,
        checkpoint: Optional[str] = None,
        gate_result: Optional[str] = None,
        actor: str = "aetherforge",
    ) -> AuditEvent:
        evt = AuditEvent(
            event_id=uuid.uuid4().hex,
            ts=time.time(),
            stage=stage,
            action=action,
            actor=actor,
            details=details or {},
            data_hash=data_hash,
            checkpoint=checkpoint,
            gate_result=gate_result,
        )
        line = json.dumps(asdict(evt), ensure_ascii=False, default=str)
        self._fh.write(line + "\n")
        self._fh.flush()
        log.debug("audit %s/%s", stage, action)
        return evt

    def close(self) -> None:
        try:
            self.record("system", "audit_close", {})
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "AuditLog":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
