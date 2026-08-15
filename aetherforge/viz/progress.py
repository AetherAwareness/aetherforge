"""
Live progress emitter for training runs.

Writes (atomically):
  <run_dir>/live_status.json   — polled by the dashboard
  <artifacts>/runs/active.json — pointer to the newest live run
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from aetherforge.utils.logging import get_logger
from aetherforge.viz.run_store import public_path

log = get_logger("viz.progress")


class StageState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageProgress:
    name: str
    state: str = StageState.PENDING.value
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    duration_sec: Optional[float] = None
    summary: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiveProgress:
    """Thread-safe-ish progress file writer (single pipeline process)."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str,
        run_name: str,
        domain: str,
        model: str,
        dry_run: bool,
        stage_list: list[str],
        artifacts_root: Optional[str | Path] = None,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / "live_status.json"
        self.artifacts_root = Path(
            artifacts_root or self.run_dir.parent
        )
        self._stages: dict[str, StageProgress] = {
            s: StageProgress(name=s) for s in stage_list
        }
        self._events: list[dict[str, Any]] = []
        self.payload: dict[str, Any] = {
            "schema": "aetherforge.live_status.v2",
            "run_id": run_id,
            "run_name": run_name,
            "domain": domain,
            "model": model,
            "dry_run": dry_run,
            "run_dir": str(public_path(self.run_dir)),
            "status": "starting",
            "promoted": False,
            "promote_blocked": None,
            "current_stage": None,
            "stage_list": list(stage_list),
            "stages": {s: self._stages[s].to_dict() for s in stage_list},
            "metrics_snapshot": {},
            # Sequential sector forge telemetry (dashboard constellation / timeline)
            "sector_mode": None,
            "sectors": {
                "overall": None,
                "current": None,
                "n_total": 0,
                "n_done": 0,
                "n_trained": 0,
                "n_blocked": 0,
                "n_skipped": 0,
                "items": [],
            },
            "visual": {
                "theme_hint": "nexus",
                "hero_label": "NEURAL COMMAND",
                "pulse": 0.0,
            },
            "controls": {
                "human_approved": False,
                "force_promote": False,
                "rejected": False,
                "notes": "",
            },
            "events": self._events,
            "updated_at": time.time(),
            "started_at": time.time(),
            "ended_at": None,
            "duration_sec": None,
            "percent": 0.0,
        }
        self.flush()
        self._write_active_pointer()

    # ── stage lifecycle ──────────────────────────────────────────────

    def start_run(self) -> None:
        self.payload["status"] = "running"
        self.event("run", "started")
        self.flush()

    def start_stage(self, name: str) -> None:
        st = self._stages.get(name) or StageProgress(name=name)
        self._stages[name] = st
        st.state = StageState.RUNNING.value
        st.started_at = time.time()
        st.error = None
        self.payload["current_stage"] = name
        self.payload["status"] = "running"
        self._sync_stages()
        self._update_percent()
        self.event("stage", f"start:{name}")
        self.flush()

    def end_stage(
        self,
        name: str,
        summary: Optional[dict[str, Any]] = None,
        *,
        failed: bool = False,
        error: Optional[str] = None,
    ) -> None:
        st = self._stages.get(name) or StageProgress(name=name)
        self._stages[name] = st
        st.ended_at = time.time()
        if st.started_at:
            st.duration_sec = st.ended_at - st.started_at
        st.state = StageState.FAILED.value if failed else StageState.DONE.value
        if summary:
            st.summary = _compact(summary)
        if error:
            st.error = error
        self._sync_stages()
        self._update_percent()
        self.event("stage", f"{'fail' if failed else 'done'}:{name}", st.summary)
        self.flush()

    def skip_stage(self, name: str, reason: str = "") -> None:
        st = self._stages.get(name) or StageProgress(name=name)
        self._stages[name] = st
        st.state = StageState.SKIPPED.value
        st.summary = {"reason": reason}
        self._sync_stages()
        self.event("stage", f"skip:{name}", {"reason": reason})
        self.flush()

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        self.payload["metrics_snapshot"] = _compact(metrics)
        self.flush()

    def set_sector_mode(self, mode: str) -> None:
        self.payload["sector_mode"] = mode
        self.event("sector", f"mode:{mode}")
        self.flush()

    def begin_sector_wave(self, *, n_total: int, mode: str = "sequential") -> None:
        """Start sequential sector forge wave (for dashboard timeline)."""
        self.payload["sector_mode"] = mode
        sec = self.payload.setdefault("sectors", {})
        sec["overall"] = "running"
        sec["n_total"] = int(n_total)
        sec["n_done"] = 0
        sec["n_trained"] = 0
        sec["n_blocked"] = 0
        sec["n_skipped"] = 0
        sec["current"] = None
        sec["items"] = []
        self.payload["visual"] = {
            "theme_hint": "nexus",
            "hero_label": "SECTOR FORGE",
            "pulse": 0.35,
        }
        self.event("sector", "wave_start", {"n_total": n_total, "mode": mode})
        self.flush()

    def sector_start(
        self,
        *,
        group_id: str,
        name: str,
        index: int,
        n_total: int,
        forensics_summary: str = "",
        readiness: str = "pass",
        n_experts: int = 0,
        n_samples: int = 0,
        color: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> None:
        """Mark a sector entering pre-train forensics / ESFT."""
        sec = self.payload.setdefault("sectors", {})
        item = {
            "group_id": group_id,
            "name": name,
            "index": index,
            "state": "running",
            "readiness": readiness,
            "n_experts": n_experts,
            "n_samples": n_samples,
            "forensics_summary": (forensics_summary or "")[:400],
            "color": color or "#5b9dff",
            "domain": domain,
            "started_at": time.time(),
            "ended_at": None,
            "duration_sec": None,
        }
        items = list(sec.get("items") or [])
        # replace if re-emitted
        items = [x for x in items if x.get("group_id") != group_id]
        items.append(item)
        items.sort(key=lambda x: x.get("index", 0))
        sec["items"] = items
        sec["current"] = group_id
        sec["n_total"] = n_total
        self.payload["visual"] = {
            "theme_hint": "nexus",
            "hero_label": f"SECTOR · {name}",
            "pulse": 0.7,
            "current_sector": name,
        }
        self.event(
            "sector",
            f"start:{name}",
            {
                "group_id": group_id,
                "index": index,
                "readiness": readiness,
                "n_experts": n_experts,
                "n_samples": n_samples,
            },
        )
        self.flush()

    def sector_end(
        self,
        *,
        group_id: str,
        name: str,
        status: str,
        readiness: str = "pass",
        n_experts: int = 0,
        n_samples: int = 0,
        forensics_summary: str = "",
        error: Optional[str] = None,
        duration_sec: Optional[float] = None,
        color: Optional[str] = None,
    ) -> None:
        """Complete one sector train step (trained | dry_run | blocked | skipped | error)."""
        sec = self.payload.setdefault("sectors", {})
        items = list(sec.get("items") or [])
        found = False
        for it in items:
            if it.get("group_id") == group_id:
                it["state"] = status
                it["readiness"] = readiness
                it["n_experts"] = n_experts
                it["n_samples"] = n_samples
                it["forensics_summary"] = (forensics_summary or it.get("forensics_summary") or "")[:400]
                it["ended_at"] = time.time()
                if duration_sec is not None:
                    it["duration_sec"] = duration_sec
                elif it.get("started_at"):
                    it["duration_sec"] = it["ended_at"] - it["started_at"]
                if error:
                    it["error"] = error
                if color:
                    it["color"] = color
                found = True
                break
        if not found:
            items.append(
                {
                    "group_id": group_id,
                    "name": name,
                    "state": status,
                    "readiness": readiness,
                    "n_experts": n_experts,
                    "n_samples": n_samples,
                    "forensics_summary": (forensics_summary or "")[:400],
                    "color": color or "#5b9dff",
                    "ended_at": time.time(),
                    "duration_sec": duration_sec,
                    "error": error,
                }
            )
        sec["items"] = items
        n_trained = sum(1 for x in items if x.get("state") in ("trained", "dry_run"))
        n_blocked = sum(1 for x in items if x.get("state") == "blocked")
        n_skipped = sum(1 for x in items if x.get("state") in ("skipped", "error"))
        n_done = sum(
            1
            for x in items
            if x.get("state") in ("trained", "dry_run", "blocked", "skipped", "error")
        )
        sec["n_trained"] = n_trained
        sec["n_blocked"] = n_blocked
        sec["n_skipped"] = n_skipped
        sec["n_done"] = n_done
        if sec.get("current") == group_id:
            sec["current"] = None
        self.payload["visual"] = {
            "theme_hint": "nexus",
            "hero_label": f"SECTOR · {status.upper()} · {name}",
            "pulse": 0.45 if status in ("trained", "dry_run") else 0.25,
            "last_sector": name,
            "last_status": status,
        }
        self.event(
            "sector",
            f"{status}:{name}",
            {"group_id": group_id, "readiness": readiness, "error": error},
        )
        self.flush()

    def end_sector_wave(self, *, overall: str = "done") -> None:
        sec = self.payload.setdefault("sectors", {})
        sec["overall"] = overall
        sec["current"] = None
        self.payload["visual"] = {
            "theme_hint": "nexus",
            "hero_label": "SECTOR FORGE · COMPLETE",
            "pulse": 0.15,
        }
        self.event("sector", "wave_end", {"overall": overall, "stats": {
            "n_trained": sec.get("n_trained"),
            "n_blocked": sec.get("n_blocked"),
            "n_skipped": sec.get("n_skipped"),
        }})
        self.flush()

    def set_promotion(
        self,
        *,
        promoted: bool,
        blocked: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        self.payload["promoted"] = promoted
        self.payload["promote_blocked"] = blocked
        if path:
            self.payload["promoted_path"] = path
        self.event(
            "promote",
            "promoted" if promoted else "blocked",
            {"blocked": blocked, "path": path},
        )
        self.flush()

    def finish(self, *, ok: bool = True, error: Optional[str] = None) -> None:
        self.payload["status"] = "completed" if ok else "failed"
        self.payload["ended_at"] = time.time()
        if self.payload.get("started_at"):
            self.payload["duration_sec"] = (
                self.payload["ended_at"] - self.payload["started_at"]
            )
        self.payload["current_stage"] = None
        self._update_percent(final=True)
        if error:
            self.payload["error"] = error
        self.event("run", "completed" if ok else "failed", {"error": error})
        self.flush()
        self._write_active_pointer()

    def event(
        self, kind: str, action: str, details: Optional[dict[str, Any]] = None
    ) -> None:
        self._events.append(
            {
                "ts": time.time(),
                "kind": kind,
                "action": action,
                "details": details or {},
            }
        )
        # cap event log
        if len(self._events) > 500:
            del self._events[:-400]
        self.payload["events"] = self._events

    def merge_controls_from_disk(self) -> dict[str, Any]:
        """Read operator control file written by the dashboard API."""
        ctrl_path = self.run_dir / "operator_controls.json"
        if not ctrl_path.exists():
            return self.payload.get("controls", {})
        try:
            data = json.loads(ctrl_path.read_text(encoding="utf-8"))
            controls = self.payload.setdefault("controls", {})
            controls.update(data)
            self.flush()
            return controls
        except Exception as e:
            log.debug("controls read failed: %s", e)
            return self.payload.get("controls", {})

    # ── IO ───────────────────────────────────────────────────────────

    def flush(self) -> None:
        self.payload["updated_at"] = time.time()
        self.payload["stages"] = {
            k: v.to_dict() for k, v in self._stages.items()
        }
        _atomic_write_json(self.status_path, self.payload)

    def _sync_stages(self) -> None:
        self.payload["stages"] = {
            k: v.to_dict() for k, v in self._stages.items()
        }

    def _update_percent(self, final: bool = False) -> None:
        total = max(len(self._stages), 1)
        done = sum(
            1
            for s in self._stages.values()
            if s.state in (StageState.DONE.value, StageState.SKIPPED.value)
        )
        running = any(s.state == StageState.RUNNING.value for s in self._stages.values())
        pct = 100.0 * done / total
        if running and not final:
            pct = min(99.0, pct + (50.0 / total))
        if final and self.payload.get("status") == "completed":
            pct = 100.0
        self.payload["percent"] = round(pct, 1)

    def _write_active_pointer(self) -> None:
        active = {
            "run_id": self.payload["run_id"],
            "run_dir": str(public_path(self.run_dir)),
            "status": self.payload["status"],
            "domain": self.payload["domain"],
            "updated_at": time.time(),
        }
        try:
            _atomic_write_json(self.artifacts_root / "active.json", active)
        except Exception as e:
            log.debug("active pointer write failed: %s", e)


def _compact(obj: Any, depth: int = 0) -> Any:
    """Keep live_status small enough for frequent polling."""
    if depth > 3:
        return str(type(obj).__name__)
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= 40:
                out["…"] = f"+{len(obj) - 40} keys"
                break
            if k in ("ranked", "routing_freq", "affinity", "grad_contrib", "per_expert"):
                if isinstance(v, list):
                    out[k] = {"_type": "list", "n": len(v), "head": v[:8]}
                else:
                    out[k] = {"_type": type(v).__name__}
            else:
                out[k] = _compact(v, depth + 1)
        return out
    if isinstance(obj, list):
        if len(obj) > 12:
            return [_compact(x, depth + 1) for x in obj[:8]] + [f"…+{len(obj) - 8}"]
        return [_compact(x, depth + 1) for x in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)[:200]


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".live_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
