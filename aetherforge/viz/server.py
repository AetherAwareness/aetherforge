"""
AetherForge Training Console — local HTTP server.

  aetherforge dashboard --port 8765
  open http://127.0.0.1:8765/

Endpoints:
  GET  /                     dashboard UI
  GET  /api/runs             list runs
  GET  /api/runs/<name>      full run bundle
  GET  /api/runs/<name>/live live_status only
  GET  /api/runs/<name>/groups           expert group plan + lattice
  GET  /api/runs/<name>/groups/<id>      deep group analysis (+ forensics)
  GET  /api/runs/<name>/forensics        full sector inventory (what each sector contains)
  POST /api/runs/<name>/groups           patch plan / repartition
  GET  /api/active           pointer to newest active run
  GET  /api/studio/preview   invent groups for a model family (no run)
  GET  /api/studio/forensics invent groups + forensics without a run
  POST /api/runs/<name>/control  {action: approve|reject|force_promote, note?: str}
"""

from __future__ import annotations

import json
import mimetypes
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from aetherforge.utils.logging import get_logger
from aetherforge.viz import run_store

log = get_logger("viz.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class DashboardHandler(BaseHTTPRequestHandler):
    runs_root: Path = run_store.default_runs_root()

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path in ("/", "/index.html", "/dashboard"):
                return self._serve_file(STATIC_DIR / "dashboard.html", "text/html; charset=utf-8")
            if path.startswith("/static/"):
                rel = path[len("/static/") :]
                return self._serve_file(STATIC_DIR / rel)
            if path == "/api/health":
                return self._json(200, {"ok": True, "runs_root": str(self.runs_root)})
            if path == "/api/runs":
                qs = parse_qs(parsed.query)
                limit = int(qs.get("limit", ["50"])[0])
                return self._json(200, {"runs": run_store.list_runs(self.runs_root, limit=limit)})
            if path == "/api/active":
                active_path = self.runs_root / "active.json"
                if not active_path.exists():
                    # fallback: newest run with live_status
                    runs = run_store.list_runs(self.runs_root, limit=1)
                    return self._json(200, {"active": runs[0] if runs else None})
                data = json.loads(active_path.read_text(encoding="utf-8"))
                return self._json(200, {"active": data})
            if path.startswith("/api/runs/"):
                rest = path[len("/api/runs/") :]
                parts = [p for p in rest.split("/") if p]
                if not parts:
                    return self._json(400, {"error": "missing run name"})
                name = parts[0]
                run_dir = self._resolve_run(name)
                if run_dir is None:
                    return self._json(404, {"error": f"run not found: {name}"})
                if len(parts) == 1:
                    return self._json(200, run_store.load_run_bundle(run_dir))
                if parts[1] == "live":
                    live = run_store._read_json(run_dir / "live_status.json")
                    return self._json(200, live or {"status": "unknown"})
                if parts[1] == "groups":
                    return self._handle_groups_get(run_dir, parts[2:])
                if parts[1] == "forensics":
                    return self._handle_forensics_get(run_dir)
                return self._json(404, {"error": "unknown sub-resource"})
            if path == "/api/studio/preview":
                qs = parse_qs(parsed.query)
                return self._json(200, self._studio_preview(qs))
            if path == "/api/studio/forensics":
                qs = parse_qs(parsed.query)
                return self._json(200, self._studio_forensics(qs))
            if path == "/api/providers":
                from aetherforge.providers import connect as pconn

                return self._json(200, pconn.catalog())
            if path == "/api/providers/status":
                from aetherforge.providers import connect as pconn

                return self._json(200, pconn.status())
            if path == "/api/remote/plan":
                from aetherforge.providers.remote_train import build_remote_bundle

                qs = parse_qs(parsed.query)
                cfgs = qs.get("config") or ["configs/base.yaml"]
                return self._json(
                    200,
                    build_remote_bundle(
                        config_paths=cfgs,
                        dry_run=(qs.get("dry_run") or ["0"])[0] in ("1", "true", "yes"),
                    ),
                )
            if path == "/api/remote/logs":
                from aetherforge.providers.remote_train import tail_remote_logs

                qs = parse_qs(parsed.query)
                n = int((qs.get("tail") or ["80"])[0])
                return self._json(
                    200,
                    tail_remote_logs(n=n, run_glob=(qs.get("run_glob") or [""])[0]),
                )
            return self._json(404, {"error": "not found", "path": path})
        except Exception as e:
            log.exception("GET failed")
            return self._json(500, {"error": str(e), "trace": traceback.format_exc()[-800:]})

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "invalid JSON body"})

            if path.startswith("/api/runs/") and path.endswith("/control"):
                name = path[len("/api/runs/") : -len("/control")].strip("/")
                run_dir = self._resolve_run(name)
                if run_dir is None:
                    return self._json(404, {"error": f"run not found: {name}"})
                action = (payload.get("action") or "").strip().lower()
                note = payload.get("note") or payload.get("notes") or ""
                if action in ("approve", "human_approve"):
                    ctrl = run_store.apply_human_approve(run_dir, note=note)
                    return self._json(200, {"ok": True, "controls": ctrl})
                if action == "reject":
                    ctrl = run_store.apply_reject(run_dir, note=note)
                    return self._json(200, {"ok": True, "controls": ctrl})
                if action in ("force_promote", "promote"):
                    result = run_store.force_promote_package(run_dir, note=note)
                    return self._json(200, result)
                if action == "note":
                    ctrl = run_store.write_controls(run_dir, {"notes": note, "action": "note"})
                    return self._json(200, {"ok": True, "controls": ctrl})
                return self._json(
                    400,
                    {
                        "error": f"unknown action: {action}",
                        "allowed": ["approve", "reject", "force_promote", "note"],
                    },
                )
            if path.startswith("/api/runs/") and path.rstrip("/").endswith("/groups"):
                name = path[len("/api/runs/") :].split("/groups")[0].strip("/")
                run_dir = self._resolve_run(name)
                if run_dir is None:
                    return self._json(404, {"error": f"run not found: {name}"})
                return self._json(200, self._handle_groups_post(run_dir, payload))
            if path == "/api/providers/connect":
                return self._json(200, self._providers_connect(payload))
            if path == "/api/providers/key":
                from aetherforge.providers import connect as pconn

                provider = (payload.get("provider") or "").strip()
                key = (payload.get("api_key") or payload.get("value") or "").strip()
                if not provider or not key:
                    return self._json(400, {"error": "provider and api_key required"})
                return self._json(200, pconn.save_api_key(provider, key))
            if path == "/api/remote/pull":
                from aetherforge.providers.remote_train import exec_pull_artifacts

                return self._json(
                    200,
                    exec_pull_artifacts(local_dir=payload.get("dest")),
                )
            if path == "/api/remote/logs":
                from aetherforge.providers.remote_train import tail_remote_logs

                return self._json(
                    200,
                    tail_remote_logs(
                        n=int(payload.get("tail") or 80),
                        run_glob=payload.get("run_glob") or "",
                    ),
                )
            return self._json(404, {"error": "not found"})
        except FileNotFoundError as e:
            return self._json(404, {"error": str(e)})
        except Exception as e:
            log.exception("POST failed")
            return self._json(500, {"error": str(e)})

    def _handle_groups_get(self, run_dir: Path, rest: list[str]) -> None:
        from aetherforge.groups.studio import analyze_group, lattice_view
        from aetherforge.groups.store import load_group_plan

        path = run_dir / "expert_groups.json"
        if not path.exists():
            return self._json(404, {"error": "no expert_groups.json — run groups stage first"})
        plan = load_group_plan(path)
        aff = run_store._read_json(run_dir / "affinity.json")
        if not rest:
            return self._json(
                200,
                {
                    "summary": plan.summary(),
                    "lattice": lattice_view(plan),
                    "plan": plan.to_dict(),
                },
            )
        return self._json(
            200, analyze_group(plan, rest[0], affinity=aff, with_forensics=True)
        )

    def _handle_forensics_get(self, run_dir: Path) -> None:
        from aetherforge.groups.forensics import run_model_forensics, forensics_markdown
        from aetherforge.groups.store import load_group_plan

        path = run_dir / "expert_groups.json"
        if not path.exists():
            return self._json(
                404, {"error": "no expert_groups.json — run groups stage first"}
            )
        plan = load_group_plan(path)
        aff = run_store._read_json(run_dir / "affinity.json")
        report = run_model_forensics(plan, affinity=aff)
        # cache for offline inspection
        try:
            (run_dir / "sector_forensics.json").write_text(
                json.dumps(report.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
            (run_dir / "sector_forensics.md").write_text(
                forensics_markdown(report), encoding="utf-8"
            )
        except Exception as e:
            log.warning("Could not cache forensics: %s", e)
        return self._json(200, report.to_dict())

    def _handle_groups_post(self, run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
        from aetherforge.groups.capacity import estimate_capacity
        from aetherforge.groups.cluster import auto_partition_groups, merge_affinity_into_plan
        from aetherforge.groups.studio import lattice_view
        from aetherforge.groups.store import apply_plan_patch, load_group_plan, save_group_plan

        path = run_dir / "expert_groups.json"
        action = (payload.get("action") or "patch").lower()

        if action == "repartition":
            # Rebuild groups with new count/strategy
            family = payload.get("family") or "deepseek_v4_flash"
            n = int(payload.get("num_groups") or payload.get("target_num_groups") or 8)
            strategy = payload.get("strategy") or "active_slots"
            aff = run_store._read_json(run_dir / "affinity.json")
            if path.exists():
                old = load_group_plan(path)
                cap = old.capacity
                family = old.family or family
            else:
                cap = estimate_capacity(
                    family=family,
                    model_name=payload.get("model_name") or "",
                    total_params_b=payload.get("total_params_b"),
                    active_params_b=payload.get("active_params_b"),
                )
            plan = auto_partition_groups(
                cap,
                num_groups=n,
                strategy=strategy,
                affinity_matrix=(aff or {}).get("affinity"),
                ranked=(aff or {}).get("ranked"),
                model_name=cap.model_name,
                target_active_fire_ratio=float(
                    payload.get("target_active_fire_ratio") or 1.0
                ),
            )
            if aff:
                merge_affinity_into_plan(
                    plan, aff.get("affinity"), aff.get("ranked")
                )
            save_group_plan(plan, path)
            return {"ok": True, "summary": plan.summary(), "lattice": lattice_view(plan)}

        if not path.exists():
            raise FileNotFoundError("no expert_groups.json")
        plan = load_group_plan(path)
        plan = apply_plan_patch(plan, payload.get("patch") or payload)
        save_group_plan(plan, path)
        return {"ok": True, "summary": plan.summary(), "lattice": lattice_view(plan), "plan": plan.to_dict()}

    def _providers_connect(self, payload: dict[str, Any]) -> dict[str, Any]:
        from aetherforge.providers import connect as pconn

        kind = (payload.get("kind") or "compute").lower()
        provider = (payload.get("provider") or "").lower()
        if kind == "llm":
            if payload.get("api_key"):
                pconn.save_api_key(provider, payload["api_key"])
            return pconn.connect_llm(
                provider,
                name=payload.get("name"),
                model=payload.get("model"),
                base_url=payload.get("base_url"),
                test=bool(payload.get("test", True)),
            )
        return pconn.connect_compute(
            provider,
            name=payload.get("name"),
            host=payload.get("host") or "",
            port=int(payload.get("port") or 22),
            user=payload.get("user") or "root",
            identity_file=payload.get("identity_file"),
            remote_dir=payload.get("remote_dir") or "/workspace/aetherforge",
            instance_id=payload.get("instance_id"),
            pod_id=payload.get("pod_id"),
            test=bool(payload.get("test", True)),
        )

    def _studio_preview(self, qs: dict) -> dict[str, Any]:
        from aetherforge.groups.studio import create_studio_plan, lattice_view

        family = (qs.get("family") or ["deepseek_v4_flash"])[0]
        n = int((qs.get("num_groups") or ["8"])[0])
        strategy = (qs.get("strategy") or ["active_slots"])[0]
        plan = create_studio_plan(
            family=family,
            model_name=(qs.get("model") or [""])[0],
            num_groups=n,
            strategy=strategy,
        )
        return {
            "summary": plan.summary(),
            "lattice": lattice_view(plan),
            "capacity": plan.capacity.to_dict(),
            "hint": (
                f"This model ≈{plan.capacity.active_params_b}B active / "
                f"{plan.capacity.total_params_b}B total. "
                f"You can carve up to ~{plan.capacity.max_disjoint_active_groups} "
                f"disjoint sectors near one active-fire mass."
            ),
        }

    def _studio_forensics(self, qs: dict) -> dict[str, Any]:
        from aetherforge.groups.studio import create_studio_plan, lattice_view
        from aetherforge.groups.forensics import run_model_forensics

        family = (qs.get("family") or ["deepseek_v4_flash"])[0]
        n = int((qs.get("num_groups") or ["12"])[0])
        strategy = (qs.get("strategy") or ["active_slots"])[0]
        plan = create_studio_plan(
            family=family,
            model_name=(qs.get("model") or [""])[0],
            num_groups=n,
            strategy=strategy,
        )
        report = run_model_forensics(plan)
        return {
            "summary": plan.summary(),
            "lattice": lattice_view(plan),
            "capacity": plan.capacity.to_dict(),
            "forensics": report.to_dict(),
        }

    def _resolve_run(self, name: str) -> Optional[Path]:
        # exact dir name, or run_id match
        direct = self.runs_root / name
        if direct.is_dir():
            return direct
        for d in self.runs_root.iterdir() if self.runs_root.exists() else []:
            if not d.is_dir():
                continue
            if name in d.name:
                return d
            live = run_store._read_json(d / "live_status.json")
            if live and live.get("run_id") == name:
                return d
            result = run_store._read_json(d / "pipeline_result.json")
            if result and result.get("run_id") == name:
                return d
        return None

    def _json(self, code: int, data: Any) -> None:
        raw = json.dumps(data, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._cors()
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _serve_file(self, path: Path, content_type: Optional[str] = None) -> None:
        if not path.exists() or not path.is_file():
            return self._json(404, {"error": f"file not found: {path.name}"})
        data = path.read_bytes()
        ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    runs_root: Optional[str | Path] = None,
) -> None:
    root = Path(runs_root) if runs_root else run_store.default_runs_root()
    root.mkdir(parents=True, exist_ok=True)
    DashboardHandler.runs_root = root
    httpd = ThreadingHTTPServer((host, port), DashboardHandler)
    log.info("AetherForge dashboard on http://%s:%d/  (runs=%s)", host, port, root)
    print(f"AetherForge Training Console → http://{host}:{port}/")
    print(f"Runs root: {root}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        httpd.server_close()
