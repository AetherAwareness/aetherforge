"""
Remote training helpers: sync project + launch aetherforge train on GPU box.

Does not silently spend money. Generates commands and optionally executes SSH.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from aetherforge.providers.registry import get_active_compute
from aetherforge.utils.logging import get_logger

log = get_logger("providers.remote_train")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_remote_bundle(
    *,
    config_paths: list[str],
    overrides: Optional[list[str]] = None,
    stages: Optional[str] = None,
    dry_run: bool = False,
    extra_args: str = "",
    background: bool = True,
) -> dict[str, Any]:
    """Produce sync + train instructions for the active compute provider."""
    comp = get_active_compute()
    if comp is None:
        return {
            "ok": False,
            "error": "No compute connection. Run: aetherforge connect vast --host … --port …",
        }

    local = str(project_root())
    sync_cmd = comp.sync_command(local)
    parts = []
    for c in config_paths:
        parts.append(f"-c {c}")
    for o in overrides or []:
        parts.append(f"-o {o}")
    if stages:
        parts.append(f"--stages {stages}")
    if dry_run:
        parts.append("--dry-run")
    if extra_args:
        parts.append(extra_args)
    train_args = " ".join(parts)
    remote_cmd = comp.remote_train_command(train_args, background=background)
    remote_fg = comp.remote_train_command(train_args, background=False)
    info = comp.connection_info()

    # full one-liner for copy-paste
    launch = (
        f"{sync_cmd} && ssh -p {info.get('port', 22)} "
        f"{info.get('user', 'root')}@{info.get('host')} {json.dumps(remote_cmd)}"
    )

    return {
        "ok": True,
        "provider": info.get("provider"),
        "connection": info,
        "sync_command": sync_cmd,
        "remote_train_command": remote_cmd,
        "remote_train_command_foreground": remote_fg,
        "background": background,
        "launch_hint": launch,
        "steps": [
            "1. Ensure your Vast/RunPod/SSH box is running and SSH works",
            "2. Run the sync_command (rsync project code)",
            "3. SSH in and run remote_train_command — or use: aetherforge remote launch --exec",
            "4. Open aetherforge dashboard locally; copy artifacts back when done",
            "5. Background mode writes artifacts/remote_train.nohup.log + .pid on the box",
        ],
        "pull_artifacts_command": _pull_artifacts_cmd(comp),
    }


def _pull_artifacts_cmd(comp: Any) -> str:
    info = comp.connection_info()
    host = info.get("host")
    port = info.get("port", 22)
    user = info.get("user", "root")
    rd = info.get("remote_dir", "/workspace/aetherforge")
    ident = info.get("identity_file")
    ssh_e = f"ssh -p {port}"
    if ident:
        ssh_e += f" -i {ident}"
    return (
        f"rsync -avz -e '{ssh_e}' "
        f"{user}@{host}:{rd}/artifacts/ ./artifacts/remote/"
    )


def exec_sync() -> dict[str, Any]:
    import subprocess

    bundle = build_remote_bundle(config_paths=["configs/base.yaml"])
    if not bundle.get("ok"):
        return bundle
    cmd = bundle["sync_command"]
    log.info("Executing sync: %s", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return {
        "ok": r.returncode == 0,
        "returncode": r.returncode,
        "stdout": (r.stdout or "")[-2000:],
        "stderr": (r.stderr or "")[-2000:],
        "command": cmd,
    }


def exec_remote_train(
    config_paths: list[str],
    *,
    overrides: Optional[list[str]] = None,
    stages: Optional[str] = None,
    dry_run: bool = False,
    timeout: int = 0,
    background: bool = True,
    skip_sync: bool = False,
) -> dict[str, Any]:
    """
    Sync + SSH-run train.

    background=True (default): nohup on the box so SSH can return; training
    continues after the desktop session disconnects.
    timeout 0 = no limit when foreground.
    """
    comp = get_active_compute()
    if comp is None or not hasattr(comp, "run_remote"):
        return {"ok": False, "error": "No SSH-capable compute connection"}
    if not comp.connection_info().get("host"):
        return {"ok": False, "error": "Compute profile has no SSH host"}

    # sync first
    if not skip_sync:
        sync = exec_sync()
        if not sync.get("ok"):
            return {"ok": False, "error": "sync failed", "sync": sync}
    else:
        sync = {"ok": True, "skipped": True}

    bundle = build_remote_bundle(
        config_paths=config_paths,
        overrides=overrides,
        stages=stages,
        dry_run=dry_run,
        background=background,
    )
    remote_cmd = bundle["remote_train_command"]
    log.info("Remote train (bg=%s): %s", background, remote_cmd)
    t0 = time.time()
    # Background start should return quickly; foreground may run for days
    ssh_timeout = 120 if background else (timeout or 3600 * 24)
    try:
        r = comp.run_remote(remote_cmd, timeout=ssh_timeout)
        ok = r.returncode == 0
        if background and ok:
            out = r.stdout or ""
            ok = "STARTED_PID=" in out or r.returncode == 0
        return {
            "ok": ok,
            "returncode": r.returncode,
            "background": background,
            "stdout_tail": (r.stdout or "")[-3000:],
            "stderr_tail": (r.stderr or "")[-2000:],
            "duration_sec": time.time() - t0,
            "remote_command": remote_cmd,
            "sync": {"ok": sync.get("ok"), "skipped": sync.get("skipped")},
            "connection": comp.connection_info(),
            "hint": (
                "Training detaches on the GPU box. "
                "Logs: artifacts/remote_train.nohup.log  |  "
                "aetherforge remote logs / pull"
                if background
                else "Foreground train finished (or timed out)."
            ),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "remote_command": remote_cmd, "sync": sync}


def exec_pull_artifacts(
    local_dir: Optional[str | Path] = None,
    *,
    remote_subdir: str = "artifacts",
) -> dict[str, Any]:
    """
    Rsync remote artifacts (and optional logs) down to local_dir.
    Default: ./artifacts/remote/
    """
    import subprocess

    comp = get_active_compute()
    if comp is None:
        return {"ok": False, "error": "No compute connection"}
    info = comp.connection_info()
    if not info.get("host"):
        return {"ok": False, "error": "No SSH host on active compute profile"}

    dest = Path(local_dir or (project_root() / "artifacts" / "remote"))
    dest.mkdir(parents=True, exist_ok=True)

    host = info["host"]
    port = int(info.get("port") or 22)
    user = info.get("user") or "root"
    rd = (info.get("remote_dir") or "/workspace/aetherforge").rstrip("/")
    ident = info.get("identity_file")
    ssh_e = f"ssh -p {port} -o StrictHostKeyChecking=accept-new"
    if ident:
        ssh_e += f" -i {ident}"

    src = f"{user}@{host}:{rd}/{remote_subdir.rstrip('/')}/"
    cmd = f"rsync -avz -e {json.dumps(ssh_e)} {json.dumps(src)} {json.dumps(str(dest) + '/')}"
    log.info("Pull artifacts: %s", cmd)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    result = {
        "ok": r.returncode == 0,
        "returncode": r.returncode,
        "local_dir": str(dest.resolve()),
        "command": cmd,
        "duration_sec": time.time() - t0,
        "stdout_tail": (r.stdout or "")[-2000:],
        "stderr_tail": (r.stderr or "")[-1500:],
    }
    # Also try to pull latest run logs if present
    if result["ok"]:
        log_pull = _pull_logs_inner(comp, dest / "_remote_logs")
        result["logs"] = log_pull
    return result


def _pull_logs_inner(comp: Any, dest: Path) -> dict[str, Any]:
    """Fetch tails of recent aetherforge.log files via SSH into dest."""
    import subprocess

    dest.mkdir(parents=True, exist_ok=True)
    info = comp.connection_info()
    rd = (info.get("remote_dir") or "/workspace/aetherforge").rstrip("/")
    # list newest run dirs and cat log tails
    remote_script = (
        f"cd {rd}/artifacts/runs 2>/dev/null || exit 0; "
        f"ls -1dt */ 2>/dev/null | head -5 | while read d; do "
        f"echo '===RUN:'$d; "
        f"test -f \"$d/aetherforge.log\" && tail -n 100 \"$d/aetherforge.log\"; "
        f"test -f \"$d/live_status.json\" && echo '--- live_status ---' && "
        f"python3 -c \"import json;p=json.load(open('${{d}}live_status.json'));"
        f"print(p.get('status'), p.get('percent'), p.get('current_stage'))\" 2>/dev/null; "
        f"done"
    )
    try:
        r = comp.run_remote(remote_script, timeout=60)
        out = r.stdout or ""
        (dest / "recent_runs_tail.txt").write_text(out, encoding="utf-8")
        return {
            "ok": r.returncode == 0,
            "path": str(dest / "recent_runs_tail.txt"),
            "preview": out[-2500:],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tail_remote_logs(n: int = 80, run_glob: str = "") -> dict[str, Any]:
    """
    Live tail of the newest remote run log (no full rsync).
    """
    comp = get_active_compute()
    if comp is None or not hasattr(comp, "run_remote"):
        return {"ok": False, "error": "No SSH-capable compute connection"}
    info = comp.connection_info()
    if not info.get("host"):
        return {"ok": False, "error": "No SSH host"}
    rd = (info.get("remote_dir") or "/workspace/aetherforge").rstrip("/")
    pattern = run_glob or "*"
    remote_script = (
        f"cd {rd}/artifacts/runs 2>/dev/null || {{ echo 'NO_RUNS'; exit 0; }}; "
        f"d=$(ls -1dt {pattern}/ 2>/dev/null | head -1); "
        f"if [ -z \"$d\" ]; then echo 'NO_RUNS'; exit 0; fi; "
        f"echo \"RUN_DIR=$d\"; "
        f"if [ -f \"$d/live_status.json\" ]; then echo '---LIVE---'; cat \"$d/live_status.json\"; fi; "
        f"if [ -f \"$d/aetherforge.log\" ]; then echo '---LOG---'; tail -n {int(n)} \"$d/aetherforge.log\"; fi; "
        f"if [ -f \"$d/flagship_report.json\" ]; then echo '---FLAGSHIP---'; cat \"$d/flagship_report.json\"; fi"
    )
    try:
        r = comp.run_remote(remote_script, timeout=45)
        text = r.stdout or ""
        live = None
        if "---LIVE---" in text:
            try:
                chunk = text.split("---LIVE---", 1)[1]
                chunk = chunk.split("---LOG---", 1)[0].strip()
                live = json.loads(chunk)
            except Exception:
                live = None
        return {
            "ok": r.returncode == 0,
            "raw": text[-8000:],
            "live_status": live,
            "returncode": r.returncode,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
