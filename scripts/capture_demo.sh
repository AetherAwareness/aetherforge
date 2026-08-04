#!/usr/bin/env bash
# Capture Neural Command demo assets (PNG + WebM + GIF) for README / HF.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3}"
[[ -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"

# Ensure a run exists
if ! ls artifacts/runs/*/pipeline_result.json >/dev/null 2>&1; then
  echo "No runs found — generating flagship dry-run first…"
  bash scripts/run_flagship.sh dry-run
fi

# Prefer playwright chromium already installed
$PY scripts/capture_demo.py --out docs/demo --theme "${1:-nexus}" "$@"
echo "Done. See docs/demo/"
