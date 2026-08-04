#!/usr/bin/env bash
# Flagship logistics A3B recipe — dry-run locally or live on GPU / remote.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-dry-run}"
PY="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
fi

CFG_BASE=( -c configs/base.yaml -c recipes/flagship_logistics_a3b.yaml )
LIVE_EXTRA=( -c configs/qwen_a3b.yaml )

echo "══ AetherForge flagship · mode=$MODE ══"

case "$MODE" in
  dry-run|dry)
    $PY -m aetherforge.cli train "${CFG_BASE[@]}" --dry-run
    echo ""
    echo "Next: aetherforge dashboard  → open latest flagship-logistics-a3b-* run"
    echo "      Paint sectors on the lattice, assign data/samples/logistics if desired."
    ;;
  live|gpu)
    echo "Live training requires GPU + model weights (will download if missing)."
    $PY -m aetherforge.cli train "${CFG_BASE[@]}" "${LIVE_EXTRA[@]}"
    ;;
  remote-plan)
    $PY -m aetherforge.cli remote plan "${CFG_BASE[@]}" "${LIVE_EXTRA[@]}"
    ;;
  remote-launch)
    echo "This will SSH-run training on the connected box (GPU \$)."
    $PY -m aetherforge.cli remote launch --exec "${CFG_BASE[@]}" "${LIVE_EXTRA[@]}"
    ;;
  pull)
    $PY -m aetherforge.cli remote pull
    $PY -m aetherforge.cli remote logs --tail 80
    ;;
  report)
    # Print latest flagship run summary
    $PY - <<'PY'
from pathlib import Path
import json
root = Path("artifacts/runs")
runs = sorted(root.glob("flagship-logistics-a3b-*"), key=lambda p: p.stat().st_mtime, reverse=True)
if not runs:
    runs = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []
if not runs:
    print("No runs found under artifacts/runs/")
    raise SystemExit(1)
r = runs[0]
print("run:", r)
for name in ("pipeline_result.json", "scorecard.json", "expert_groups.json", "flagship_report.json"):
    p = r / name
    if p.exists():
        print(f"\n== {name} ==")
        data = json.loads(p.read_text())
        if name == "expert_groups.json":
            print(json.dumps(data.get("capacity", data) if isinstance(data, dict) else data, indent=2)[:1500])
        else:
            print(json.dumps(data, indent=2, default=str)[:2500])
PY
    ;;
  *)
    echo "Usage: $0 {dry-run|live|remote-plan|remote-launch|pull|report}"
    exit 1
    ;;
esac
