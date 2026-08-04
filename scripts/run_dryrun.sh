#!/usr/bin/env bash
# Smoke-test the full AetherForge pipeline without loading model weights.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python3 -m aetherforge.cli doctor || true
python3 -m aetherforge.cli train \
  -c configs/base.yaml \
  -c recipes/a3b_cardiology_dryrun.yaml \
  --dry-run

echo "Dry-run complete. Artifacts under artifacts/runs/"
