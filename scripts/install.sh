#!/usr/bin/env bash
# Portable install for GitHub / Hugging Face downloads
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel
pip install -e ".[dev]"

echo ""
echo "AetherForge installed."
echo "  Activate:  source $ROOT/.venv/bin/activate"
echo "  Doctor:    aetherforge doctor"
echo "  Groups:    aetherforge groups --preview --family deepseek_v4_flash --num-groups 12"
echo "  Dashboard: aetherforge dashboard"
echo "  Dry-run:   aetherforge train -c configs/base.yaml -c recipes/generic_dryrun.yaml --dry-run"
