#!/usr/bin/env bash
# Build sdist + wheel for GitHub Releases / pip install
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
rm -rf dist build *.egg-info
python3 -m pip install -q build
python3 -m build
echo "Artifacts in dist/:"
ls -la dist/
