#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONNOUSERSITE=1
exec .venv/bin/python scripts/predict_parallel.py "$@"
