#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
CONFIG="${CONFIG:-configs/autodl.json}"
python -m autodl_rag --config "$CONFIG" doctor
python -m autodl_rag --config "$CONFIG" prepare
python -m autodl_rag --config "$CONFIG" index
