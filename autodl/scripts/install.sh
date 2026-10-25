#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE_PYTHON="${PYTHON:-python}"
"$BASE_PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "Use Python >=3.10"; print("Base Python:",sys.executable,sys.version)'
if [[ ! -x .venv/bin/python ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
# Reuse working image-provided torch; bootstrap CUDA torch if missing/CPU/old.
python -m autodl_rag.bootstrap --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install -e '.[gpu]'
python -m autodl_rag.bootstrap --verify-only
python -m unittest discover -s tests -v
python -m autodl_rag --config "${CONFIG:-configs/autodl.json}" doctor
echo 'Installed. For each new terminal: source .venv/bin/activate'
