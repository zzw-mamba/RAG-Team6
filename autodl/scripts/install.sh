#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IMAGE_PYTHON="${IMAGE_PYTHON:-/root/Qwen2.5/text-generation-webui/installer_files/env/bin/python}"
if [[ -z "${RAG_ENV_PROFILE:-}" && -x "$IMAGE_PYTHON" ]]; then
  export RAG_ENV_PROFILE=qwen-image
fi
if [[ "${RAG_ENV_PROFILE:-}" == qwen-image ]]; then
  exec bash scripts/install_qwen_image.sh
fi
BASE_PYTHON="${PYTHON:-python}"
"$BASE_PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "Use Python >=3.10"; print("Base Python:",sys.executable,sys.version)'
if [[ ! -x .venv/bin/python ]]; then
  if [[ -n "${WHEELHOUSE:-}" ]]; then
    "$BASE_PYTHON" -m venv .venv
  else
    "$BASE_PYTHON" -m venv --system-site-packages .venv
  fi
fi
if [[ -n "${WHEELHOUSE:-}" ]]; then
  # Offline bundles include the whole dependency set; exclude unrelated base packages.
  "$BASE_PYTHON" - <<'PY'
from pathlib import Path
import re
p = Path('.venv/pyvenv.cfg')
p.write_text(re.sub(r'(?m)^include-system-site-packages\s*=.*$',
                    'include-system-site-packages = false', p.read_text()))
PY
fi
source .venv/bin/activate
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-20}"
export PIP_RESUME_RETRIES="${PIP_RESUME_RETRIES:-30}"
if [[ -n "${WHEELHOUSE:-}" ]]; then
  # Verify before installing anything, then forbid network package resolution.
  python -m autodl_rag.envpack verify --directory "$WHEELHOUSE"
  export PIP_NO_INDEX=1
  python -m pip install --no-index --find-links "$WHEELHOUSE" --upgrade pip setuptools wheel
  python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$WHEELHOUSE/resolved-requirements.txt"
  python -m pip install --no-index --no-deps --no-build-isolation -e .
else
  python -m pip install --upgrade 'pip>=25.2,<27'
  # Reuse working image-provided torch; bootstrap CUDA torch if missing/CPU/old.
  python -m autodl_rag.bootstrap --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  python -m pip install -e '.[gpu]'
fi
python -m autodl_rag.bootstrap --verify-only
python -m unittest discover -s tests -v
python -m autodl_rag --config "${CONFIG:-configs/autodl.json}" doctor
echo 'Installed. For each new terminal: source .venv/bin/activate'
