#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE_PYTHON="${PYTHON:-python}"
if [[ ! -x .cache/env-builder/bin/python ]]; then
  "$BASE_PYTHON" -m venv .cache/env-builder
fi
BUILDER="$PWD/.cache/env-builder/bin/python"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-20}"
export PIP_RESUME_RETRIES="${PIP_RESUME_RETRIES:-30}"
"$BUILDER" -m pip install --upgrade 'pip>=25.2,<27' --index-url "${DEPENDENCY_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
"$BUILDER" -m autodl_rag.envpack download --directory "${WHEELHOUSE:-$PWD/.cache/wheelhouse}" --dependency-index "${DEPENDENCY_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
