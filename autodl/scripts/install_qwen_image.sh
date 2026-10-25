#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IMAGE_PYTHON="${IMAGE_PYTHON:-/root/Qwen2.5/text-generation-webui/installer_files/env/bin/python}"
if [[ ! -x "$IMAGE_PYTHON" ]]; then
  printf '找不到镜像自带的 Python：%s\n请确认实例使用 Qwen 镜像，或通过 IMAGE_PYTHON 指定其解释器。\n' "$IMAGE_PYTHON" >&2
  exit 2
fi
export PYTHONNOUSERSITE=1
export PATH="$(dirname "$IMAGE_PYTHON"):$PATH"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-20}"
export PIP_RESUME_RETRIES="${PIP_RESUME_RETRIES:-30}"
"$IMAGE_PYTHON" -m autodl_rag.image_env --index-url "${DEPENDENCY_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m autodl_rag --config "${CONFIG:-configs/autodl-qwen-image.json}" doctor
