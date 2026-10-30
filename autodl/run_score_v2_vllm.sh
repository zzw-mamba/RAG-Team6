#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE="${1:-trial}"
CONFIG="${2:-/root/autodl-fs/zzw/configs/new-server.json}"
case "$STAGE" in install|serve|validate|predict|trial|all) ;; *) echo "Usage: bash run_score_v2_vllm.sh {install|serve|validate|predict|trial|all} [config]" >&2; exit 2 ;; esac
RAG_PYTHON="${RAG_PYTHON:-/root/autodl-tmp/zzw/RAG-Team6/autodl/.venv/bin/python}"
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
STORAGE="$("$RAG_PYTHON" -c 'import json,pathlib,sys; print(pathlib.Path(json.load(open(sys.argv[1]))["work_dir"]).parent)' "$CONFIG")"
mkdir -p "$STORAGE/tmp" "$STORAGE/cache" "$STORAGE/logs" "$STORAGE/vllm"
export TMPDIR="$STORAGE/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$STORAGE/cache"
export HF_HOME="$STORAGE/cache/huggingface"
export TORCH_HOME="$STORAGE/cache/torch"
export VLLM_CACHE_ROOT="$STORAGE/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$STORAGE/cache/torchinductor"
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_V1=1
VENV="$STORAGE/.venv-vllm"
if [[ "$STAGE" == install || "$STAGE" == trial || "$STAGE" == all ]]; then
  if [[ ! -x "$VENV/bin/python" ]]; then
    "$RAG_PYTHON" -m venv "$VENV"
  fi
  unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL
  "$VENV/bin/python" -m pip install --no-cache-dir 'pip>=25.2,<27' \
    --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
  "$VENV/bin/python" -m pip install --no-cache-dir --timeout 120 --retries 20 --resume-retries 30 \
    --only-binary=:all: 'vllm==0.11.2' \
    --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128
  "$VENV/bin/python" -m pip check
fi
ARGS=(--config "$CONFIG" --name "${SCORE_RUN_NAME:-score-v2}" --concurrency "${VLLM_CONCURRENCY:-32}"
      --pilot-size "${VLLM_PILOT_SIZE:-32}" --min-speedup "${VLLM_MIN_SPEEDUP:-1.5}")
if [[ "$STAGE" == serve || "$STAGE" == trial || "$STAGE" == all ]]; then
  "$RAG_PYTHON" -u scripts/score_predict_vllm.py "${ARGS[@]}" --stage stop
  export CUDA_VISIBLE_DEVICES="${VLLM_GPU:-0}"
  READY_FILE="$STORAGE/vllm/launch-$$-${RANDOM}.json"
  nohup "$VENV/bin/python" -u scripts/vllm_server.py --config "$CONFIG" \
    --port "${VLLM_PORT:-8106}" --memory-utilization "${VLLM_MEMORY_UTILIZATION:-0.65}" \
    --max-seqs "${VLLM_CONCURRENCY:-32}" --ready-file "$READY_FILE" \
    >> "$STORAGE/logs/vllm-server.log" 2>&1 < /dev/null &
  SERVER_PID=$!
  "$RAG_PYTHON" - "$STORAGE/vllm/server.json" "$SERVER_PID" "$READY_FILE" <<'PY'
import os
import sys
import time
from pathlib import Path
from autodl_rag.core import fingerprint, read_json
from scripts.vllm_client import request_json
deadline = time.monotonic() + 1200
while time.monotonic() < deadline:
    path = Path(sys.argv[1])
    if path.exists() and Path(sys.argv[3]).exists():
        manifest = read_json(path)
        ack = read_json(sys.argv[3])
        if manifest.get("status") == "ready" and ack.get("deployment") == fingerprint(manifest):
            try:
                request_json(manifest["endpoint"], "/version", timeout=3)
                print("vLLM server ready", flush=True)
                break
            except (OSError, RuntimeError):
                pass
    try:
        os.kill(int(sys.argv[2]), 0)
    except ProcessLookupError:
        raise SystemExit("vLLM launcher exited; inspect logs/vllm-server.log")
    time.sleep(2)
else:
    raise SystemExit("vLLM startup timeout; inspect logs/vllm-server.log")
PY
fi
if [[ "$STAGE" == validate || "$STAGE" == trial ]]; then
  exec "$RAG_PYTHON" -u scripts/score_predict_vllm.py "${ARGS[@]}" --stage validate
elif [[ "$STAGE" == predict || "$STAGE" == all ]]; then
  CLIENT_STAGE=all
  if [[ "$STAGE" == predict ]]; then CLIENT_STAGE=predict; fi
  exec "$RAG_PYTHON" -u scripts/score_predict_vllm.py "${ARGS[@]}" --stage "$CLIENT_STAGE"
fi
