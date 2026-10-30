#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE="${1:-tune}"
CONFIG="${2:-/root/autodl-fs/zzw/configs/new-server.json}"
if [[ "$STAGE" != tune && "$STAGE" != confirmation && "$STAGE" != predict && "$STAGE" != all ]]; then
  echo "Usage: bash run_score_v2.sh {tune|confirmation|predict|all} [config.json]" >&2
  exit 2
fi
RAG_PYTHON="${RAG_PYTHON:-}"
if [[ -z "$RAG_PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then
    RAG_PYTHON="$PWD/.venv/bin/python"
  elif [[ -x /root/autodl-tmp/zzw/RAG-Team6/autodl/.venv/bin/python ]]; then
    RAG_PYTHON=/root/autodl-tmp/zzw/RAG-Team6/autodl/.venv/bin/python
  else
    RAG_PYTHON="$(command -v python)"
  fi
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
STORAGE="$("$RAG_PYTHON" -c 'import json,pathlib,sys; print(pathlib.Path(json.load(open(sys.argv[1]))["work_dir"]).parent)' "$CONFIG")"
mkdir -p "$STORAGE/tmp" "$STORAGE/cache" "$STORAGE/logs"
export TMPDIR="$STORAGE/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$STORAGE/cache"
export HF_HOME="$STORAGE/cache/huggingface"
export TORCH_HOME="$STORAGE/cache/torch"
if [[ "$STAGE" == all ]]; then
  for PART in tune confirmation predict; do
    if [[ "$PART" == predict && "${SCORE_WORKERS:-2}" != 1 ]]; then
      "$RAG_PYTHON" -u scripts/score_predict_fast.py --config "$CONFIG" --name "${SCORE_RUN_NAME:-score-v2}" --workers "${SCORE_WORKERS:-2}"
    else
      "$RAG_PYTHON" -u -m autodl_rag.optimize --config "$CONFIG" --stage "$PART" --name "${SCORE_RUN_NAME:-score-v2}"
    fi
  done
else
  if [[ "$STAGE" == predict && "${SCORE_WORKERS:-2}" != 1 ]]; then
    exec "$RAG_PYTHON" -u scripts/score_predict_fast.py --config "$CONFIG" --name "${SCORE_RUN_NAME:-score-v2}" --workers "${SCORE_WORKERS:-2}"
  else
    exec "$RAG_PYTHON" -u -m autodl_rag.optimize --config "$CONFIG" --stage "$STAGE" --name "${SCORE_RUN_NAME:-score-v2}"
  fi
fi
