#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE="${1:-check}"
BASE="${2:-/root/autodl-fs/zzw/configs/new-server.json}"
case "$STAGE" in check|tune|confirmation|predict) ;; *) echo 'Usage: bash run_score_v5.sh {check|tune|confirmation|predict} [base-config]'; exit 2;; esac
export RAG_PYTHON="${RAG_PYTHON:-/root/autodl-tmp/zzw/RAG-Team6/autodl/.venv/bin/python}"
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
CONFIG="$("$RAG_PYTHON" - "$BASE" <<'PY'
import sys
import os
from pathlib import Path
from autodl_rag.core import load_config, save_json, read_json
cfg = load_config(sys.argv[1])
cfg['optimization_suite'] = 'score-v5'
# BGE is optional and explicit; no automatic model downloads.
bge = os.environ.get('BGE_PATH', '')
if bge and not Path(bge).is_dir():
    raise SystemExit('BGE_PATH is not a local model directory')
cfg['reranker_path'] = str(Path(bge).resolve()) if bge else None
cfg['reranker_device'] = os.environ.get('BGE_DEVICE', 'cpu')
path = Path(cfg['work_dir']).parent / 'configs' / 'score-v5.json'
if path.exists() and read_json(path) != cfg:
    raise SystemExit('Existing score-v5 config differs; preserve it and use a new experiment name/config before changing settings.')
if not path.exists():
    save_json(path, cfg)
print(path)
PY
)"
export SCORE_RUN_NAME=score-v5
export SCORE_WORKERS=1
if [[ "$STAGE" == check ]]; then
  bash run_score_v2.sh tune "$CONFIG"
  bash run_score_v2.sh confirmation "$CONFIG"
elif [[ "$STAGE" == predict ]]; then
  # Require a real paired batch pilot before any full new test run.
  bash run_score_v2_batch.sh all "$CONFIG"
else
  bash run_score_v2.sh "$STAGE" "$CONFIG"
fi
