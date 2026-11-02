#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE="${1:-check}"
BASE="${2:-/root/autodl-fs/zzw/configs/new-server.json}"
case "$STAGE" in check|tune|confirmation|predict) ;; *) echo 'Usage: bash run_score_v3.sh {check|tune|confirmation|predict} [base-config]'; exit 2;; esac
export RAG_PYTHON="${RAG_PYTHON:-/root/autodl-tmp/zzw/RAG-Team6/autodl/.venv/bin/python}"
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
CONFIG="$("$RAG_PYTHON" - "$BASE" <<'PY'
import sys
from pathlib import Path
from autodl_rag.core import load_config, save_json, read_json
cfg = load_config(sys.argv[1])
cfg['optimization_suite'] = 'score-v3'
path = Path(cfg['work_dir']).parent / 'configs' / 'score-v3.json'
if path.exists() and read_json(path) != cfg:
    raise SystemExit('Existing score-v3 config differs; preserve it and use a new experiment name/config before changing settings.')
if not path.exists():
    save_json(path, cfg)
print(path)
PY
)"
export SCORE_RUN_NAME=score-v3
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
