#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
CONFIG="${CONFIG:-configs/autodl.json}"
WORK="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["work_dir"])' "$CONFIG")"
python -m autodl_rag --config "$CONFIG" doctor
python -m autodl_rag --config "$CONFIG" run --split tune --mode rag --limit 4 --output "$WORK/smoke/rag.jsonl" --resume --restart-on-code-change
python -m autodl_rag --config "$CONFIG" run --split tune --mode critic --limit 4 --output "$WORK/smoke/critic.jsonl" --resume --restart-on-code-change
if [[ ! -f "$WORK/smoke/train/summary.json" ]]; then
  python -m autodl_rag --config "$CONFIG" train --groups 2 --save-every 2 --run-dir "$WORK/smoke/train" --probe-backward
fi
python -m autodl_rag --config "$CONFIG" run --split tune --mode rl --adapter "$WORK/smoke/train/checkpoints/group-000002" --limit 4 --output "$WORK/smoke/rl.jsonl" --resume --restart-on-code-change
python -m autodl_rag --config "$CONFIG" compare --split tune --rag "$WORK/smoke/rag.jsonl" --critic "$WORK/smoke/critic.jsonl" --rl "$WORK/smoke/rl.jsonl" --output "$WORK/smoke/comparison.json"
echo 'Real GPU smoke completed. These four-question diagnostic scores are not research results.'
