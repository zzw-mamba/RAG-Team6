#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
CONFIG="${CONFIG:-configs/autodl.json}"
WORK="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["work_dir"])' "$CONFIG")"
RUN_NAME="${RUN_NAME:-pilot64}"
ADAPTER="$(cat "$WORK/training/$RUN_NAME/selection.adapter.txt")"
OUT="$WORK/test/$RUN_NAME"
for mode in rag critic rl; do
  EXTRA=()
  if [[ "$mode" == rl ]]; then EXTRA+=(--adapter "$ADAPTER"); fi
  python -m autodl_rag --config "$CONFIG" run --split test --mode "$mode" "${EXTRA[@]}" --output "$OUT/$mode.jsonl" --resume
  if [[ ! -f "$OUT/$mode.csv" ]]; then
    python -m autodl_rag --config "$CONFIG" export --predictions "$OUT/$mode.jsonl" --output "$OUT/$mode.csv"
  fi
done
echo "Three CSVs: $OUT/{rag,critic,rl}.csv. Official accuracy requires Kaggle evaluation."
