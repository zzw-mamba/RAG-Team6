#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
CONFIG="${CONFIG:-configs/autodl.json}"
WORK="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["work_dir"])' "$CONFIG")"
RUN_NAME="${RUN_NAME:-pilot64}"
ADAPTER="$(cat "$WORK/training/$RUN_NAME/selection.adapter.txt")"
OUT="$WORK/evaluation/$RUN_NAME"
python -m autodl_rag --config "$CONFIG" run --split confirmation --mode rag --output "$OUT/rag.jsonl" --resume --restart-on-code-change
python -m autodl_rag --config "$CONFIG" run --split confirmation --mode critic --output "$OUT/critic.jsonl" --resume --restart-on-code-change
python -m autodl_rag --config "$CONFIG" run --split confirmation --mode rl --adapter "$ADAPTER" --output "$OUT/rl.jsonl" --resume --restart-on-code-change
python -m autodl_rag --config "$CONFIG" compare --rag "$OUT/rag.jsonl" --critic "$OUT/critic.jsonl" --rl "$OUT/rl.jsonl" --output "$OUT/comparison.json"
python -m autodl_rag --config "$CONFIG" demo --predictions "$OUT/rl.jsonl" --output "$OUT/demo.html"
