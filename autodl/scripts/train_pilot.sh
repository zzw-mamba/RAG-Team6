#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
CONFIG="${CONFIG:-configs/autodl.json}"
WORK="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["work_dir"])' "$CONFIG")"
GROUPS_TOTAL="${PILOT_GROUPS:-64}"
RUN_NAME="${RUN_NAME:-pilot64}"
TRAIN_DIR="$WORK/training/$RUN_NAME"
TRAIN_ARGS=(--groups "$GROUPS_TOTAL" --save-every "${SAVE_EVERY:-16}" --run-dir "$TRAIN_DIR")
if [[ -n "${RESUME:-}" ]]; then TRAIN_ARGS+=(--resume "$RESUME"); fi
if [[ -n "${WARM_START:-}" ]]; then TRAIN_ARGS+=(--warm-start "$WARM_START"); fi
if [[ -z "${RESUME:-}" && -f "$TRAIN_DIR/latest.json" && ! -f "$TRAIN_DIR/summary.json" ]]; then
  LATEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$TRAIN_DIR/latest.json")"
  TRAIN_ARGS+=(--resume "$LATEST")
fi
if [[ ! -f "$TRAIN_DIR/summary.json" ]]; then
  python -m autodl_rag --config "$CONFIG" train "${TRAIN_ARGS[@]}"
else
  python -c 'import json,sys; assert json.load(open(sys.argv[1]))["groups"]==int(sys.argv[2]), "Use a NEW RUN_NAME when changing PILOT_GROUPS"' "$TRAIN_DIR/summary.json" "$GROUPS_TOTAL"
fi
mkdir -p "$TRAIN_DIR/tune"
python -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["trained"], "No optimizer updates occurred. Inspect progress/trajectories; this is not a trained RL model."' "$TRAIN_DIR/summary.json"
for checkpoint in "$TRAIN_DIR"/checkpoints/group-*; do
  [[ "$checkpoint" =~ /group-[0-9]{6}$ ]] || continue
  [[ -f "$checkpoint/complete.json" ]] || continue
  tag="$(basename "$checkpoint")"
  python -m autodl_rag --config "$CONFIG" run --split tune --mode rl --adapter "$checkpoint" --output "$TRAIN_DIR/tune/$tag.jsonl" --resume
done
python -m autodl_rag --config "$CONFIG" select --candidates "$TRAIN_DIR/tune" --output "$TRAIN_DIR/selection.json"
echo "Selection: $TRAIN_DIR/selection.json"
echo "Adapter: $TRAIN_DIR/selection.adapter.txt"
