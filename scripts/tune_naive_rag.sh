#!/usr/bin/env bash
set -euo pipefail

RAG_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAG_REPO_DIR"
export PYTHONPATH="$RAG_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
RAG_PYTHON="${RAG_PYTHON:-python}"
RAG_DEV_INPUT="${1:-data/evaluation-v1/dev.jsonl}"
RAG_TUNE_OUTPUT="${2:-outputs/naive-rag-tune-v1}"
RAG_DRY_RUN=false
case "${3:-}" in
  "") ;;
  --dry-run) RAG_DRY_RUN=true ;;
  *) printf 'Usage: bash scripts/tune_naive_rag.sh [input] [output-root] [--dry-run]\n' >&2; exit 2 ;;
esac

RAG_RUN_NAMES=(baseline bm25 evidence)
RAG_RUN_CONFIGS=(
  configs/autodl_baseline_best_supported_batch16.json
  configs/autodl_best_supported_bm25_batch16.json
  configs/autodl_evidence_answer_batch16.json
)

for RAG_RUN_INDEX in "${!RAG_RUN_NAMES[@]}"; do
  RAG_RUN_DIR="$RAG_TUNE_OUTPUT/${RAG_RUN_NAMES[$RAG_RUN_INDEX]}"
  RAG_EVAL_ARGS=(--input "$RAG_DEV_INPUT" --config "${RAG_RUN_CONFIGS[$RAG_RUN_INDEX]}" --output "$RAG_RUN_DIR")
  if [[ -f "$RAG_RUN_DIR/manifest.json" ]]; then
    RAG_EVAL_ARGS+=(--resume)
  fi
  if [[ "$RAG_DRY_RUN" == true ]]; then
    RAG_EVAL_ARGS+=(--dry-run)
  fi
  "$RAG_PYTHON" -u -m naive_rag.cli evaluate "${RAG_EVAL_ARGS[@]}"
done

if [[ ! -e "$RAG_TUNE_OUTPUT/comparison" ]]; then
  "$RAG_PYTHON" -m naive_rag.cli compare \
    --reports "$RAG_TUNE_OUTPUT/baseline/report" "$RAG_TUNE_OUTPUT/bm25/report" "$RAG_TUNE_OUTPUT/evidence/report" \
    --output "$RAG_TUNE_OUTPUT/comparison"
else
  cat "$RAG_TUNE_OUTPUT/comparison/comparison.md"
fi
