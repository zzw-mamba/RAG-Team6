#!/usr/bin/env bash
# Reuse the environment shipped with AutoDL.Art Qwen2.5-7B-Instruct v1.0.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RAG_ENV_PROFILE=qwen-image
export CONFIG="${CONFIG:-$PROJECT_DIR/configs/autodl-qwen-image.json}"
exec bash "$PROJECT_DIR/run_all.sh" "$@"
