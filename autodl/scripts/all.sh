#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash scripts/prepare.sh
bash scripts/train_pilot.sh
bash scripts/evaluate.sh
# Full test inference is an explicit separate step: bash scripts/predict.sh
