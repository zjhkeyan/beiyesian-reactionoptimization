#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../external_evaluation/Suzuki}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/suzuki}"
BACKEND="${BACKEND:-native}"

bgolearn-rxn run-all \
  --config configs/suzuki.yaml \
  --train "${DATA_ROOT}/suzuki_train.csv" \
  --candidates "${DATA_ROOT}/suzuki_test_features.csv" \
  --oracle "${DATA_ROOT}/suzuki_test.csv" \
  --backend "${BACKEND}" \
  --output-dir "${OUTPUT_DIR}"
