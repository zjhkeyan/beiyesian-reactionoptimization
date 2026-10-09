#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../external_evaluation/Buchwald_sub4}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/buchwald_sub4}"
BACKEND="${BACKEND:-native}"

bgolearn-rxn run-all \
  --config configs/buchwald_sub4.yaml \
  --train "${DATA_ROOT}/buchwald_sub4_train.csv" \
  --candidates "${DATA_ROOT}/buchwald_sub4_test_features.csv" \
  --oracle "${DATA_ROOT}/buchwald_sub4_test.csv" \
  --backend "${BACKEND}" \
  --output-dir "${OUTPUT_DIR}"
