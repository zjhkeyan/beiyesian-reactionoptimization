#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 CONFIG TRAIN CANDIDATES OUTPUT_DIR" >&2
  exit 2
fi

CONFIG="$1"
TRAIN="$2"
CANDIDATES="$3"
OUTPUT_DIR="$4"

count=0
for submission in "${OUTPUT_DIR}"/seed_*.pt; do
  bgolearn-rxn validate \
    --config "${CONFIG}" \
    --train "${TRAIN}" \
    --candidates "${CANDIDATES}" \
    --submission "${submission}"
  count=$((count + 1))
done

if [[ "${count}" -ne 20 ]]; then
  echo "expected 20 submissions, found ${count}" >&2
  exit 1
fi
