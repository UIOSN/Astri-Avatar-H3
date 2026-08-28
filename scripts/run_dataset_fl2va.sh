#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${ENDPOINT:-http://127.0.0.1:30010}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/outputs}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DURATION="${DURATION:-4}"
SHORT_EDGE="${SHORT_EDGE:-768}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR/data/halfbody_dataset_dev0000}"
MANIFEST="${MANIFEST:-$DATASET_ROOT/manifests/passed.jsonl}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python3
fi

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/infer_h3.py" \
  --endpoint "$ENDPOINT" \
  --mode dataset-fl2va \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$MANIFEST" \
  --sample-index "$SAMPLE_INDEX" \
  --duration "$DURATION" \
  --short-edge "$SHORT_EDGE" \
  --output "$OUT_DIR/dataset_fl2va_${SAMPLE_INDEX}.mp4" \
  --request-json-out "$OUT_DIR/dataset_fl2va_${SAMPLE_INDEX}.request.json" \
  "$@"
