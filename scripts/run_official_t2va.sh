#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${ENDPOINT:-http://127.0.0.1:30010}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/outputs}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DURATION="${DURATION:-4}"
SHORT_EDGE="${SHORT_EDGE:-768}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python3
fi

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/infer_h3.py" \
  --endpoint "$ENDPOINT" \
  --mode official-t2va \
  --duration "$DURATION" \
  --short-edge "$SHORT_EDGE" \
  --output "$OUT_DIR/official_t2va.mp4" \
  --request-json-out "$OUT_DIR/official_t2va.request.json" \
  "$@"
