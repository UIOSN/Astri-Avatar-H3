#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-MiniMaxAI/MiniMax-H3}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/MiniMax-H3}"
VARIANT="${VARIANT:-FL2VA}"

case "$VARIANT" in
  FL2VA|fl2va)
    INCLUDE_VARIANT="FL2VA/*"
    ;;
  Ref2VA|ref2va)
    INCLUDE_VARIANT="Ref2VA/*"
    ;;
  both|BOTH|all|ALL)
    INCLUDE_VARIANT=""
    ;;
  *)
    echo "Unsupported VARIANT=$VARIANT. Use FL2VA, Ref2VA, or both." >&2
    exit 2
    ;;
esac

mkdir -p "$MODEL_DIR"

if [[ -n "$INCLUDE_VARIANT" ]]; then
  hf download "$MODEL_ID" \
    --include "model_index.json" "modular_model_index.json" "$INCLUDE_VARIANT" \
    --local-dir "$MODEL_DIR"
else
  hf download "$MODEL_ID" \
    --include "model_index.json" "modular_model_index.json" "FL2VA/*" "Ref2VA/*" \
    --local-dir "$MODEL_DIR"
fi

echo "Model downloaded to $MODEL_DIR"
