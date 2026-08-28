#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-2,3,4,5}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_DIR/prompts/smoke_t2va.txt}"
OUTPUT="${OUTPUT:-$PROJECT_DIR/outputs/smoke_t2va_fp8_4gpu_patched_4s.mp4}"
DURATION="${DURATION:-4}"
SHORT_EDGE="${SHORT_EDGE:-768}"

"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/patch_runtime_encoder_device.py"

VISIBLE_GPU_IDS="$VISIBLE_GPU_IDS" \
CHECK_GPUS=0 \
GPU_IDS="$GPU_IDS" \
NUM_GPUS="$NUM_GPUS" \
MODE=custom \
INFER_ARGS="--task t2va --duration $DURATION --short-edge $SHORT_EDGE --prompt-file $PROMPT_FILE --output $OUTPUT" \
EXTRA_ARGS="--quantization fp8" \
"$PROJECT_DIR/scripts/run_multigpu_infer.sh"
