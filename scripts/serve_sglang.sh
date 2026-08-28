#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/MiniMax-H3}"
MODEL_VARIANT="${MODEL_VARIANT:-fl2va}"
MODEL_TYPE="${MODEL_TYPE:-diffusion}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30010}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-}"
RING_DEGREE="${RING_DEGREE:-}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-speed}"
SGLANG_BIN="${SGLANG_BIN:-$PROJECT_DIR/.venv/bin/sglang}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_DIR/server_outputs}"
INPUT_SAVE_PATH="${INPUT_SAVE_PATH:-$PROJECT_DIR/server_inputs}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/minimax_h3_torch_cache/inductor}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/dev/shm/minimax_h3_torch_cache/triton}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
WARMUP_MODE="${WARMUP_MODE:-off}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$NUM_GPUS" ]]; then
  IFS=',' read -r -a _gpu_array <<< "$GPU_IDS"
  NUM_GPUS="${#_gpu_array[@]}"
fi

if [[ -z "$ULYSSES_DEGREE" ]]; then
  ULYSSES_DEGREE="$NUM_GPUS"
fi

if [[ ! -x "$SGLANG_BIN" ]]; then
  SGLANG_BIN="sglang"
fi

if ! command -v "$SGLANG_BIN" >/dev/null 2>&1; then
  echo "sglang command not found. Install dependencies first, for example:" >&2
  echo "  cd $PROJECT_DIR && ./scripts/install_runtime.sh" >&2
  exit 127
fi

mkdir -p "$OUTPUT_PATH" "$INPUT_SAVE_PATH"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "Starting MiniMax-H3 $MODEL_VARIANT on GPUs $GPU_IDS"
echo "Endpoint: http://$HOST:$PORT/v1/videos"

read -r -a extra_args <<< "$EXTRA_ARGS"
ring_args=()
if [[ -n "$RING_DEGREE" ]]; then
  ring_args=(--ring-degree "$RING_DEGREE")
fi

TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
exec "$SGLANG_BIN" serve \
  --model-path "$MODEL_PATH" \
  --model-type "$MODEL_TYPE" \
  --num-gpus "$NUM_GPUS" \
  --gpu-ids "$GPU_IDS" \
  --ulysses-degree "$ULYSSES_DEGREE" \
  "${ring_args[@]}" \
  --performance-mode "$PERFORMANCE_MODE" \
  --host "$HOST" \
  --port "$PORT" \
  --model-variant "$MODEL_VARIANT" \
  --output-path "$OUTPUT_PATH" \
  --input-save-path "$INPUT_SAVE_PATH" \
  --warmup-mode "$WARMUP_MODE" \
  "${extra_args[@]}"
