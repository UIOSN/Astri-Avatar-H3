#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:-dataset-fl2va}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
GPU_COUNT="${GPU_COUNT:-4}"
PORT="${PORT:-30010}"
HOST="${HOST:-0.0.0.0}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
ENDPOINT="${ENDPOINT:-http://$CLIENT_HOST:$PORT}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
SERVER_LOG="${SERVER_LOG:-$PROJECT_DIR/server_outputs/sglang_${MODE}_$(date +%Y%m%d_%H%M%S).log}"
LOW_MEMORY="${LOW_MEMORY:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
INFER_ARGS="${INFER_ARGS:-}"
CHECK_GPUS="${CHECK_GPUS:-1}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-70000}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

mkdir -p "$(dirname "$SERVER_LOG")"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python3
fi

if [[ "$LOW_MEMORY" == "1" || "$LOW_MEMORY" == "true" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--text-encoder-cpu-offload"
fi

check_gpu() {
  local gpu_id="$1"
  local free_mem

  free_mem="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [[ -z "$free_mem" ]]; then
    echo "GPU $gpu_id: unavailable to nvidia-smi" >&2
    return 1
  fi
  if (( free_mem < MIN_FREE_MEM_MB )); then
    echo "GPU $gpu_id: only ${free_mem} MiB free; need at least ${MIN_FREE_MEM_MB} MiB" >&2
    return 1
  fi
  if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" - <<'PY' >/tmp/minimax_h3_gpu_check.out 2>&1
import torch
torch.empty((1,), device="cuda")
print("ok")
PY
  then
    echo "GPU $gpu_id: CUDA allocation failed" >&2
    grep -m1 -E "CUDA error|AcceleratorError|RuntimeError" /tmp/minimax_h3_gpu_check.out >&2 || true
    return 1
  fi
  echo "GPU $gpu_id: ok (${free_mem} MiB free)"
}

if [[ "$GPU_IDS" == "auto" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; set GPU_IDS explicitly." >&2
    exit 127
  fi
  selected_gpus=()
  while IFS= read -r gpu_id; do
    if check_gpu "$gpu_id"; then
      selected_gpus+=("$gpu_id")
      if (( ${#selected_gpus[@]} >= GPU_COUNT )); then
        break
      fi
    fi
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  if (( ${#selected_gpus[@]} < GPU_COUNT )); then
    echo "Could not find ${GPU_COUNT} healthy GPUs with at least ${MIN_FREE_MEM_MB} MiB free." >&2
    exit 1
  fi
  GPU_IDS="$(IFS=,; echo "${selected_gpus[*]}")"
  CHECK_GPUS=0
  echo "Selected GPU_IDS=$GPU_IDS"
fi

if [[ "$CHECK_GPUS" == "1" || "$CHECK_GPUS" == "true" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; set CHECK_GPUS=0 to skip the preflight gate." >&2
    exit 127
  fi
  IFS=',' read -r -a selected_gpus <<< "$GPU_IDS"
  gpu_check_failed=0
  for gpu_id in "${selected_gpus[@]}"; do
    check_gpu "$gpu_id" || gpu_check_failed=1
  done
  if (( gpu_check_failed )); then
    echo "GPU preflight failed; choose healthy GPUs or set CHECK_GPUS=0 to bypass this gate." >&2
    exit 1
  fi
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting MiniMax-H3 server on GPUs $GPU_IDS; log: $SERVER_LOG"
server_env=(
  PROJECT_DIR="$PROJECT_DIR"
  GPU_IDS="$GPU_IDS"
  PORT="$PORT"
  HOST="$HOST"
  EXTRA_ARGS="$EXTRA_ARGS"
)
if [[ -n "$VISIBLE_GPU_IDS" ]]; then
  server_env+=(CUDA_VISIBLE_DEVICES="$VISIBLE_GPU_IDS")
fi

setsid env \
  "${server_env[@]}" \
  "$PROJECT_DIR/scripts/serve_sglang.sh" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + SERVER_READY_TIMEOUT))
until curl -fsS --max-time 5 "$ENDPOINT/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server exited before becoming ready. Last log lines:" >&2
    tail -80 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for $ENDPOINT/health. Last log lines:" >&2
    tail -80 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done

echo "Server is ready at $ENDPOINT"
read -r -a infer_extra_args <<< "$INFER_ARGS"

case "$MODE" in
  official-t2va)
    ENDPOINT="$ENDPOINT" "$PROJECT_DIR/scripts/run_official_t2va.sh" "${infer_extra_args[@]}"
    ;;
  dataset-fl2va)
    ENDPOINT="$ENDPOINT" "$PROJECT_DIR/scripts/run_dataset_fl2va.sh" "${infer_extra_args[@]}"
    ;;
  custom)
    "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/infer_h3.py" \
      --endpoint "$ENDPOINT" \
      --mode custom \
      "${infer_extra_args[@]}"
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use official-t2va, dataset-fl2va, or custom." >&2
    exit 2
    ;;
esac
