#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PHYSICAL_GPU_IDS="${PHYSICAL_GPU_IDS:-2,3,4,5,6}"
VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-1,2,3,4,5}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4}"
NUM_GPUS="${NUM_GPUS:-5}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-1}"
RING_DEGREE="${RING_DEGREE:-5}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-70000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MODE="${MODE:-dataset-fl2va}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
DURATION="${DURATION:-4}"
SHORT_EDGE="${SHORT_EDGE:-768}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/outputs}"
WATCH_LOG="${WATCH_LOG:-$PROJECT_DIR/server_outputs/wait_and_run_$(date +%Y%m%d_%H%M%S).log}"
INFER_ARGS="${INFER_ARGS:-}"

mkdir -p "$(dirname "$WATCH_LOG")" "$OUT_DIR"

log() {
  printf '%s %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*" | tee -a "$WATCH_LOG"
}

gpu_is_free() {
  local gpu_id="$1"
  local free_mem
  free_mem="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  [[ -n "$free_mem" ]] && (( free_mem >= MIN_FREE_MEM_MB ))
}

selected_summary() {
  local gpu_id
  IFS=',' read -r -a ids <<< "$PHYSICAL_GPU_IDS"
  for gpu_id in "${ids[@]}"; do
    nvidia-smi --id="$gpu_id" --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits
  done | tr '\n' ';'
}

while true; do
  all_free=1
  IFS=',' read -r -a physical_ids <<< "$PHYSICAL_GPU_IDS"
  for gpu_id in "${physical_ids[@]}"; do
    if ! gpu_is_free "$gpu_id"; then
      all_free=0
      break
    fi
  done

  if (( all_free )); then
    log "GPUs free: $(selected_summary)"
    log "Starting MiniMax-H3 inference: MODE=$MODE PHYSICAL_GPU_IDS=$PHYSICAL_GPU_IDS VISIBLE_GPU_IDS=$VISIBLE_GPU_IDS GPU_IDS=$GPU_IDS"
    set +e
    VISIBLE_GPU_IDS="$VISIBLE_GPU_IDS" \
    CHECK_GPUS=0 \
    GPU_IDS="$GPU_IDS" \
    NUM_GPUS="$NUM_GPUS" \
    ULYSSES_DEGREE="$ULYSSES_DEGREE" \
    RING_DEGREE="$RING_DEGREE" \
    MODE="$MODE" \
    SAMPLE_INDEX="$SAMPLE_INDEX" \
    DURATION="$DURATION" \
    SHORT_EDGE="$SHORT_EDGE" \
    OUT_DIR="$OUT_DIR" \
    INFER_ARGS="$INFER_ARGS" \
      "$PROJECT_DIR/scripts/run_multigpu_infer.sh" 2>&1 | tee -a "$WATCH_LOG"
    status=${PIPESTATUS[0]}
    set -e
    log "Inference command exited with status $status"
    exit "$status"
  fi

  log "Waiting; selected GPUs not free enough: $(selected_summary)"
  sleep "$POLL_SECONDS"
done
