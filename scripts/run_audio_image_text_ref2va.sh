#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-2,3,4,5}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
DURATION="${DURATION:-4}"
SHORT_EDGE="${SHORT_EDGE:-768}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
OUTPUT="${OUTPUT:-$PROJECT_DIR/outputs/audio_image_text_ref2va.mp4}"

if [[ -z "${IMAGE:-}" || -z "${AUDIO:-}" ]]; then
  echo "Usage: IMAGE=/path/to/image.png AUDIO=/path/to/audio.wav PROMPT='...' $0" >&2
  exit 2
fi

prompt_file="${PROMPT_FILE:-$PROJECT_DIR/outputs/audio_image_text_ref2va_prompt.txt}"
if [[ -z "${PROMPT_FILE:-}" ]]; then
  PROMPT="${PROMPT:-integrated_multimodal_description: Generate a realistic audiovisual video preserving the reference image identity and using the reference audio as the soundtrack.}"
  mkdir -p "$(dirname "$prompt_file")"
  printf "%s\n" "$PROMPT" >"$prompt_file"
fi

"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/patch_runtime_encoder_device.py"

infer_args=(
  --task ref2va
  --duration "$DURATION"
  --short-edge "$SHORT_EDGE"
  --aspect-ratio "$ASPECT_RATIO"
  --reference-image "$IMAGE"
  --reference-audio "$AUDIO"
  --output "$OUTPUT"
  --prompt-file "$prompt_file"
)

VISIBLE_GPU_IDS="$VISIBLE_GPU_IDS" \
CHECK_GPUS=0 \
GPU_IDS="$GPU_IDS" \
NUM_GPUS="$NUM_GPUS" \
MODE=custom \
INFER_ARGS="${infer_args[*]}" \
EXTRA_ARGS="--quantization fp8" \
"$PROJECT_DIR/scripts/run_multigpu_infer.sh"
