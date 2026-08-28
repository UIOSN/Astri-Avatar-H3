#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="$PROJECT_DIR/outputs/english_cases"
PORT="${PORT:-30011}"
mkdir -p "$OUT_DIR"

declare -a CASES=(
  "A friendly presenter speaks fluent English to the camera and says: \"Hello everyone. Today I will explain how this system works.\" Natural facial expressions and subtle hand gestures."
  "A calm teacher speaks clear American English and says: \"Welcome to the lesson. We will start with a simple example.\" Soft indoor lighting and natural head movement."
  "A technology reporter speaks fluent English and says: \"This new device is designed to make everyday tasks faster and easier.\" Realistic studio presentation, subtle gestures."
  "A doctor speaks clear English and says: \"Please take a deep breath and relax. We will begin the examination now.\" Quiet clinic, natural professional expression."
  "A news anchor speaks fluent British English and says: \"Here is the latest update from the city center. Traffic is moving normally this morning.\" Realistic newsroom lighting."
  "A tour guide speaks natural English and says: \"On your left, you can see the oldest building in this historic district.\" Warm daylight, gentle hand gesture."
  "A customer support representative speaks clear English and says: \"Thank you for calling. I will help you solve this problem step by step.\" Friendly expression and subtle movement."
  "A scientist speaks fluent English and says: \"Our results show a clear improvement after the second experiment.\" Modern laboratory, calm explanatory gestures."
  "A young professional speaks natural English and says: \"I am excited to share this project with the team today.\" Bright office, realistic facial motion."
  "An elderly storyteller speaks clear English and says: \"When I was young, this neighborhood looked completely different.\" Cozy room, warm light, gentle expressive movement."
)

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$PROJECT_DIR"
setsid env PROJECT_DIR="$PROJECT_DIR" GPU_IDS=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 PORT="$PORT" \
  EXTRA_ARGS="--quantization fp8 --text-encoder-cpu-offload" \
  "$PROJECT_DIR/scripts/serve_sglang.sh" >"$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + 1800))
until curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -80 "$OUT_DIR/server.log"; exit 1; }
  (( SECONDS < deadline )) || { tail -80 "$OUT_DIR/server.log"; exit 1; }
  sleep 10
done

for i in "${!CASES[@]}"; do
  n=$(printf '%02d' $((i + 1)))
  echo "Generating English case $n/10"
  "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/infer_h3.py" \
    --endpoint "http://127.0.0.1:$PORT" --mode custom --task t2va \
    --prompt "integrated_multimodal_description: ${CASES[$i]}" \
    --duration 4 --short-edge 768 --seed "$i" \
    --output "$OUT_DIR/case_${n}_raw.mp4" \
    --request-json-out "$OUT_DIR/case_${n}.request.json"
done
