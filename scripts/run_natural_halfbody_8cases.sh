#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/outputs/natural_halfbody_8cases}"
PORT="${PORT:-30012}"
VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-2,3,4,5}"
mkdir -p "$OUT_DIR"

declare -a CASES=(
  'A friendly young woman with shoulder-length curly hair, wearing a mustard cardigan, speaks clearly to the camera in a bright home office. She says: "Today I want to share a simple idea that can make everyday work feel easier." She uses a gentle open-palm gesture, then relaxes both hands near her waist.'
  'A middle-aged man with short dark hair and light stubble, wearing a blue denim shirt, speaks calmly in a quiet library. He says: "Let us look at this example carefully and take one small step at a time." He makes a small counting gesture with one hand and nods naturally.'
  'A confident young man with glasses and neatly combed black hair, wearing a charcoal blazer over a white shirt, gives a natural presentation in a modern office. He says: "The main goal is to make this process clear, practical, and easy to follow." He briefly gestures with both hands as if explaining three connected points.'
  'A warm elderly woman with short silver hair, wearing a soft burgundy sweater, speaks naturally in a cozy living room. She says: "I have learned that patience and kindness can change the way we solve difficult problems." She gently brings her hands together, then opens one palm toward the camera.'
  'A young woman with long straight dark hair, wearing a light blue blouse, speaks in a calm conversational tone in a quiet kitchen. She says: "A good conversation begins when we listen carefully and give the other person enough time." She makes a small side-to-side hand gesture and gives a subtle smile.'
  'A middle-aged Black woman with braided hair, wearing a green casual sweater, speaks clearly in a softly lit studio. She says: "This new plan gives everyone a clear role and leaves room for thoughtful decisions." She emphasizes the first phrase with one relaxed hand and then lowers it.'
  'A middle-aged East Asian man with short gray hair, wearing a beige overshirt, speaks professionally in a simple meeting room. He says: "We can improve the result by checking the details before making the final decision." He points gently downward once and follows with a small two-handed explanatory gesture.'
  'A young South Asian woman with wavy hair, wearing a cream knit sweater, speaks enthusiastically but naturally beside a sunlit window. She says: "I am excited to show you how this small change can create a much better result." She opens both hands briefly, nods, and returns to a relaxed posture.'
)

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$PROJECT_DIR"
setsid env \
  PROJECT_DIR="$PROJECT_DIR" \
  GPU_IDS=0,1,2,3 \
  CUDA_VISIBLE_DEVICES="$VISIBLE_GPU_IDS" \
  PORT="$PORT" \
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
  echo "Generating natural half-body case $n/8"
  "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/infer_h3.py" \
    --endpoint "http://127.0.0.1:$PORT" \
    --mode custom --task t2va \
    --prompt "integrated_multimodal_description: Live-action photorealistic video. Single continuous static medium shot, natural half-body framing from mid-torso up, face, mouth, shoulders, and both hands visible throughout. Preserve realistic anatomy and identity. No camera movement, no cuts, no zoom, no subtitles, no logos, no watermark, and no text. ${CASES[$i]} Keep every gesture subtle, spontaneous, anatomically correct, and away from the face and mouth. overall_soundscape: Clean isolated close-microphone human speech only. Clear dry intelligible voice, naturally synchronized with the lips. No background music, no melody, no sound effects, no room tone, no traffic, no wind, no crowd, no appliances, no hiss, no hum, no echo, no reverb, no distortion, no clipping, and no clicks. non_diegetic_music: N/A" \
    --duration 4 --short-edge 768 --seed "$i" \
    --output "$OUT_DIR/case_${n}.mp4" \
    --request-json-out "$OUT_DIR/case_${n}.request.json"
done

echo "Generated 8 clips in $OUT_DIR"
