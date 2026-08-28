#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/MiniMax-H3}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR/data/halfbody_dataset_dev0000}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DURATION="${DURATION:-4}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python3
fi

printf "Project: %s\n" "$PROJECT_DIR"
printf "Model dir: %s\n" "$MODEL_DIR"
printf "Dataset: %s\n" "$DATASET_ROOT"
printf "Python: %s\n" "$PYTHON_BIN"

printf "\n[Disk]\n"
df -h "$PROJECT_DIR" "$DATASET_ROOT" || true

printf "\n[GPU]\n"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
else
  echo "nvidia-smi not found"
fi

printf "\n[CUDA allocation]\n"
if command -v nvidia-smi >/dev/null 2>&1; then
  while IFS= read -r gpu_id; do
    printf "GPU %s: " "$gpu_id"
    if CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" - <<'PY' >/tmp/minimax_h3_cuda_probe.out 2>&1
import torch
torch.empty((1,), device="cuda")
print("ok")
PY
    then
      cat /tmp/minimax_h3_cuda_probe.out
    else
      grep -m1 -E "CUDA error|AcceleratorError|RuntimeError" /tmp/minimax_h3_cuda_probe.out || head -1 /tmp/minimax_h3_cuda_probe.out
    fi
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
  echo "skipped"
fi

printf "\n[Python deps]\n"
"$PYTHON_BIN" - <<PY
import importlib.util
for name in ["torch", "huggingface_hub", "sglang"]:
    spec = importlib.util.find_spec(name)
    status = "ok" if spec else "missing"
    print(f"{name}: {status}")
PY

printf "\n[Model files]\n"
if [[ -d "$MODEL_DIR" ]]; then
  du -shL "$MODEL_DIR"
  find -L "$MODEL_DIR" -maxdepth 2 \( -name model_index.json -o -name modular_model_index.json \) | sort
else
  echo "missing: $MODEL_DIR"
fi

printf "\n[Dataset request smoke test]\n"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/infer_h3.py" \
  --mode dataset-fl2va \
  --sample-index 0 \
  --duration "$DURATION" \
  --dry-run \
  --request-json-out /tmp/minimax_h3_preflight_request.json >/tmp/minimax_h3_preflight_stdout.json
"$PYTHON_BIN" - <<PY
import json
p = json.load(open("/tmp/minimax_h3_preflight_request.json"))
summary = {
    "task": p["task"],
    "target": p["target"],
    "conditions": [{k: (v[:32] + "..." if k == "uri" else v) for k, v in c.items()} for c in p["conditions"]],
}
print(summary)
PY
