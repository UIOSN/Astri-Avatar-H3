#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
TMPDIR="${TMPDIR:-/dev/shm/minimax_h3_pip_tmp}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dev/shm/minimax_h3_uv_cache}"
export TMPDIR UV_CACHE_DIR

mkdir -p "$TMPDIR" "$UV_CACHE_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip uv

source "$VENV_DIR/bin/activate"

uv pip install \
  --prerelease=allow \
  --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu129/ \
  "sglang[diffusion]"

# The SGLang resolver currently prefers CUDA 13 wheels. This machine exposes
# CUDA 12.8 through the driver, so pin the PyTorch and SGLang kernels to cu129.
uv pip install --force-reinstall \
  torch==2.11.0 torchaudio==2.11.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu129
uv pip install --force-reinstall sglang-kernel \
  --index-url https://docs.sglang.ai/whl/cu129/
uv pip install --force-reinstall sgl-deep-gemm \
  --index-url https://docs.sglang.ai/whl/cu129/ --no-deps

"$VENV_DIR/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
