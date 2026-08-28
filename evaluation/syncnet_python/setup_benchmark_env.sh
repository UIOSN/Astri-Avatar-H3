#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
cd "${ROOT_DIR}"

if [[ "${1:-}" == "--cuda" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--cuda]" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv python install 3.10
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv "${VENV_DIR}" --python 3.10
fi
uv pip install --python "${VENV_DIR}/bin/python" \
  torch==2.5.1 torchvision==0.20.1 --index-url "${TORCH_INDEX}"
uv pip install --python "${VENV_DIR}/bin/python" -r "${ROOT_DIR}/requirements-benchmark.txt"

"${VENV_DIR}/bin/python" -c \
  "from benchmark_metrics import ensure_fvd_model; print('I3D:', ensure_fvd_model())"
"${VENV_DIR}/bin/python" -m py_compile \
  "${ROOT_DIR}/benchmark_metrics.py" "${ROOT_DIR}/run_benchmark.py"

echo "Benchmark environment ready: ${VENV_DIR}"
