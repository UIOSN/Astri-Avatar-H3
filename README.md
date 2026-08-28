# MiniMax-H3 Natural Half-Body 200-Case Generation

This repository contains only the code needed to generate the 200 natural
half-body T2VA cases.

## Included files

- `scripts/run_natural_halfbody_200cases.py`: builds the 200 prompts and runs
  the jobs sequentially.
- `scripts/infer_h3.py`: submits jobs to the local SGLang `/v1/videos` API and
  downloads completed videos.
- `scripts/serve_sglang.sh`: starts the MiniMax-H3 SGLang server.
- `scripts/install_runtime.sh`: installs the local inference runtime.
- `scripts/download_model.sh`: downloads MiniMax-H3 model files.
- `requirements.txt`: inference dependency declaration.

Generated videos, prompts, logs, model files, datasets, caches, and virtual
environments are excluded by `.gitignore`.

## Requirements

- Linux
- NVIDIA GPUs with enough memory for MiniMax-H3 multi-GPU inference
- CUDA-compatible NVIDIA driver
- Python 3.10 or newer
- `uv` for environment setup

The default scripts use four GPUs. Override the physical GPU list with
`VISIBLE_GPU_IDS`, for example `VISIBLE_GPU_IDS=2,3,4,5`.

## Setup

```bash
./scripts/install_runtime.sh
./scripts/download_model.sh
```

By default, the model is downloaded to `models/MiniMax-H3`. To use another
location, set `MODEL_DIR` when downloading and `MODEL_PATH` when starting the
server.

## Generate 200 cases

```bash
./scripts/run_natural_halfbody_200cases.py
```

The default output directory is `outputs/natural_halfbody_200cases`. Use
`--output-dir`, `--count`, `--start-index`, or `--limit` to control the run.
Use `--no-server` when an SGLang server is already running on the selected
port.

Each case writes an MP4, the exact prompt text, and the exact request JSON.
The prompts use deterministic seeds from `0` through `199`.

## Prompt design

The fixed prompt structure enforces a live-action photorealistic, front-facing
static medium shot with mid-torso framing, visible face and hands, subtle
gestures, synchronized English speech, and clean isolated audio. The 200 cases
vary the subject, clothing, environment, emotion, dialogue, and gesture using
deterministic modular combinations.
