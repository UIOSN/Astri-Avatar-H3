# MiniMax-H3 Natural Half-Body 200-Case Generation

This repository contains the code needed to generate the 200 natural
half-body T2VA cases and evaluate the generated videos.

## Included files

- `scripts/run_natural_halfbody_200cases.py`: builds the 200 prompts and runs
  the jobs sequentially.
- `scripts/generate_ref2va_halfbody_400_prompts.py`: generates 400 prompt-only
  Ref2VA cases from 20 expressions and 20 gestures without starting a model.
- `scripts/infer_h3.py`: submits jobs to the local SGLang `/v1/videos` API and
  downloads completed videos.
- `scripts/serve_sglang.sh`: starts the MiniMax-H3 SGLang server.
- `scripts/install_runtime.sh`: installs the local inference runtime.
- `scripts/download_model.sh`: downloads MiniMax-H3 model files.
- `requirements.txt`: inference dependency declaration.
- `evaluation/syncnet_python/`: CPBD, SyncNet, FID, FVD, ID-Sim, and AU-F1
  benchmark implementation, environment setup, tests, and documentation.

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

## Generate 400 Ref2VA prompts only

```bash
./scripts/generate_ref2va_halfbody_400_prompts.py
```

This command does not start SGLang or call a model. It writes 400 individual
prompt files and a `prompts.jsonl` manifest to
`prompts/ref2va_halfbody_400cases`. Every prompt uses `<Picture 1>` as the
subject identity reference and `<Audio 1>` as the preserved driving speech.
Use `--output-dir` to select another output directory.

## Evaluation

```bash
cd evaluation/syncnet_python
./setup_benchmark_env.sh --cuda
./download_model.sh
```

Prepare a CSV or JSONL manifest following `benchmark_manifest.example.csv`,
then run:

```bash
.venv/bin/python run_benchmark.py \
  --manifest /path/to/manifest.csv \
  --output_dir /path/to/benchmark_outputs \
  --metrics all \
  --device cuda
```

See `evaluation/syncnet_python/BENCHMARK.md` for metric definitions, manifest
fields, sampling controls, and reporting guidance. Evaluation weights and
outputs are downloaded or generated locally and are not committed.
