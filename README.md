# MiniMax-H3 多卡本地推理部署

## Repository Layout

The runnable inference scripts are under `scripts/`. The batch generator is
`scripts/run_natural_halfbody_200cases.py`; it expects a local MiniMax-H3 model
under `models/MiniMax-H3` and writes generated media under `outputs/`.

The reproducible evaluation implementation is under
`evaluation/syncnet_python/`. It includes CPBD, SyncNet, FID, FVD, ID-Sim and
AU-F1 benchmark code. Downloaded weights, datasets, videos and benchmark
outputs are intentionally excluded from this repository.

See `evaluation/syncnet_python/BENCHMARK.md` for the manifest format and metric
commands. Paths in the examples below refer to the original development
machine; replace them with the clone path on your machine.

本目录是在 `/ephemeral/yexin/workspace/project/minimax_h3` 下整理好的
MiniMax-H3 本地推理交付。部署方式参考官方 Hugging Face 页面：

```text
https://huggingface.co/MiniMaxAI/MiniMax-H3
```

官方推荐使用 SGLang 的 `/v1/videos` 接口进行本地服务化推理。这里已经准备好
FL2VA 权重、SGLang 运行环境、多卡服务脚本和推理客户端脚本。

## 当前状态

- 运行环境：`/ephemeral/yexin/workspace/project/minimax_h3/.venv`
- 推理框架：SGLang diffusion runtime
- 已验证 Python 包：`torch 2.11.0+cu129`、`sglang 0.5.17.dev20260805`
- 已下载模型：FL2VA
- 模型实际位置：`/dev/shm/minimax_h3_models/MiniMax-H3`
- 项目内模型入口：`models/MiniMax-H3 -> /dev/shm/minimax_h3_models/MiniMax-H3`
- 默认数据集：`/ephemeral/yexin/workspace/dataset/seamless_interaction/halfbody_dataset_dev0000`

注意：`/dev/shm` 是临时内存盘，机器重启后模型可能消失。若模型丢失，见下方
“重新下载模型”。

## 最短使用方式

进入目录：

```bash
cd /ephemeral/yexin/workspace/project/minimax_h3
```

当前已验证可直接出视频的多卡命令是 4 卡 T2VA smoke。默认使用物理 GPU
`2,3,4,5`，它们会映射成服务内的 `0,1,2,3`：

```bash
./scripts/run_verified_t2va.sh
```

输出：

```text
outputs/smoke_t2va_fp8_4gpu_patched_4s.mp4
```

如果要换物理 GPU：

```bash
VISIBLE_GPU_IDS=3,4,5,6 GPU_IDS=0,1,2,3 ./scripts/run_verified_t2va.sh
```

数据集第 0 条样本的 FL2VA 入口也已提供：

```bash
GPU_IDS=auto GPU_COUNT=4 MODE=dataset-fl2va SAMPLE_INDEX=0 ./scripts/run_multigpu_infer.sh
```

指定 GPU 做 FL2VA：

```bash
GPU_IDS=0,1,2,3 MODE=dataset-fl2va SAMPLE_INDEX=0 ./scripts/run_multigpu_infer.sh
```

如果 CUDA ordinal 和 `nvidia-smi` index 不一致，先用 `VISIBLE_GPU_IDS`
限制物理卡，再用相对编号启动。当前机器上这个方式可以避开 GPU 0/1
的 CUDA/ECC 问题：

```bash
VISIBLE_GPU_IDS=2,3,4,5 CHECK_GPUS=0 GPU_IDS=0,1,2,3 \
  MODE=official-t2va EXTRA_ARGS="--quantization fp8" ./scripts/run_multigpu_infer.sh
```

运行官方风格 T2VA case：

```bash
GPU_IDS=auto GPU_COUNT=4 MODE=official-t2va ./scripts/run_multigpu_infer.sh
```

输出默认写到：

```text
/ephemeral/yexin/workspace/project/minimax_h3/outputs
```

后台等待 GPU 空闲后自动跑：

```bash
cd /ephemeral/yexin/workspace/project/minimax_h3
tmux new-session -d -s minimax_h3_watch \
  "cd /ephemeral/yexin/workspace/project/minimax_h3 && ./scripts/wait_and_run_gpu_free.sh"
```

默认 watcher 等待物理 GPU `2,3,4,5,6` 都至少有 70GB 空闲，然后用
`VISIBLE_GPU_IDS=1,2,3,4,5 GPU_IDS=0,1,2,3,4 NUM_GPUS=5 ULYSSES_DEGREE=1 RING_DEGREE=5`
启动 5 卡推理。可以通过环境变量覆盖这些值。

查看 watcher：

```bash
tmux attach -t minimax_h3_watch
tmux capture-pane -pt minimax_h3_watch -S -80
```

## 主要脚本

- `scripts/run_multigpu_infer.sh`
  一键多卡推理入口：启动 SGLang、等待服务 ready、提交推理请求、下载 mp4、退出时停止服务。

- `scripts/serve_sglang.sh`
  只启动 SGLang 多卡服务，适合手动调试或长期服务。

- `scripts/infer_h3.py`
  Python 推理客户端，负责构造 `/v1/videos` 请求、轮询状态、下载结果。

- `scripts/run_dataset_fl2va.sh`
  使用 `halfbody_dataset_dev0000/manifests/passed.jsonl` 中的样本做 first-frame FL2VA。

- `scripts/run_official_t2va.sh`
  使用脚本内置的官方风格 prompt 做 T2VA。

- `scripts/preflight.sh`
  检查磁盘、GPU、CUDA allocation、Python 包、模型文件和数据集请求构造。

- `scripts/wait_and_run_gpu_free.sh`
  后台等待指定 GPU 空闲，然后自动调用一键多卡推理脚本。

- `scripts/run_verified_t2va.sh`
  已验证的 4 卡 T2VA 一键推理脚本，默认 `fp8`、开启 text encoder CPU
  offload，并自动应用本地 runtime 补丁。

- `scripts/patch_runtime_encoder_device.py`
  修补 SGLang 0.5.17 MiniMax-H3 Qwen3-VL encoder 在 FSDP CPU offload 下
  的输入设备选择问题。重装 `.venv` 后可重复运行。

## GPU 要求

MiniMax-H3 FL2VA 官方本地示例使用：

```text
--num-gpus 4 --ulysses-degree 4 --model-variant fl2va
```

本机是 H100 80GB。实际测试中，4 卡加载 FL2VA 时需要开启 text encoder CPU offload：

```bash
EXTRA_ARGS="--text-encoder-cpu-offload"
```

一键脚本默认 `LOW_MEMORY=1`，会自动加上这个参数。已验证的 T2VA 路径还需要：

```bash
EXTRA_ARGS="--quantization fp8"
```

H3 的 attention heads 数量是 56，因此常规 Ulysses degree 需要能整除 56。这里推荐：

- 4 张空闲 GPU：推荐路径
- 8 张空闲 GPU：也可用
- 6 张 GPU：不推荐，会因为 56 不能被 6 整除而失败；Ring attention
  又被 MiniMax-H3 packed multi-segment attention 禁用

一键脚本默认会检查：

- 每张选中的 GPU 能否实际分配 CUDA tensor
- 每张选中的 GPU 是否至少有 70000 MiB 空闲显存

可调参数：

```bash
MIN_FREE_MEM_MB=65000 GPU_IDS=auto GPU_COUNT=4 ./scripts/run_multigpu_infer.sh
CHECK_GPUS=0 GPU_IDS=0,1,2,3 ./scripts/run_multigpu_infer.sh
```

## 手动启动服务

FL2VA：

```bash
cd /ephemeral/yexin/workspace/project/minimax_h3
GPU_IDS=0,1,2,3 MODEL_VARIANT=fl2va PORT=30010 \
  EXTRA_ARGS="--quantization fp8 --text-encoder-cpu-offload" \
  ./scripts/serve_sglang.sh
```

服务地址：

```text
http://127.0.0.1:30010/v1/videos
```

另开一个 shell 做数据集样本推理：

```bash
cd /ephemeral/yexin/workspace/project/minimax_h3
ENDPOINT=http://127.0.0.1:30010 SAMPLE_INDEX=0 ./scripts/run_dataset_fl2va.sh
```

官方 T2VA case：

```bash
ENDPOINT=http://127.0.0.1:30010 ./scripts/run_official_t2va.sh
```

## 数据集推理参数

默认读取：

```text
/ephemeral/yexin/workspace/dataset/seamless_interaction/halfbody_dataset_dev0000/manifests/passed.jsonl
```

常用参数：

```bash
SAMPLE_INDEX=3 DURATION=4 SHORT_EDGE=768 \
  ENDPOINT=http://127.0.0.1:30010 \
  ./scripts/run_dataset_fl2va.sh
```

`DURATION` 默认是 4 秒，用于 smoke test。MiniMax-H3 官方范围是 4 到 15 秒。

## 自定义推理

使用自定义首帧图片做 FL2VA：

```bash
cd /ephemeral/yexin/workspace/project/minimax_h3
.venv/bin/python scripts/infer_h3.py \
  --endpoint http://127.0.0.1:30010 \
  --mode custom \
  --task fl2va \
  --first-frame /path/to/frame.png \
  --prompt "integrated_multimodal_description: Generate a realistic talking person video from the provided first frame." \
  --duration 4 \
  --short-edge 768 \
  --output outputs/custom_fl2va.mp4
```

只生成请求 JSON，不访问服务：

```bash
.venv/bin/python scripts/infer_h3.py \
  --mode dataset-fl2va \
  --sample-index 0 \
  --duration 4 \
  --dry-run \
  --request-json-out outputs/sample0.request.json
```

本地图片、音频、视频会被编码为 Base64 data URI 放进请求体，因此不需要额外启动静态文件服务。

## 重新安装环境

```bash
cd /ephemeral/yexin/workspace/project/minimax_h3
./scripts/install_runtime.sh
```

安装脚本默认使用：

```text
/home/ubuntu/miniconda3/envs/longcat-video/bin/python
```

创建项目 venv，并安装 SGLang diffusion runtime、PyTorch CUDA 12.9 wheels 和相关 kernel 包。

## 重新下载模型

FL2VA：

```bash
cd /ephemeral/yexin/workspace/project/minimax_h3
VARIANT=FL2VA MODEL_DIR=/dev/shm/minimax_h3_models/MiniMax-H3 ./scripts/download_model.sh
```

Ref2VA：

```bash
VARIANT=Ref2VA MODEL_DIR=/dev/shm/minimax_h3_models/MiniMax-H3 ./scripts/download_model.sh
```

如果模型放到其他路径，启动时指定：

```bash
MODEL_PATH=/path/to/MiniMax-H3 GPU_IDS=0,1,2,3 ./scripts/serve_sglang.sh
```

## 预检查

```bash
cd /ephemeral/yexin/workspace/project
minimax_h3/scripts/preflight.sh
```

预检查会输出：

- `/ephemeral` 和数据集磁盘情况
- 每张 GPU 的显存占用
- 每张 GPU 是否能实际分配 CUDA tensor
- `torch`、`huggingface_hub`、`sglang` 是否可 import
- 模型文件是否存在
- 数据集 sample 0 的请求体是否可构造

## 当前机器上的已知限制

### 时长和分辨率

- `duration_seconds` 只能是 `4` 到 `15`，不能用 `2s` 做 smoke test。
  传 `2` 会在 SGLang 的 MiniMax-H3 请求校验阶段直接失败，进不到生成阶段。
- `short_edge` 必须是 `768`。传 `512` 会直接被校验拒绝。

### 最近实测结论

- 4 卡、`duration=4`、`short_edge=768`、`--quantization fp8`、开启
  `--text-encoder-cpu-offload`，并应用 `patch_runtime_encoder_device.py`：
  T2VA 已成功生成 mp4，验证输出为
  `outputs/smoke_t2va_fp8_4gpu_patched_4s.mp4`。
- 4 卡、`duration=4`、`short_edge=768`、bf16、`--text-encoder-cpu-offload`：
  服务可以启动，但生成阶段 OOM。每卡约 78GB 已占用，额外申请约 1.45GB 时失败。
- 4 卡、`duration=4`、`short_edge=768`、`--quantization fp8`、开启
  `--text-encoder-cpu-offload`，未应用本地补丁：
  服务可以启动，transformer 常驻显存降到约 32GB/卡；T2VA 会触发
  CPU/CUDA tensor 混用错误。
- 应用本地补丁后，T2VA 可跑通；但 FL2VA 数据集样本仍会在 Qwen3-VL
  keyframe 视觉编码里触发 `torch.Tensor` 和 `DTensor` 混用错误。
- 4 卡、`duration=4`、`short_edge=768`、`--quantization fp8`、关闭
  `--text-encoder-cpu-offload`：
  设备混用问题消失，但加载 transformer 时仍 OOM，GPU 只剩约 160MiB
  时还需要额外申请约 498MiB。
- `--vae-cpu-offload` 或 DiT layerwise offload 当前也会触发 DTensor 相关错误，
  不建议作为稳定方案。

因此，OOM 不是因为 prompt 太长，也不能通过 2s 测试绕过；当前主要瓶颈是
MiniMax-H3 在 `768`、最短合法 `4s` 下的模型常驻显存和生成峰值。目前已跑通
的是 T2VA 多卡推理；FL2VA 的视觉 encoder 分布式路径还需要进一步修 SGLang
runtime。
