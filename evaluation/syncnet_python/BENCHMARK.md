# 视频生成 Benchmark 说明

这个 benchmark 面向 talking-head / audio-driven video generation，例如蒸馏后的 MiniMax-H3。它覆盖 `metrics.txt` 里的 7 个指标：

| 指标 | 类型 | 越高越好 | 输入 |
| --- | --- | --- | --- |
| CPBD | 样本级 | 是 | generated_video |
| SyncNet Conf | 样本级 | 是 | generated_video，视频内需有音频 |
| SyncNet Dist | 样本级 | 否 | generated_video，视频内需有音频 |
| FVD | 数据集级 | 否 | real_video + generated_video |
| ID-Sim | 样本级 | 是 | source_image + generated_video |
| Action Unit F1 | 样本级 | 是 | gt_au_csv/driver_au_csv + pred_au_csv，或 OpenFace |
| FID | 数据集级 | 否 | real_video + generated_video |

## 1. 准备环境

仓库提供了一键环境脚本。默认创建 CPU 环境（可在任何机器复现）：

```bash
cd evaluation/syncnet_python
./setup_benchmark_env.sh
source .venv/bin/activate
```

需要 CUDA 12.4 PyTorch 时使用：

```bash
./setup_benchmark_env.sh --cuda
```

脚本会安装 Python 3.10 和 Python 依赖，并下载 FVD 使用的 TFHub I3D TorchScript 权重。SyncNet 的 `data/syncnet_v2.model` 和 `detectors/s3fd/weights/sfd_face.pth` 需要通过 `download_model.sh` 单独下载；这些模型权重不在 Git 仓库中。

如果要直接从视频提取 AU，还需要安装 OpenFace，并在运行时传入 `--openface_bin /path/to/FeatureExtraction`。如果已经离线生成 OpenFace AU CSV，则不需要 OpenFace 二进制。

## 2. 准备 manifest

支持 CSV 或 JSONL。最小 CSV 示例：

```csv
id,generated_video,real_video,source_image,gt_au_csv,pred_au_csv
sample_0001,/data/gen/0001.mp4,/data/real/0001.mp4,/data/source/0001.jpg,/data/au/real_0001.csv,/data/au/gen_0001.csv
sample_0002,/data/gen/0002.mp4,/data/real/0002.mp4,/data/source/0002.jpg,/data/au/real_0002.csv,/data/au/gen_0002.csv
```

相对路径按 manifest 文件所在目录解析，而不是按运行命令时的当前目录解析。样本 `id` 必须唯一。

字段说明：

- `id`：样本 ID。可选，缺省时会自动生成。
- `generated_video`：模型生成视频，必须有。
- `real_video` / `gt_video`：真实视频。计算 FID/FVD 需要。
- `source_image` / `source_face`：身份参考图。计算 ID-Sim 需要。
- `gt_au_csv` / `driver_au_csv`：目标 AU CSV。计算 AU F1 需要。
- `pred_au_csv`：生成视频的 AU CSV。缺省时可以用 `--openface_bin` 自动生成。
- `driver_video` / `target_video`：目标动作视频。没有 `gt_au_csv` 时可配合 `--openface_bin` 自动生成目标 AU。

## 3. 运行 benchmark

计算全部指标：

```bash
cd evaluation/syncnet_python
python run_benchmark.py \
  --manifest /path/to/manifest.csv \
  --output_dir /path/to/benchmark_out \
  --metrics all \
  --device cuda \
  --overwrite
```

建议正式评测显式固定抽样协议，例如：

```bash
python run_benchmark.py \
  --manifest /path/to/manifest.csv \
  --output_dir /path/to/benchmark_out \
  --metrics all --device cuda \
  --cpbd_max_frames 64 --cpbd_stride 5 \
  --fid_max_frames 128 --fid_stride 4 \
  --fvd_clip_len 16 --fvd_max_clips_per_video 4
```

只计算 SyncNet 和 CPBD：

```bash
python run_benchmark.py \
  --manifest /path/to/manifest.csv \
  --output_dir /path/to/benchmark_out \
  --metrics syncnet,cpbd
```

如果没有 `pred_au_csv`，但有 OpenFace：

```bash
python run_benchmark.py \
  --manifest /path/to/manifest.csv \
  --output_dir /path/to/benchmark_out \
  --metrics au_f1 \
  --openface_bin /path/to/OpenFace/build/bin/FeatureExtraction
```

主要输出：

```text
/path/to/benchmark_out/benchmark_report.json
/path/to/benchmark_out/samples/<id>.json
/path/to/benchmark_out/work/
```

`benchmark_report.json` 里包含：

- `summary`：全局聚合结果，样本级指标会有 mean/median/std/n。
- `dataset_errors`：FID/FVD 这类数据集级指标的失败原因。
- `samples`：每个样本的指标和失败原因。

## 4. 每个指标怎么计算

### CPBD

衡量生成视频的清晰度/模糊程度。

计算流程：

1. 从 `generated_video` 抽帧。
2. 每帧转灰度图。
3. 用 Canny 将 64x64 block 分类为 edge/non-edge block，并用 Sobel 提取待测边缘。
4. 用 Marziliano 方法沿梯度方向测量边缘宽度。
5. 根据 block 对比度选择 3 或 5 像素的 JNB（Just Noticeable Blur）宽度。
6. 用 `P_blur = 1 - exp(-(edge_width / JNB)^3.6)` 计算每条边缘可感知模糊概率。
7. 统计 `P_blur < 0.64` 的边缘概率质量，得到每帧 CPBD。
8. 对所有有效帧取平均，输出 `cpbd`。

输出：

- `cpbd`：平均清晰度分数，越高越清晰。
- `cpbd_std`：帧间标准差。
- `cpbd_frames`：参与计算的帧数。

实现使用 `cpbd==1.0.7` 的标准 edge-width 算法，并兼容新版本 SciPy。CPBD 的 Python 实现逐边缘计算，速度较慢，因此默认每 5 帧采一帧、每个视频最多 64 帧。

### SyncNet Conf / SyncNet Dist / AV Offset

衡量嘴型和音频是否同步。

计算流程：

1. `run_pipeline.py` 把视频转成 25fps，抽 16kHz 单声道音频。
2. S3FD 检测人脸，并跟踪出 face tracks。
3. 每条 face track 裁剪成 224x224 的人脸视频并带上对应音频。
4. `SyncNetInstance.evaluate()` 提取视觉特征和音频 MFCC 特征。
5. 在 `[-vshift, +vshift]` 范围内计算音画特征距离。
6. 取距离最小的位置作为 `syncnet_offset`。
7. `syncnet_dist` 是最佳 offset 的最小距离。
8. `syncnet_conf = median(distance) - min(distance)`。
9. 多个人脸 track 时，选 `syncnet_conf` 最高的 track 作为样本结果。

输出：

- `syncnet_conf`：同步置信度，越高越好。
- `syncnet_dist`：同步距离，越低越好。
- `syncnet_offset`：最佳音画偏移，越接近 0 越好。
- `syncnet_abs_offset`：偏移绝对值，越低越好。
- `syncnet_tracks`：检测到的人脸轨数量。

### FVD

衡量真实视频集合和生成视频集合的整体视频分布差异。

计算流程：

1. 对 manifest 中同时有 `real_video` 和 `generated_video` 的样本抽取视频 clip。
2. 使用 Google FVD 参考实现对应的 TFHub I3D Kinetics-400 模型提取特征；本项目使用与该模型数值对齐的 TorchScript 版本。
3. 分别估计真实视频特征和生成视频特征的均值与协方差。
4. 计算 Fréchet distance：

```text
||mu_real - mu_gen||^2 + Tr(Sigma_real + Sigma_gen - 2 * sqrt(Sigma_real * Sigma_gen))
```

输出：

- `fvd`：越低越好。
- `fvd_real_features` / `fvd_generated_features`：参与计算的 clip 数。
- `fvd_backend`：固定为 `tfhub-i3d-kinetics400-torchscript`。
- `fvd_clip_len`：每个 clip 的帧数。

默认取连续 16 帧、缩放到 224x224，并映射到 `[-1, 1]`。FVD 对抽帧方式、clip 数和分辨率非常敏感；只有 checkpoint、`clip_len`、抽样策略和样本数全部一致时，才能与外部结果比较。正式结果建议至少使用数百个视频；少于两个 clip 时协方差不可估计，程序会明确报错。

### FID

衡量真实帧集合和生成帧集合的图像分布差异。

计算流程：

1. 从 `real_video` 和 `generated_video` 抽帧。
2. 用 `torch-fidelity` 的 FID Inception-V3 权重提取 2048 维 pool3 特征（由 TorchMetrics 调用）。这套权重和常见 FID 实现对齐，不是 torchvision ImageNet 分类权重。
3. 分别估计真实帧和生成帧特征的均值与协方差。
4. 计算 Fréchet distance。

输出：

- `fid`：越低越好。
- `fid_real_features` / `fid_generated_features`：参与计算的帧数。
- `fid_backend`：固定为 `torchmetrics-inception-v3-2048`。

FID 需要足够多且抽样协议一致的真实帧和生成帧。正式比较建议使用数千到数万帧；只用极少帧得到的协方差和分数没有统计稳定性。

### ID-Sim

衡量生成视频是否保持源身份。

计算流程：

1. 用 InsightFace/ArcFace 从 `source_image` 提取身份 embedding。
2. 从 `generated_video` 抽帧。
3. 每帧检测最大人脸并提取 ArcFace `buffalo_l` embedding。
4. 计算 source embedding 和每帧 generated embedding 的 cosine similarity。
5. 对所有有效帧取平均。

输出：

- `id_sim`：平均身份相似度，越高越好。
- `id_sim_std`：帧间标准差。
- `id_sim_frames`：成功检测到人脸的帧数。

### Action Unit F1

衡量生成表情/面部动作是否匹配目标动作。

计算流程：

1. 读取 `gt_au_csv` / `driver_au_csv` 和 `pred_au_csv`。
2. 默认使用 OpenFace 的 AU classification 列，即列名以 `_c` 结尾的列，例如 `AU01_c`。
3. 对齐两份 CSV 的公共 AU 列。
4. 按最短帧数截断，因此两份 CSV 必须使用相同 FPS、相同起始时刻；否则应先重采样/时间戳对齐。
5. 对所有 AU 和所有帧展开，计算 micro precision/recall/F1。

输出：

- `au_f1`：AU 分类 F1，越高越好。
- `au_precision`
- `au_recall`
- `au_frames`
- `au_columns`

如果没有 `pred_au_csv`，脚本可以通过 `--openface_bin` 调 OpenFace 从生成视频提取 AU。如果目标 AU CSV 也没有，可以提供 `driver_video` / `target_video`（或 `real_video`），脚本会自动提取目标 AU。

## 5. 评测协议与统计要求

- CPBD、SyncNet、ID-Sim、AU F1 是样本级指标，报告 mean、median、std、有效样本数 `n`。
- FID、FVD 是集合级指标，不能先逐视频计算再平均。
- FID/FVD 的真实集和生成集必须一一采用同一输入集合，且固定帧率、分辨率、clip 长度和抽样间隔。
- 失败样本不能静默丢弃。报告中的 `sample_metric_failures`、`samples[*].errors` 和 `dataset_errors` 用于审计失败原因。
- SyncNet 只适用于有可见说话人脸和音频的视频；静音、侧脸或脸过小的样本应单独统计失败率。

## 6. 推荐报告方式

对蒸馏 MiniMax-H3，至少报告：

```text
syncnet_conf_mean ↑
syncnet_dist_mean ↓
syncnet_abs_offset_mean ↓
cpbd_mean ↑
id_sim_mean ↑
au_f1_mean ↑
fid ↓
fvd ↓
failure rate ↓
```

对比时应固定：

- 同一份 manifest。
- 同一批输入音频、source image、prompt。
- 同一分辨率、fps、采样帧数。
- 同一随机种子或每个 prompt 多 seed 后平均。
- 同一预处理和 checkpoint。

## 7. 常见失败原因

- SyncNet 无结果：视频里没有可检测的人脸、脸太小、视频太短、没有音频。
- FID/FVD 失败：缺少 `real_video`，或 torchvision 权重无法下载。
- ID-Sim 失败：缺少 `source_image`，InsightFace 未安装，或视频帧中检测不到人脸。
- AU F1 失败：缺少 AU CSV，OpenFace 未安装，或 GT/PRED 没有公共 AU 列。

## 8. 参考实现与论文

- CPBD: Narvekar and Karam, *A No-Reference Image Blur Metric Based on the Cumulative Probability of Blur Detection*, IEEE TIP 2011.
- SyncNet: Chung and Zisserman, *Out of Time: Automated Lip Sync in the Wild*, ACCV Workshops 2016；本仓库模型与实现。
- FID: Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*, NeurIPS 2017；`torch-fidelity` FID Inception 权重。
- FVD: Unterthiner et al., *Towards Accurate Generative Models of Video: A New Metric & Challenges*；Google Research FVD 的 TFHub I3D 特征协议。
- ID-Sim: Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019；InsightFace `buffalo_l`。
- AU: Baltrusaitis et al., *OpenFace 2.0: Facial Behavior Analysis Toolkit*, FG 2018。
