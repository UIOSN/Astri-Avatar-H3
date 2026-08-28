# SyncNet

This repository contains the demo for the audio-to-video synchronisation network (SyncNet). This network can be used for audio-visual synchronisation tasks including: 
1. Removing temporal lags between the audio and visual streams in a video;
2. Determining who is speaking amongst multiple faces in a video. 

Please cite the paper below if you make use of the software. 

## Dependencies

GPU (CUDA):
```
conda env create -f environment.yml
```

CPU only:
```
conda env create -f environment-cpu.yml
```

The code automatically detects and uses a CUDA GPU if available, and falls back to CPU otherwise.

## Video Generation Benchmark

The repository also includes a complete benchmark runner for CPBD, SyncNet confidence/distance, FID, FVD, identity similarity, and Action Unit F1. See [BENCHMARK.md](BENCHMARK.md) for environment setup, manifest format, metric definitions, and reporting protocol.

```bash
./setup_benchmark_env.sh
.venv/bin/python run_benchmark.py \
  --manifest benchmark_manifest.example.csv \
  --output_dir benchmark_outputs/run_001 \
  --metrics all --device cpu
```


## Getting Started

Download the pretrained model:
```
sh download_model.sh
```

## Demo

SyncNet demo:
```
python demo_syncnet.py --videofile data/example.avi --tmp_dir /path/to/temp/directory
```

Check that this script returns approximately the following values (minor differences are expected depending on your platform and package versions):
```
AV offset:      3 
Min dist:       5.353
Confidence:     10.021
```

## Full Pipeline

Run the three stages — face detection and tracking, sync offset estimation, and visualisation:
```
python run_pipeline.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_syncnet.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_visualise.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
```

Outputs:
```
$DATA_DIR/pycrop/$REFERENCE/*.avi - cropped face tracks
$DATA_DIR/pyavi/$REFERENCE/video_out.avi - output video (as shown below)
```
<p align="center">
  <img src="img/ex1.jpg" width="45%"/>
  <img src="img/ex2.jpg" width="45%"/>
</p>

## Publications
 
```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```
