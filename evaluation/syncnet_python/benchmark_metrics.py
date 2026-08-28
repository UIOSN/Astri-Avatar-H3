#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy import linalg


def run_checked(cmd, **kwargs):
    proc = subprocess.run(cmd, text=True, capture_output=True, **kwargs)
    if proc.returncode != 0:
        detail = "\n".join(x for x in [proc.stdout.strip(), proc.stderr.strip()] if x)
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{detail}")
    return proc


def read_manifest(path):
    path = Path(path)
    rows = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    for idx, row in enumerate(rows):
        if not row.get("id"):
            row["id"] = f"sample_{idx:05d}"
        row["id"] = str(row["id"])

    path_fields = {
        "generated_video", "gen_video", "video", "real_video", "gt_video",
        "source_image", "source_face", "gt_au_csv", "driver_au_csv",
        "pred_au_csv", "driver_video", "target_video",
    }
    base_dir = path.resolve().parent
    for row in rows:
        for field in path_fields:
            value = row.get(field)
            if not value:
                continue
            value_path = Path(str(value)).expanduser()
            if not value_path.is_absolute():
                value_path = base_dir / value_path
            row[field] = str(value_path.resolve())
    return rows


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def video_frames(video_path, max_frames=None, stride=1, resize=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames


def frechet_distance(features_a, features_b, eps=1e-6):
    features_a = np.asarray(features_a, dtype=np.float64)
    features_b = np.asarray(features_b, dtype=np.float64)
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("Frechet features must be 2-D arrays")
    if features_a.shape[0] < 2 or features_b.shape[0] < 2:
        raise ValueError("Frechet distance requires at least two real and two generated features")
    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError("Real and generated feature dimensions do not match")
    mu_a = np.mean(features_a, axis=0)
    mu_b = np.mean(features_b, axis=0)
    sigma_a = np.cov(features_a, rowvar=False)
    sigma_b = np.cov(features_b, rowvar=False)

    if sigma_a.ndim == 0:
        sigma_a = np.array([[sigma_a]])
    if sigma_b.ndim == 0:
        sigma_b = np.array([[sigma_b]])

    try:
        covmean, _ = linalg.sqrtm(sigma_a.dot(sigma_b), disp=False)
    except TypeError:
        # SciPy 1.16 removed the legacy disp argument and returns only the matrix.
        covmean = linalg.sqrtm(sigma_a.dot(sigma_b))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_a + offset).dot(sigma_b + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu_a - mu_b
    distance = float(diff.dot(diff) + np.trace(sigma_a + sigma_b - 2.0 * covmean))
    return max(distance, 0.0)


@lru_cache(maxsize=1)
def _cpbd_compute_function():
    # cpbd 1.0.7 imports the removed scipy.ndimage.imread even though array
    # evaluation never uses it. Supplying the legacy symbol keeps the original
    # CPBD implementation usable with current SciPy.
    import scipy.ndimage

    if not hasattr(scipy.ndimage, "imread"):
        scipy.ndimage.imread = None
    from cpbd import compute

    return compute


def compute_cpbd_frame(gray):
    if gray.ndim != 2:
        raise ValueError("CPBD expects a grayscale frame")
    return float(_cpbd_compute_function()(gray.astype(np.uint8, copy=False)))


def compute_cpbd_video(video_path, max_frames=256, stride=1):
    scores = []
    for frame in video_frames(video_path, max_frames=max_frames, stride=stride):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        score = compute_cpbd_frame(gray)
        if not np.isnan(score):
            scores.append(score)
    if not scores:
        raise RuntimeError("CPBD failed: no valid edge frames")
    return {
        "cpbd": float(np.mean(scores)),
        "cpbd_std": float(np.std(scores)),
        "cpbd_frames": int(len(scores)),
    }


def _iter_frame_batches(video_paths, max_frames, stride, batch_size):
    import torch

    batch = []
    for video_path in video_paths:
        for frame in video_frames(video_path, max_frames=max_frames, stride=stride):
            batch.append(torch.from_numpy(frame.copy()).permute(2, 0, 1))
            if len(batch) >= batch_size:
                yield torch.stack(batch)
                batch = []
    if batch:
        yield torch.stack(batch)


FVD_MODEL_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
FVD_MODEL_SHA256 = "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_fvd_model(model_path=None):
    model_path = Path(model_path or Path(__file__).resolve().parent / "data" / "i3d_torchscript.pt")
    if model_path.exists() and _sha256(model_path) == FVD_MODEL_SHA256:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".download")
    urllib.request.urlretrieve(FVD_MODEL_URL, tmp_path)
    if _sha256(tmp_path) != FVD_MODEL_SHA256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded I3D checkpoint checksum does not match")
    os.replace(tmp_path, model_path)
    return model_path


@lru_cache(maxsize=4)
def _load_fvd_model(model_path, device):
    import torch

    return torch.jit.load(str(model_path), map_location=device).eval()


def extract_fvd_features(video_paths, clip_len=16, max_clips_per_video=4, device=None, model_path=None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ensure_fvd_model(model_path)
    model = _load_fvd_model(str(model_path), str(device))
    feats = []

    with torch.no_grad():
        for video_path in video_paths:
            frames = video_frames(video_path, max_frames=clip_len * max_clips_per_video, stride=1)
            if len(frames) < clip_len:
                continue
            clips = []
            for start in range(0, len(frames) - clip_len + 1, clip_len):
                clip = np.stack(frames[start:start + clip_len], axis=0)
                clips.append(torch.from_numpy(clip.copy()).permute(3, 0, 1, 2))
            if clips:
                x = torch.stack(clips).to(device=device, dtype=torch.uint8)
                features = model(x, rescale=True, resize=True, return_features=True)
                feats.append(features.detach().cpu().numpy())

    if not feats:
        raise RuntimeError("No FVD features extracted")
    return np.concatenate(feats, axis=0)


def compute_fid(real_videos, generated_videos, device=None, max_frames=128, stride=4):
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.hub.set_dir(str(Path(__file__).resolve().parent / ".cache" / "torch"))
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    real_count = 0
    generated_count = 0
    for batch in _iter_frame_batches(real_videos, max_frames, stride, batch_size=32):
        metric.update(batch.to(device=device, dtype=torch.uint8), real=True)
        real_count += batch.shape[0]
    for batch in _iter_frame_batches(generated_videos, max_frames, stride, batch_size=32):
        metric.update(batch.to(device=device, dtype=torch.uint8), real=False)
        generated_count += batch.shape[0]
    if real_count < 2 or generated_count < 2:
        raise RuntimeError("FID requires at least two real and two generated frames")
    return {
        "fid": max(float(metric.compute().detach().cpu().item()), 0.0),
        "fid_real_features": int(real_count),
        "fid_generated_features": int(generated_count),
        "fid_backend": "torchmetrics-inception-v3-2048",
    }


def compute_fvd(real_videos, generated_videos, device=None, clip_len=16, max_clips_per_video=4, model_path=None):
    real_features = extract_fvd_features(
        real_videos, clip_len=clip_len, max_clips_per_video=max_clips_per_video,
        device=device, model_path=model_path,
    )
    gen_features = extract_fvd_features(
        generated_videos, clip_len=clip_len, max_clips_per_video=max_clips_per_video,
        device=device, model_path=model_path,
    )
    return {
        "fvd": frechet_distance(real_features, gen_features),
        "fvd_real_features": int(real_features.shape[0]),
        "fvd_generated_features": int(gen_features.shape[0]),
        "fvd_backend": "tfhub-i3d-kinetics400-torchscript",
        "fvd_clip_len": int(clip_len),
    }


@lru_cache(maxsize=2)
def _get_face_analysis(use_cuda):
    import insightface

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    model_root = Path(__file__).resolve().parent / ".cache" / "insightface"
    app = insightface.app.FaceAnalysis(name="buffalo_l", root=str(model_root), providers=providers)
    app.prepare(ctx_id=0 if use_cuda else -1, det_size=(640, 640))
    return app


def _detect_faces_with_context(app, image):
    faces = app.get(image)
    if faces:
        return faces
    # Tight talking-head crops can remove the context required by RetinaFace.
    # Replicated padding preserves the face pixels while restoring that context.
    pad = max(20, int(round(min(image.shape[:2]) * 0.2)))
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return app.get(padded)


def compute_id_similarity(source_image, generated_video, max_frames=32, stride=5):
    try:
        import insightface
    except ImportError as exc:
        raise RuntimeError("ID-Sim requires insightface. Install requirements-benchmark.txt.") from exc

    import onnxruntime

    use_cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    app = _get_face_analysis(use_cuda)

    src = cv2.imread(str(source_image))
    if src is None:
        raise RuntimeError(f"Cannot read source image: {source_image}")
    src_faces = _detect_faces_with_context(app, src)
    if not src_faces:
        raise RuntimeError("No face found in source image")
    src_face = max(src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    src_emb = src_face.normed_embedding

    sims = []
    for frame in video_frames(generated_video, max_frames=max_frames, stride=stride):
        faces = _detect_faces_with_context(app, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not faces:
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        sims.append(float(np.dot(src_emb, face.normed_embedding)))

    if not sims:
        raise RuntimeError("No faces found in generated video for ID-Sim")
    return {
        "id_sim": float(np.mean(sims)),
        "id_sim_std": float(np.std(sims)),
        "id_sim_frames": int(len(sims)),
    }


def _read_au_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty AU CSV: {path}")
    raw_cols = [c for c in rows[0].keys() if c.strip().endswith("_c")]
    if not raw_cols:
        raw_cols = [c for c in rows[0].keys() if c.strip().startswith("AU")]
    cols = [c.strip() for c in raw_cols]
    data = []
    for row in rows:
        data.append([float(row[c]) for c in raw_cols])
    return cols, np.asarray(data, dtype=np.float32)


def run_openface(openface_bin, video_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(openface_bin),
        "-f", str(video_path),
        "-out_dir", str(output_dir),
        "-aus",
    ]
    run_checked(cmd)
    csvs = sorted(output_dir.glob("*.csv"))
    if not csvs:
        raise RuntimeError(f"OpenFace produced no CSV in {output_dir}")
    return str(csvs[0])


def compute_au_f1(gt_au_csv, pred_au_csv):
    gt_cols, gt = _read_au_csv(gt_au_csv)
    pred_cols, pred = _read_au_csv(pred_au_csv)
    common = [c for c in gt_cols if c in pred_cols]
    if not common:
        raise RuntimeError("No common AU classification columns")

    gt_idx = [gt_cols.index(c) for c in common]
    pred_idx = [pred_cols.index(c) for c in common]
    n = min(gt.shape[0], pred.shape[0])
    gt = (gt[:n, gt_idx] > 0.5).astype(np.int32)
    pred = (pred[:n, pred_idx] > 0.5).astype(np.int32)

    tp = int(np.logical_and(gt == 1, pred == 1).sum())
    fp = int(np.logical_and(gt == 0, pred == 1).sum())
    fn = int(np.logical_and(gt == 1, pred == 0).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    return {
        "au_f1": float(f1),
        "au_precision": float(precision),
        "au_recall": float(recall),
        "au_frames": int(n),
        "au_columns": common,
    }


def compute_syncnet(syncnet_root, video_path, reference, work_dir, initial_model=None, batch_size=20, vshift=15, overwrite=False):
    syncnet_root = Path(syncnet_root)
    work_dir = Path(work_dir)
    initial_model = initial_model or syncnet_root / "data" / "syncnet_v2.model"

    if str(syncnet_root) not in sys.path:
        sys.path.insert(0, str(syncnet_root))

    crop_dir = work_dir / "pycrop" / reference
    if overwrite or not crop_dir.exists() or not list(crop_dir.glob("0*.avi")):
        cmd = [
            sys.executable,
            str(syncnet_root / "run_pipeline.py"),
            "--videofile", str(video_path),
            "--reference", reference,
            "--data_dir", str(work_dir),
            "--overwrite",
        ]
        run_checked(cmd, cwd=str(syncnet_root))

    from SyncNetInstance import SyncNetInstance

    class Opt:
        pass

    opt = Opt()
    opt.batch_size = int(batch_size)
    opt.vshift = int(vshift)
    opt.reference = reference
    opt.tmp_dir = str(work_dir / "pytmp")

    model = SyncNetInstance()
    model.loadParameters(str(initial_model))

    tracks = []
    for crop in sorted(crop_dir.glob("0*.avi")):
        offset, conf, dists = model.evaluate(opt, videofile=str(crop))
        mdist = np.asarray(dists).mean(axis=0)
        min_dist = float(np.min(mdist))
        tracks.append({
            "crop": str(crop),
            "offset": int(np.asarray(offset).item()),
            "confidence": float(np.asarray(conf).item()),
            "min_dist": min_dist,
        })

    if not tracks:
        raise RuntimeError("SyncNet failed: no face tracks found")

    best = max(tracks, key=lambda x: x["confidence"])
    return {
        "syncnet_conf": best["confidence"],
        "syncnet_dist": best["min_dist"],
        "syncnet_offset": best["offset"],
        "syncnet_abs_offset": abs(best["offset"]),
        "syncnet_tracks": len(tracks),
        "syncnet_track_details": tracks,
    }


def aggregate_numeric(sample_results):
    primary_metrics = {
        "cpbd",
        "syncnet_conf",
        "syncnet_dist",
        "syncnet_offset",
        "syncnet_abs_offset",
        "id_sim",
        "au_f1",
        "au_precision",
        "au_recall",
    }
    values = {}
    for sample in sample_results:
        metrics = sample.get("metrics", {})
        for key, value in metrics.items():
            if key not in primary_metrics:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                values.setdefault(key, []).append(float(value))

    summary = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(arr))
        summary[f"{key}_median"] = float(np.median(arr))
        summary[f"{key}_std"] = float(np.std(arr))
        summary[f"{key}_n"] = int(arr.size)
    return summary
