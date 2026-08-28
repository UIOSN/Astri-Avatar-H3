#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import traceback
from pathlib import Path

from benchmark_metrics import (
    aggregate_numeric,
    compute_au_f1,
    compute_cpbd_video,
    compute_fid,
    compute_fvd,
    compute_id_similarity,
    compute_syncnet,
    read_manifest,
    run_openface,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch benchmark for talking-head/video generation outputs.")
    parser.add_argument("--manifest", required=True, help="CSV or JSONL manifest. See BENCHMARK.md.")
    parser.add_argument("--output_dir", required=True, help="Directory for metrics JSON files.")
    parser.add_argument("--syncnet_root", default=str(Path(__file__).resolve().parent), help="Path to syncnet_python.")
    parser.add_argument("--syncnet_model", default="", help="Path to syncnet_v2.model.")
    parser.add_argument("--work_dir", default="", help="Temporary SyncNet/OpenFace work directory.")
    parser.add_argument("--metrics", default="all", help="Comma list: cpbd,syncnet,fid,fvd,id_sim,au_f1 or all.")
    parser.add_argument("--device", default="", help="Torch device for FID/FVD, e.g. cuda or cpu.")
    parser.add_argument("--openface_bin", default="", help="Optional OpenFace FeatureExtraction binary.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite/recompute reusable intermediate outputs.")
    parser.add_argument("--cpbd_max_frames", type=int, default=64)
    parser.add_argument("--cpbd_stride", type=int, default=5)
    parser.add_argument("--fid_max_frames", type=int, default=128)
    parser.add_argument("--fid_stride", type=int, default=4)
    parser.add_argument("--fvd_clip_len", type=int, default=16)
    parser.add_argument("--fvd_max_clips_per_video", type=int, default=4)
    parser.add_argument("--fvd_model", default="", help="Optional tfhub-I3D TorchScript checkpoint path.")
    return parser.parse_args()


def metric_enabled(enabled, name):
    return "all" in enabled or name in enabled


def fail(errors, name, exc):
    errors[name] = {
        "error": str(exc),
        "traceback": traceback.format_exc(limit=5),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    enabled = {x.strip() for x in args.metrics.split(",") if x.strip()}
    valid_metrics = {"all", "cpbd", "syncnet", "fid", "fvd", "id_sim", "au_f1"}
    unknown_metrics = enabled - valid_metrics
    if unknown_metrics:
        raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown_metrics))}")
    samples = read_manifest(args.manifest)
    sample_ids = [row["id"] for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manifest sample IDs must be unique")
    safe_sample_ids = [re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample" for value in sample_ids]
    if len(safe_sample_ids) != len(set(safe_sample_ids)):
        raise ValueError("Manifest sample IDs collide after filename sanitization")
    device = args.device or None

    sample_results = []
    real_videos = []
    generated_videos = []

    for row, safe_sample_id in zip(samples, safe_sample_ids):
        sample_id = row["id"]
        generated_video = row.get("generated_video") or row.get("gen_video") or row.get("video")
        if not generated_video:
            sample_results.append({
                "id": sample_id,
                "metrics": {},
                "errors": {"sample": {"error": "Missing generated_video"}},
            })
            continue

        generated_video = str(Path(generated_video).expanduser())
        result = {
            "id": sample_id,
            "generated_video": generated_video,
            "metrics": {},
            "errors": {},
        }
        real_video = row.get("real_video") or row.get("gt_video")

        if metric_enabled(enabled, "cpbd"):
            try:
                result["metrics"].update(compute_cpbd_video(
                    generated_video,
                    max_frames=args.cpbd_max_frames,
                    stride=args.cpbd_stride,
                ))
            except Exception as exc:
                fail(result["errors"], "cpbd", exc)

        if metric_enabled(enabled, "syncnet"):
            try:
                result["metrics"].update(compute_syncnet(
                    syncnet_root=args.syncnet_root,
                    video_path=generated_video,
                    reference=safe_sample_id,
                    work_dir=work_dir / "syncnet",
                    initial_model=args.syncnet_model or None,
                    overwrite=args.overwrite,
                ))
            except Exception as exc:
                fail(result["errors"], "syncnet", exc)

        if metric_enabled(enabled, "id_sim"):
            try:
                source_image = row.get("source_image") or row.get("source_face")
                if not source_image:
                    raise RuntimeError("Missing source_image/source_face in manifest")
                result["metrics"].update(compute_id_similarity(source_image, generated_video))
            except Exception as exc:
                fail(result["errors"], "id_sim", exc)

        if metric_enabled(enabled, "au_f1"):
            try:
                gt_au_csv = row.get("gt_au_csv") or row.get("driver_au_csv")
                pred_au_csv = row.get("pred_au_csv")
                if not gt_au_csv:
                    target_video = row.get("driver_video") or row.get("target_video") or real_video
                    if not target_video or not args.openface_bin:
                        raise RuntimeError(
                            "Missing gt_au_csv/driver_au_csv; pass a target video and --openface_bin"
                        )
                    gt_au_csv = run_openface(
                        args.openface_bin, target_video, work_dir / "openface" / f"{safe_sample_id}_target"
                    )
                if not pred_au_csv:
                    if not args.openface_bin:
                        raise RuntimeError("Missing pred_au_csv, or pass --openface_bin to generate it")
                    pred_au_csv = run_openface(
                        args.openface_bin, generated_video, work_dir / "openface" / safe_sample_id
                    )
                result["metrics"].update(compute_au_f1(gt_au_csv, pred_au_csv))
            except Exception as exc:
                fail(result["errors"], "au_f1", exc)

        if real_video:
            real_videos.append(str(Path(real_video).expanduser()))
            generated_videos.append(generated_video)

        sample_results.append(result)
        write_json(output_dir / "samples" / f"{safe_sample_id}.json", result)

    summary = aggregate_numeric(sample_results)
    dataset_errors = {}

    if metric_enabled(enabled, "fid"):
        try:
            if not real_videos:
                raise RuntimeError("FID requires real_video/gt_video for at least one sample")
            summary.update(compute_fid(
                real_videos,
                generated_videos,
                device=device,
                max_frames=args.fid_max_frames,
                stride=args.fid_stride,
            ))
        except Exception as exc:
            fail(dataset_errors, "fid", exc)

    if metric_enabled(enabled, "fvd"):
        try:
            if not real_videos:
                raise RuntimeError("FVD requires real_video/gt_video for at least one sample")
            summary.update(compute_fvd(
                real_videos,
                generated_videos,
                device=device,
                clip_len=args.fvd_clip_len,
                max_clips_per_video=args.fvd_max_clips_per_video,
                model_path=args.fvd_model or None,
            ))
        except Exception as exc:
            fail(dataset_errors, "fvd", exc)

    metric_failures = sum(len(sample.get("errors", {})) for sample in sample_results)
    report = {
        "manifest": args.manifest,
        "num_samples": len(samples),
        "enabled_metrics": sorted(enabled),
        "summary": summary,
        "dataset_errors": dataset_errors,
        "sample_metric_failures": metric_failures,
        "samples": sample_results,
    }
    write_json(output_dir / "benchmark_report.json", report)
    print(f"Wrote {output_dir / 'benchmark_report.json'}")


if __name__ == "__main__":
    main()
