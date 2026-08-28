#!/usr/bin/env python3
"""Submit MiniMax-H3 video jobs to a local SGLang /v1/videos endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "halfbody_dataset_dev0000"

OFFICIAL_T2VA_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Cinematic, medium wide shot, "
    "pushing in slowly. In the cavernous, dimly lit bridge of a starship, sleek "
    "metallic consoles with glowing amber displays flank a massive, curved "
    "observation window. A female captain, in her late 40s with an athletic build "
    "and short silver-streaked black hair, stands in the center midground. She "
    "wears a structured, high-collared dark navy military tunic with silver chest "
    "insignias. Her back is to the camera, silhouetted against the cool ambient "
    "starlight. Outside the window, a massive armada of dark grey dreadnoughts "
    "hovers in formation against a deep purple space nebula. The fleet thrusters "
    "build to an intense blue light. [Shot 2] At 00:04.500, the camera cuts to a "
    "close-up of the captain's face and shakes strongly. A blinding white flash "
    "floods through the window as the fleet jumps to hyperspace; the bridge "
    "rattles violently, then falls back to a hollow room tone.\n"
    "overall_soundscape: low resonant ship hum, escalating electronic whine, "
    "deafening bass-heavy boom, metallic rattles, then abrupt silence.\n"
    "non_diegetic_music: cinematic space-opera orchestral score, slow tempo, "
    "mournful French horn over sustained string dissonances."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30010")
    parser.add_argument(
        "--mode",
        choices=("official-t2va", "dataset-fl2va", "custom"),
        default="dataset-fl2va",
    )
    parser.add_argument("--task", choices=("t2va", "fl2va", "ref2va"), default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--first-frame", type=Path, default=None)
    parser.add_argument("--last-frame", type=Path, default=None)
    parser.add_argument("--reference-image", type=Path, default=None)
    parser.add_argument("--reference-video", type=Path, default=None)
    parser.add_argument("--reference-audio", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--short-edge", type=int, default=768)
    parser.add_argument("--aspect-ratio", default="auto")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/h3_result.mp4"))
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=60 * 60)
    parser.add_argument("--request-json-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build the request and exit without POSTing.")
    return parser.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt
    if args.mode == "official-t2va":
        return OFFICIAL_T2VA_PROMPT
    return (
        "integrated_multimodal_description: Generate a realistic half-body talking "
        "person video from the provided first frame. Keep the identity, framing, "
        "lighting, clothing, and background consistent. The person should make "
        "natural subtle head, shoulder, and facial movements while speaking.\n"
        "overall_soundscape: clean close-mic human speech with a quiet indoor room tone."
    )


def data_uri(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def load_dataset_sample(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest or args.dataset_root / "manifests" / "passed.jsonl"
    with manifest.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == args.sample_index:
                return json.loads(line)
    raise IndexError(f"sample-index {args.sample_index} not found in {manifest}")


def build_conditions(args: argparse.Namespace, sample: dict[str, Any] | None) -> list[dict[str, Any]]:
    first_frame = args.first_frame
    if sample and not first_frame:
        first_frame = Path(sample["first_frame_path"])

    conditions: list[dict[str, Any]] = []
    if first_frame:
        conditions.append(
            {
                "type": "image",
                "uri": data_uri(first_frame),
                "role": "keyframe",
                "frame_index": 0,
            }
        )
    if args.last_frame:
        conditions.append(
            {
                "type": "image",
                "uri": data_uri(args.last_frame),
                "role": "keyframe",
                "frame_index": -1,
            }
        )
    if args.reference_image:
        conditions.append({"type": "image", "uri": data_uri(args.reference_image), "role": "reference"})
    if args.reference_video:
        conditions.append({"type": "video", "uri": data_uri(args.reference_video), "role": "reference"})
    if args.reference_audio:
        conditions.append({"type": "audio", "uri": data_uri(args.reference_audio), "role": "reference"})
    return conditions


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    sample = load_dataset_sample(args) if args.mode == "dataset-fl2va" else None
    task = args.task
    if task is None:
        task = "t2va" if args.mode == "official-t2va" else "fl2va"
    duration = args.duration
    if duration is None:
        duration = float(sample.get("duration", 8.0)) if sample else 10.0
        duration = max(4.0, min(15.0, duration))

    payload = {
        "task": task,
        "prompt": read_prompt(args),
        "conditions": build_conditions(args, sample),
        "target": {
            "short_edge": args.short_edge,
            "aspect_ratio": "16:9" if args.mode == "official-t2va" else args.aspect_ratio,
            "duration_seconds": duration,
        },
        "seed": args.seed,
    }
    return payload


def request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}: {body}") from exc


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response:
        output.write_bytes(response.read())


def main() -> int:
    args = parse_args()
    endpoint = args.endpoint.rstrip("/")
    payload = build_payload(args)

    if args.request_json_out:
        args.request_json_out.parent.mkdir(parents=True, exist_ok=True)
        args.request_json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    created = request_json("POST", f"{endpoint}/v1/videos", payload)
    video_id = created.get("id")
    if not video_id:
        print(json.dumps(created, ensure_ascii=False, indent=2), file=sys.stderr)
        raise RuntimeError("Response does not contain video id")

    deadline = time.time() + args.timeout
    status = None
    while time.time() < deadline:
        state = request_json("GET", f"{endpoint}/v1/videos/{video_id}")
        status = state.get("status")
        print(json.dumps({"id": video_id, "status": status}, ensure_ascii=False), flush=True)
        if status in {"completed", "succeeded", "success"}:
            download(f"{endpoint}/v1/videos/{video_id}/content", args.output)
            print(f"saved {args.output}")
            return 0
        if status in {"failed", "error", "cancelled", "canceled"}:
            print(json.dumps(state, ensure_ascii=False, indent=2), file=sys.stderr)
            return 1
        time.sleep(args.poll_interval)

    raise TimeoutError(f"Timed out waiting for {video_id}; last status={status}")


if __name__ == "__main__":
    raise SystemExit(main())
