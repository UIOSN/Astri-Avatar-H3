#!/usr/bin/env python3
"""Generate 200 diverse, front-facing, natural half-body speech clips."""

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_DIR / "outputs" / "natural_halfbody_200cases"

PERSONS = [
    "a young woman with shoulder-length curly brown hair",
    "a young man with short dark hair and light stubble",
    "a middle-aged woman with straight dark hair and glasses",
    "a middle-aged man with short gray hair",
    "an elderly woman with short silver hair",
    "an elderly man with neatly combed white hair",
    "a young woman with long straight black hair",
    "a young man with wavy brown hair",
    "a middle-aged woman with braided hair",
    "a middle-aged man with a short beard and glasses",
]

CLOTHES = [
    "a mustard cardigan over a white shirt",
    "a blue denim shirt",
    "a charcoal blazer over a light shirt",
    "a burgundy knit sweater",
    "a light blue blouse",
    "a green casual sweater",
    "a beige overshirt",
    "a cream knit sweater",
    "a navy polo shirt",
    "a rust-colored button-up shirt",
]

ENVIRONMENTS = [
    "a bright home office",
    "a quiet library",
    "a modern office",
    "a cozy living room",
    "a clean kitchen",
    "a softly lit studio",
    "a simple meeting room",
    "a room beside a sunlit window",
    "a calm classroom",
    "a quiet community center",
]

EMOTIONS = [
    ("calm and reassuring", "a gentle smile and relaxed eyebrows"),
    ("friendly and upbeat", "bright eyes and a warm smile"),
    ("focused and thoughtful", "attentive eyes and a slightly concentrated expression"),
    ("mildly enthusiastic", "a small confident smile and lively but controlled eyes"),
    ("professional and confident", "steady eye contact and a composed expression"),
    ("warm and empathetic", "soft eyes and a caring, relaxed smile"),
    ("curious and engaged", "slightly raised eyebrows and an interested expression"),
    ("relieved and optimistic", "a relaxed smile and an easy, open expression"),
    ("serious but approachable", "calm eyes and a sincere, measured expression"),
    ("patient and instructional", "an attentive expression and a reassuring smile"),
]

GESTURES = [
    "open both palms gently while introducing the idea, then relax both hands near the waist",
    "make a small counting gesture with one hand, followed by one natural nod",
    "use both hands in a small symmetrical explanatory motion near chest level",
    "briefly bring both hands together, then open one palm toward the camera",
    "make a small side-to-side gesture with one hand while keeping the other relaxed",
    "emphasize one phrase with a gentle downward motion of one hand, then lower it",
    "raise both hands slightly to introduce the point, then return them to a relaxed position",
    "make a small pinch-and-release gesture to illustrate a detail, without moving toward the face",
    "gesture once with an open right hand and once with an open left hand, then remain still",
    "keep both hands loosely clasped at first, then make one slow explanatory gesture",
]

DIALOGUES = [
    "Today I want to share a simple idea that can make everyday work feel easier.",
    "Let us look at this example carefully and take one small step at a time.",
    "The main goal is to make this process clear, practical, and easy to follow.",
    "Patience and kindness can change the way we solve difficult problems.",
    "A good conversation begins when we listen carefully and give the other person enough time.",
    "This new plan gives everyone a clear role and leaves room for thoughtful decisions.",
    "We can improve the result by checking the details before making the final decision.",
    "I am excited to show you how this small change can create a much better result.",
    "A quiet routine can help us stay focused when the day becomes busy.",
    "The best solutions usually begin with a clear question and a careful observation.",
    "There is no need to rush because steady progress is often the most reliable approach.",
    "I would like to explain why this choice is useful in everyday situations.",
    "When we work together, each small contribution can improve the final outcome.",
    "This example shows how a simple plan can make a complicated task more manageable.",
    "It is helpful to pause, consider the options, and then choose the next step.",
    "I appreciate your attention, and I hope this explanation makes the idea easier to understand.",
    "A thoughtful habit can save time and make ordinary decisions feel more comfortable.",
    "We can start with the basics and build confidence through a few simple exercises.",
    "The important thing is to remain curious, flexible, and open to learning.",
    "Thank you for listening. I am happy to walk through the idea one step at a time.",
]


def make_prompt(index):
    person = PERSONS[index % len(PERSONS)]
    clothes = CLOTHES[(index * 3) % len(CLOTHES)]
    environment = ENVIRONMENTS[(index * 7) % len(ENVIRONMENTS)]
    emotion, expression = EMOTIONS[(index * 5) % len(EMOTIONS)]
    gesture = GESTURES[(index * 7) % len(GESTURES)]
    dialogue = DIALOGUES[(index * 11) % len(DIALOGUES)]
    return (
        "integrated_multimodal_description: Live-action photorealistic video. "
        "Strictly front-facing natural half-body portrait in a single continuous static medium shot. "
        f"The subject is {person}, wearing {clothes}, in {environment}. "
        "Frame from mid-torso up. The subject faces the camera lens directly throughout, "
        "with the head centered and the shoulders and chest square to the camera. "
        "Both eyes, the complete face, mouth, shoulders, and both hands remain visible. "
        "The camera is at eye level on the subject's central axis. "
        f"Explicit emotion: {emotion}. Facial expression: {expression}. "
        "Speaking style: clear, natural, conversational English at a moderate pace. "
        f'The subject says exactly: "{dialogue}" '
        f"Explicit gesture sequence: {gesture}. "
        "The gesture is subtle, ordinary, anatomically correct, and coordinated with the speech. "
        "Keep every hand movement in front of the torso and away from the face and mouth. "
        "Do not turn the head, shoulders, or torso. Never show a profile or three-quarter view. "
        "No looking away, no looking up or down, no camera movement, no cuts, no zoom, "
        "no sudden pose changes, no exaggerated acting, no subtitles, no captions, "
        "no logos, no watermark, and no text.\n"
        "overall_soundscape: Clean isolated close-microphone human speech only. "
        "The voice is dry, centered, clear, intelligible, and naturally synchronized with the lips. "
        "No background music, no melody, no sound effects, no room tone, no traffic, no wind, "
        "no crowd, no appliances, no hiss, no hum, no echo, no reverb, no distortion, "
        "no clipping, and no clicks.\n"
        "non_diegetic_music: N/A"
    )


def write_manifest(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        case_id = f"case_{index + 1:03d}"
        rows.append({"index": index, "id": case_id, "seed": index, "prompt": make_prompt(index)})
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def wait_health(endpoint, process, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        if process.poll() is not None:
            raise RuntimeError(f"SGLang exited with code {process.returncode}")
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for {endpoint}")


def append_jsonl(path, payload):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--port", type=int, default=30013)
    parser.add_argument("--visible-gpus", default="2,3,4,5")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-server", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = write_manifest(args.output_dir / "prompts.jsonl", args.count)
    rows = rows[args.start_index :]
    if args.limit > 0:
        rows = rows[: args.limit]
    endpoint = f"http://127.0.0.1:{args.port}"
    server = None
    log = (args.output_dir / "server.log").open("w", encoding="utf-8")
    if not args.no_server:
        env = os.environ.copy()
        env.update(
            {
                "PROJECT_DIR": str(PROJECT_DIR),
                "GPU_IDS": "0,1,2,3",
                "CUDA_VISIBLE_DEVICES": args.visible_gpus,
                "PORT": str(args.port),
                "EXTRA_ARGS": "--quantization fp8 --text-encoder-cpu-offload",
            }
        )
        server = subprocess.Popen(
            [str(PROJECT_DIR / "scripts" / "serve_sglang.sh")],
            cwd=PROJECT_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_health(endpoint, server, timeout=1800)

    client = PROJECT_DIR / ".venv" / "bin" / "python"
    infer = PROJECT_DIR / "scripts" / "infer_h3.py"
    log_path = args.output_dir / "generation_log.jsonl"
    failures = 0
    try:
        for row in rows:
            output = args.output_dir / f"{row['id']}.mp4"
            if output.exists() and output.stat().st_size > 0:
                append_jsonl(log_path, {"id": row["id"], "status": "skipped", "output": str(output)})
                continue
            prompt_path = args.output_dir / f"{row['id']}.prompt.txt"
            prompt_path.write_text(row["prompt"], encoding="utf-8")
            command = [
                str(client), str(infer), "--endpoint", endpoint, "--mode", "custom", "--task", "t2va",
                "--prompt-file", str(prompt_path), "--duration", "4", "--short-edge", "768",
                "--seed", str(row["seed"]), "--output", str(output),
                "--request-json-out", str(args.output_dir / f"{row['id']}.request.json"),
            ]
            print(f"Generating {row['id']}", flush=True)
            result = subprocess.run(command, text=True)
            status = "completed" if result.returncode == 0 else "failed"
            if status == "failed":
                failures += 1
            append_jsonl(log_path, {"id": row["id"], "status": status, "returncode": result.returncode, "output": str(output)})
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
        log.close()
    if failures:
        raise SystemExit(f"{failures} generation jobs failed; see {log_path}")


if __name__ == "__main__":
    main()
