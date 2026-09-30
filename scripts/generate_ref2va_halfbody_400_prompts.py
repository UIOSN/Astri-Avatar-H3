#!/usr/bin/env python3
"""Generate 400 Ref2VA half-body avatar prompts without running a model."""

import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "prompts" / "ref2va_halfbody_400cases"


EXPRESSIONS = [
    (
        "neutral_attentive",
        "a neutral and attentive expression, with relaxed facial muscles and steady eye contact",
    ),
    (
        "warm_friendly",
        "a warm and friendly expression, with soft eyes and a gentle natural smile",
    ),
    (
        "calm_reassuring",
        "a calm and reassuring expression, with relaxed eyebrows and a subtle smile",
    ),
    (
        "confident_composed",
        "a confident and composed expression, with focused eyes and a restrained smile",
    ),
    (
        "mildly_enthusiastic",
        "a mildly enthusiastic expression, with bright eyes and a small energetic smile",
    ),
    (
        "thoughtful",
        "a thoughtful expression, with attentive eyes and slight concentration in the eyebrows",
    ),
    (
        "serious_approachable",
        "a serious but approachable expression, with calm focused eyes and relaxed facial tension",
    ),
    (
        "empathetic",
        "an empathetic expression, with soft eyes and a caring, understanding look",
    ),
    (
        "curious_engaged",
        "a curious and engaged expression, with slightly raised eyebrows and interested eyes",
    ),
    (
        "patient_instructional",
        "a patient and instructional expression, with attentive eyes and a reassuring half-smile",
    ),
    (
        "optimistic",
        "an optimistic expression, with relaxed eyes and an open, positive smile",
    ),
    (
        "mildly_concerned",
        "a mildly concerned expression, with subtly drawn eyebrows and attentive eyes",
    ),
    (
        "determined",
        "a determined expression, with focused eyes and a firm but natural mouth",
    ),
    (
        "pleasantly_surprised",
        "a pleasantly surprised expression, with slightly widened eyes and a light smile",
    ),
    (
        "relieved",
        "a relieved expression, with softened eyes and a relaxed gentle smile",
    ),
    (
        "sincere",
        "a sincere expression, with steady eye contact and subtle, restrained emotion",
    ),
    (
        "cheerful",
        "a cheerful expression, with lively eyes and a clear but natural smile",
    ),
    (
        "reflective",
        "a reflective expression, with calm eyes and a quiet contemplative look",
    ),
    (
        "encouraging",
        "an encouraging expression, with supportive eye contact and a warm smile",
    ),
    (
        "professional_neutral",
        "a professional and neutral expression, with controlled facial changes and steady eye contact",
    ),
]


GESTURES = [
    (
        "subtle_nod",
        "The subject makes one small natural nod at an appropriate point, then returns to a neutral posture.",
    ),
    (
        "two_separate_nods",
        "The subject makes two subtle nods at separate meaningful moments, without repetitive head bobbing.",
    ),
    (
        "gentle_head_tilt",
        "The subject gently tilts the head while explaining one point, then naturally returns to center.",
    ),
    (
        "slight_forward_lean",
        "The subject leans forward slightly to emphasize one point, then returns to the initial upright posture.",
    ),
    (
        "mostly_still",
        "The subject remains mostly still, using only natural blinking and subtle head and shoulder movement.",
    ),
    (
        "right_open_palm",
        "The subject raises the right hand and makes one small open-palm explanatory gesture in front of the chest, then lowers it naturally.",
    ),
    (
        "left_open_palm",
        "The subject raises the left hand and makes one small open-palm explanatory gesture in front of the chest, then lowers it naturally.",
    ),
    (
        "single_counting_gesture",
        "The subject makes one small counting gesture with one hand to mark an important point, then relaxes the hand.",
    ),
    (
        "downward_emphasis",
        "The subject makes one gentle downward emphasis gesture with one hand, then returns the hand to rest.",
    ),
    (
        "sideward_explanation",
        "The subject makes one small sideward explanatory gesture with one hand, then returns to a centered posture.",
    ),
    (
        "pinch_detail",
        "The subject makes one small pinch-and-release gesture to indicate a precise detail, then relaxes the hand.",
    ),
    (
        "hand_near_chest",
        "The subject briefly brings one hand near the chest to convey sincerity, then lowers it naturally.",
    ),
    (
        "both_palms_open",
        "The subject gently opens both palms once in a welcoming explanatory gesture, then returns both hands to rest.",
    ),
    (
        "symmetric_explanation",
        "The subject makes one small symmetrical explanatory motion with both hands near chest level, then relaxes them.",
    ),
    (
        "hands_together_then_open",
        "The subject briefly brings both hands together, then opens them slightly to present the idea before relaxing.",
    ),
    (
        "both_hands_raise",
        "The subject raises both hands slightly for one moment of emphasis, then lowers them smoothly.",
    ),
    (
        "clasp_then_explain",
        "The subject keeps both hands loosely clasped at first, then makes one slow explanatory gesture and returns them to rest.",
    ),
    (
        "alternating_hands",
        "The subject gestures once with the right hand and later once with the left hand, keeping both motions small and distinct.",
    ),
    (
        "open_then_nod",
        "The subject makes one gentle open-palm gesture with both hands, lowers them, and later gives one small natural nod.",
    ),
    (
        "right_gesture_then_smile",
        "The subject makes one small right-hand explanatory gesture, returns the hand to rest, and finishes with a subtle warm smile.",
    ),
]


def build_prompt(expression, gesture):
    return f"""subject_definitions:
<Subject 1> is the speaking person in the target video. The subject's identity and facial features are referenced from <Picture 1>.
<Picture 1> is the identity and facial-feature reference for <Subject 1>.
<Audio 1> is the driving speech audio for <Subject 1>, providing the complete spoken content, original voice, speaking pace, pauses, rhythm, and emotional timing.

summary:
[reference generation + audio reuse] The target video generates a natural single-person half-body digital-human speaking performance. The identity and facial features of <Subject 1> are referenced from <Picture 1>, while the half-body framing, body posture, and movements are newly generated. <Subject 1> speaks according to <Audio 1>, with facial expressions and body movements synchronized to the audio.

retention_analysis:
<Subject 1> (present throughout the target video): partially_preserved - The subject's identity and facial features remain consistent with <Picture 1>, while the half-body pose, hand gestures, and composition are newly generated for the target video.
<Picture 1> (identity reference for <Subject 1>): attribute_transfer - The facial structure, facial features, and overall identity of the person in the reference image are transferred to <Subject 1>.
<Audio 1> (driving speech audio for <Subject 1>): fully_preserved - The spoken content, original voice, speaking pace, pauses, and rhythm are fully preserved and used to drive the subject's lip movements, expressions, and actions.

detailed_description:
The target video presents a realistic single-person half-body digital-human speaking performance.

[Shot 1] The camera presents <Subject 1> in a front-facing half-body medium shot. The subject is centered and looks naturally toward the camera, with the head, shoulders, upper body, arms, and hands visible. The identity and facial features of <Subject 1> are referenced from <Picture 1>. The body, pose, hands, and half-body composition may be newly generated even when they are not visible in <Picture 1>. The subject speaks according to <Audio 1>, with accurate lip synchronization.

Facial expression: {expression}. The subject blinks naturally, and the eyebrows, eyes, cheeks, and corners of the mouth change subtly according to the meaning, tone, emphasis, pauses, and rhythm of <Audio 1>.

Body movement: {gesture} Natural subtle head and shoulder movement accompanies the speech. The body remains balanced and stable throughout.

All expressions and movements should be natural, fluid, restrained, and synchronized with <Audio 1>. Keep the identity and facial features of <Subject 1> stable. Avoid facial drift, identity changes, inaccurate lip synchronization, malformed hands, extra fingers, sudden movements, exaggerated acting, or meaningless repeated gestures. Keep the camera stable. Do not add cuts, camera movement, subtitles, titles, logos, watermarks, or other text.

overall_soundscape:
Fully preserve the original speech in <Audio 1>. Do not regenerate, replace, rewrite, or repeat the spoken content. Do not alter the speaker's voice, speaking pace, pauses, or timing. Do not add additional voices or prominent environmental sounds beyond those already present in <Audio 1>.

non_diegetic_music:
N/A"""


def build_cases():
    cases = []
    case_index = 0
    for gesture_index, (gesture_id, gesture) in enumerate(GESTURES):
        for expression_index, (expression_id, expression) in enumerate(EXPRESSIONS):
            case_index += 1
            cases.append(
                {
                    "index": case_index - 1,
                    "id": f"case_{case_index:03d}",
                    "expression_index": expression_index,
                    "expression_id": expression_id,
                    "expression": expression,
                    "gesture_index": gesture_index,
                    "gesture_id": gesture_id,
                    "gesture": gesture,
                    "prompt": build_prompt(expression, gesture),
                }
            )
    return cases


def write_cases(output_dir, cases):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "prompts.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for case in cases:
            prompt_path = output_dir / f"{case['id']}.prompt.txt"
            prompt_path.write_text(case["prompt"] + "\n", encoding="utf-8")
            manifest.write(json.dumps(case, ensure_ascii=False) + "\n")

    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cases = build_cases()

    expected_count = len(EXPRESSIONS) * len(GESTURES)
    if len(cases) != expected_count or expected_count != 400:
        raise RuntimeError(
            f"Expected 20 expressions x 20 gestures = 400 prompts, got {len(cases)}"
        )
    if len({case["prompt"] for case in cases}) != 400:
        raise RuntimeError("Prompt combinations are not unique")

    manifest_path = write_cases(args.output_dir, cases)
    print(f"Generated {len(cases)} unique prompts")
    print(f"Prompt directory: {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
