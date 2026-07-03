"""
Open-ended grading rubrics for MMOral-OPG.

Two rubrics, built to be a clean A/B ablation of the coordinate-reward bias
(finding F-open-1, RESEARCH_PLAN.md §3.9):

- ORIGINAL : the 9-shot GPT-4 grading prompt, VERBATIM from VLMEvalKit
             (vlmeval/dataset/utils/mmoral_opg.py :: build_mmoral_opg_gpt4_prompt).
             Examples 7-9 (tooth #31) award a plain-correct answer "Crown" only 0.8,
             but the SAME answer plus pixel coordinates 0.9-1.0. Since none of those
             questions ask for coordinates, this silently rewards models that emit
             them (e.g. the authors' OralGPT).

- REPHRASED: identical to ORIGINAL except (a) one explicit scoring rule is added,
             and (b) the 3 coordinate examples are rescored to be FORMAT-INVARIANT
             (prose == prose+coords). The 6 non-coordinate examples are byte-for-byte
             the same, so any score difference between the two rubrics is attributable
             ONLY to coordinate handling.

build_grading_prompt(question, ground_truth, prediction, rubric=...) returns the full
judge prompt with the real (question | gt | prediction | ) row appended and a trailing
blank Correctness cell, exactly as VLMEvalKit does at run time.
"""

# --- shared instruction (verbatim from VLMEvalKit) --------------------------
INSTRUCTION = (
    "Given the question, compare the ground truth and prediction from AI\n"
    "models, to generate a correctness score for the prediction.\n"
    "The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5,\n"
    "0.6, 0.7, 0.8, 0.9, or 1.0 (totally right).\n"
    "Just complete the last space of the correctness score."
)

# The one added rule that de-biases the rephrased rubric. It is the ONLY prose
# added; everything else in REPHRASED is either verbatim-shared or a rescored
# version of the 3 coordinate rows.
DEBIAS_RULE = (
    "Scoring rule: judge ONLY the clinical and diagnostic correctness and "
    "completeness of the prediction relative to the ground truth. Coordinates, "
    "bounding boxes, and pixel positions are NOT required unless the question "
    "explicitly asks for them; never add or subtract points based on the presence, "
    "absence, or formatting of coordinate/location data."
)

TABLE_HEADER = "Question | Ground truth | Prediction | Correctness\n--- | --- | --- | ---"

# --- the 6 non-coordinate examples (VERBATIM, shared by both rubrics) --------
SHARED_EXAMPLES = [
    ("How many teeth are visualized in the radiograph?",
     "30 teeth are visualized with clear anatomical definition.",
     "30", "1.0"),
    ("How many teeth are visualized in the radiograph?",
     "30 teeth are visualized with clear anatomical definition.",
     "29 teeth are visualized with clear anatomical definition.", "0.0"),
    ("What is the status of the wisdom teeth in the radiograph?",
     "Three wisdom teeth are detected, all of which are impacted: #18, #28, and #48.",
     "#18: impacted, #28: impacted, #48: erupted", "0.7"),
    ("What is the condition of the teeth #26 and #14?",
     "Teeth #26 and #14 show signs of periapical abscesses.",
     "Teeth #26 and #23 show signs of periapical abscesses.", "0.5"),
    ("What is the condition of the bone architecture and visible structures in the jaw?",
     "No apparent bone loss is observed. Bilateral mandibular canals and maxillary sinuses are clearly visible.",
     "Bilateral mandibular canals and maxillary sinuses are clearly visible.", "0.5"),
    ("What is the clinical priority concern regarding the periapical lesions?",
     "Periapical cysts at #11 and #12, and granuloma at #46 require endodontic evaluation.",
     "Periapical lesions at #11, #12, and #46 require endodontic evaluation.", "0.8"),
]

# --- the 3 coordinate examples ----------------------------------------------
_COORD_Q = "What radiographic features are visible in tooth #31 on the panoramic X-ray?"
_COORD_GT = ('[{"Teeth position": {"point_2d": [1242, 726]}}, '
             '{"Crown": {"box_2d": [1220, 637, 1266, 741]}}]')

# ORIGINAL: the coordinate reward — plain "Crown" 0.8, +coords 0.9/1.0.
ORIGINAL_COORD_EXAMPLES = [
    (_COORD_Q, _COORD_GT, "Crown", "0.8"),
    (_COORD_Q, _COORD_GT, "Crown at position: [1230, 627, 1276, 750]", "0.9"),
    (_COORD_Q, _COORD_GT,
     'Teeth at position: {"point_2d": [1242, 726]}, '
     '{Crown at position: {"box_2d": [1230, 627, 1276, 750]}}', "1.0"),
]

# REPHRASED: same 3 items, rescored to be format-invariant. The correct clinical
# finding (a crown on #31) earns full credit whether or not coordinates are given;
# a wrong feature anchors the bottom. Coordinates change nothing.
REPHRASED_COORD_EXAMPLES = [
    (_COORD_Q, _COORD_GT, "Crown", "1.0"),
    (_COORD_Q, _COORD_GT, "Crown at position: [1230, 627, 1276, 750]", "1.0"),
    (_COORD_Q, _COORD_GT, "Root canal treatment", "0.0"),
]


def _format_examples(examples):
    return "\n".join(f"{q} | {gt} | {pred} | {score}" for q, gt, pred, score in examples)


def _prep_gt(ground_truth):
    """Mirror VLMEvalKit: space out <AND>/<OR> tokens in the ground truth."""
    return str(ground_truth).replace("<AND>", " <AND> ").replace("<OR>", " <OR> ")


def build_grading_prompt(question, ground_truth, prediction, rubric="original"):
    """Return the full judge prompt for one (question, gt, prediction) triple.

    rubric: 'original' (verbatim 9-shot) or 'rephrased' (coordinate-debiased).
    """
    if rubric == "original":
        # VLMEvalKit order: the 6 shared rows, then the 3 coordinate rows last.
        instruction = INSTRUCTION
        examples = SHARED_EXAMPLES + ORIGINAL_COORD_EXAMPLES
    elif rubric == "rephrased":
        instruction = INSTRUCTION + "\n" + DEBIAS_RULE
        examples = SHARED_EXAMPLES + REPHRASED_COORD_EXAMPLES
    else:
        raise ValueError(f"unknown rubric {rubric!r} (expected 'original' or 'rephrased')")

    gt = _prep_gt(ground_truth)
    real_row = f"{question} | {gt} | {prediction} | "
    return (
        f"{instruction}\n\n"
        f"{TABLE_HEADER}\n"
        f"{_format_examples(examples)}\n"
        f"{real_row}"
    )


if __name__ == "__main__":
    # sanity: show both prompts on the canonical coordinate case
    print("=" * 30, "ORIGINAL", "=" * 30)
    print(build_grading_prompt(_COORD_Q, _COORD_GT, "Crown", "original"))
    print("\n" + "=" * 30, "REPHRASED", "=" * 30)
    print(build_grading_prompt(_COORD_Q, _COORD_GT, "Crown", "rephrased"))
