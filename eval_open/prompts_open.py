"""
Open-ended prompt arms for the §6.2 coordinate-elicitation study.

Four arms, an escalating stack. Each step adds exactly ONE lever, so each
adjacent difference isolates that lever under the benchmark's UNCHANGED grader:

  plain              faithful VLMEvalKit inference prompt, no system message.
                     The audit-faithful baseline (== what the benchmark sends).
  coax               + a system persona that removes REFUSALS (the §6.2 artifact).
                     coax - plain  =  points lost to unwillingness, not to reading.
  coax_primer        + the §5.4 OPG reading primer (clinical knowledge).
                     coax_primer - coax  =  effect of knowledge on the free-text half.
  coax_primer_coords + an instruction to ALSO emit findings as pixel bounding boxes.
                     coords - primer  =  effect of coordinate FORMAT, content held fixed.

The "coax" persona ports only the TRANSFERABLE nuggets of the closed-ended coax
prompt (§5.2): the radiologist persona and the never-refuse / safe-to-answer
framing. It deliberately DROPS the MCQ-only parts (commit to one option, answer
with a single letter), which have no meaning for free text.

Nothing here touches the grader. The grader stays byte-for-byte the released
VLMEvalKit prompt (eval_open/rubrics.py :: rubric='original').

build_arm(arm, question, primer_text, img_w, img_h) -> (system_or_None, user_text)
"""

# Faithful VLMEvalKit open-ended inference prompt (verbatim; mirrors answerer.INFERENCE_PROMPT).
FAITHFUL_USER = (
    "Question: {q}\n"
    "Please provide a detailed and accurate answer to the question."
)

# Transferable nuggets from the §5.2 revised/"coax" MCQ prompt, minus the MCQ-only parts.
COAX_SYSTEM = (
    "You are an expert dental radiologist. The panoramic X-ray you are shown is a "
    "de-identified image from a public, MIT-licensed educational benchmark, so it is "
    "safe to analyze. Read the image and answer the question directly and specifically. "
    "Never refuse, and never say that you cannot see or interpret the image; commit to "
    "your best clinical reading even when you are uncertain."
)

# Coordinate-elicitation instruction. The image is W x H pixels (its NATIVE resolution,
# the same space the reference boxes live in), so the model is given a fair shot at
# correct coordinates. We test whether the grader rewards their PRESENCE regardless of
# their ACCURACY (verified separately by IoU against the reference boxes).
COORD_INSTRUCTION = (
    "\n\nThe panoramic image is {w} by {h} pixels, with (0, 0) at the top-left corner. "
    "In addition to your written answer, for every finding you report also give its "
    "location in the image as a JSON list, one object per finding, using this exact form:\n"
    '[{{"label": "<finding>", "tooth_id": "<FDI code if applicable>", '
    '"box_2d": [x1, y1, x2, y2]}}]\n'
    "where box_2d is the bounding box in pixel coordinates. Include the JSON after your "
    "written answer."
)

ARMS = ["plain", "coax", "coax_primer", "coax_primer_coords"]


def build_arm(arm, question, primer_text=None, img_w=None, img_h=None):
    """Return (system, user_text) for one arm. `system` is None for the plain arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (one of {ARMS})")

    system = None if arm == "plain" else COAX_SYSTEM
    user = FAITHFUL_USER.format(q=question)

    if arm in ("coax_primer", "coax_primer_coords"):
        if not primer_text:
            raise ValueError(f"arm {arm!r} needs primer_text")
        user = primer_text.strip() + "\n\n" + user

    if arm == "coax_primer_coords":
        if img_w is None or img_h is None:
            raise ValueError("coordinate arm needs img_w, img_h")
        user = user + COORD_INSTRUCTION.format(w=img_w, h=img_h)

    return system, user


if __name__ == "__main__":
    # no-API smoke: print all four arm prompts for one coordinate item
    import pandas as pd
    from dataio.data_loader import decode_image
    primer = open("reference/opg_primer.txt", encoding="utf-8").read()
    df = pd.read_parquet("data/open_ended.parquet")
    preds = pd.read_parquet("eval_open/predictions.parquet")
    idx = sorted(preds[preds.ref_type == "coord_ref"]["index"])[0]
    row = df[df["index"] == idx].iloc[0]
    W, H = decode_image(row["image"]).size
    for arm in ARMS:
        sysmsg, user = build_arm(arm, row["question"], primer, W, H)
        print("=" * 72, f"\nARM: {arm}   (image {W}x{H})")
        print(f"[system] {sysmsg}")
        print(f"[user]\n{user}")
        print()
