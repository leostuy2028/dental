"""
5-image accurate-overlay CEILING test (reproduces results/open/overlay_needle.csv).

Reads the COMMITTED hand-placed FDI overlays in results/open/overlay_ceiling_5img/
and measures plain vs overlay on gemini-3.5-flash: all of each image's questions
batched in one call, graded under the UNCHANGED rubric='original' (GPT-4o judge).

FIDELITY NOTES (this is exactly what produced the committed CSV, warts and all):
  * The overlay note is passed as `detection_text`, so it flows through
    run_batched.build_user()'s EXISTING "TOOTH CHART ... produced by a tooth-detection
    tool" wrapper -- build_user was NOT modified for this test. (The RESEARCH_PLAN
    100-image plan proposes a cleaner one-line note instead; that is a DELIBERATE
    difference, so the future blind run will not be prompt-identical to this ceiling run.)
  * 016640 is EXCLUDED: disrupted dentition, could not be hand-numbered accurately.
  * CEILING ONLY: the overlays were hand-placed KNOWING the answer key
    (see results/open/overlay_ceiling_5img/README.md). No answer text reaches the model
    (numbers only) and all scores are real model output, but numbering was not blind.
  * NONDETERMINISTIC: gemini-3.5-flash + the GPT-4o judge are not fixed-seed, so exact
    percentages vary run to run (observed aggregate deltas ranged +23..+27). Re-running
    OVERWRITES overlay_needle.csv with a fresh (slightly different) draw.

  python -m eval_open.test_overlay_ceiling
"""
import base64, os, re
import pandas as pd

import eval_open.run_batched as rb
from eval_open.prompts_open import COAX_SYSTEM
from eval_open.judges import grade
from eval_open.rubrics import build_grading_prompt

DATA = "data/open_ended.parquet"
PRIMER = "reference/opg_primer.txt"
OVDIR = "results/open/overlay_ceiling_5img"
OUT = "results/open/overlay_needle.csv"
NOTE = ("The green numbers printed on this X-ray are FDI tooth codes labeling each tooth; "
        "use them to state which tooth a finding is on.")
IMAGES = ["016690.jpg", "016713.jpg", "016726.jpg", "016655.jpg"]  # 016640 excluded (disrupted)


def main():
    rb.REASONING_EFFORT = "minimal"
    primer = open(PRIMER, encoding="utf-8").read()
    df = pd.read_parquet(DATA)
    rows = []
    for img in IMAGES:
        g = df[df.image_name == img].sort_values("index")
        ov = os.path.join(OVDIR, img.replace(".jpg", "_overlay.jpg"))
        ovb64 = base64.b64encode(open(ov, "rb").read()).decode()
        qs = g["question"].tolist()
        # plain arm = original image from the parquet; overlay arm = committed overlay + note
        a_plain, _ = rb.answer_image(g.iloc[0]["image"], qs, primer, COAX_SYSTEM, "gemini", "gemini-3.5-flash")
        a_ov, _ = rb.answer_image(ovb64, qs, primer, COAX_SYSTEM, "gemini", "gemini-3.5-flash", detection_text=NOTE)
        for i, (_, r) in enumerate(g.iterrows()):
            sp, _ = grade(build_grading_prompt(r["question"], str(r["answer"]), a_plain[i], "original"), judge="gpt-4o")
            so, _ = grade(build_grading_prompt(r["question"], str(r["answer"]), a_ov[i], "original"), judge="gpt-4o")
            rows.append({"img": img[:6], "teeth": "Teeth" in str(r["category"]),
                         "isloc": bool(re.search(r"which (tooth|teeth)", str(r["question"]), re.I)),
                         "plain": sp, "ov": so})
    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.to_csv(OUT, index=False)

    def s(sub, n):
        if len(sub):
            print(f"  {n:16s} n={len(sub):2d}  {sub.plain.mean()*100:5.1f}% -> {sub.ov.mean()*100:5.1f}%  "
                  f"({(sub.ov.mean()-sub.plain.mean())*100:+.1f})")
    print("gemini-3.5-flash, plain vs accurate overlay (4 images):")
    s(d, "ALL"); s(d[d.teeth], "TEETH"); s(d[d.isloc], "LOCALIZE"); s(d[~d.teeth], "non-teeth")
    print("\nper image (all Qs):")
    for im, gg in d.groupby("img"):
        print(f"  {im}: {gg.plain.mean()*100:.0f}% -> {gg.ov.mean()*100:.0f}%  (n={len(gg)})")
    print(f"\nwrote {OUT}  (nondeterministic; exact %s vary run to run)")


if __name__ == "__main__":
    main()
