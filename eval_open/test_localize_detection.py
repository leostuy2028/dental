"""
Focused test: does a tooth-detection map fix the LOCALIZE ("which tooth has X")
failures? CONTROL is reused from the baseline batched gpt-5-mini run (no detection);
the +DETECTION arm re-answers ONLY the 39 LOCALIZE images, batched the SAME way (all
of an image's questions together) but with the tooth chart injected — so the control
and treatment share batching and differ only in the map. gpt-5-mini, minimal reasoning.
Grades the 49 LOCALIZE items under the benchmark's unchanged rubric; reports the paired
delta and the wrong-tooth/mirror corrections.

  python -m eval_open.test_localize_detection --workers 8
"""
import argparse, json, os, re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import eval_open.run_batched as rb
from eval_open.prompts_open import COAX_SYSTEM
from eval_open.judges import grade
from eval_open.rubrics import build_grading_prompt

DATA = "data/open_ended.parquet"
PRIMER = "reference/opg_primer.txt"
DET = "reference/teeth_detections.json"
BASELINE = "results/open/batched_gpt5mini_scores.csv"   # reused no-detection control


def localize_idx(df):
    loc = df[(df.category.astype(str).str.contains("Teeth")) &
             (df.question.str.contains(r"which (tooth|teeth)", case=False, regex=True))]
    return set(loc["index"])


def teeth_set(t):
    return set(re.findall(r"#?\b([1-4][1-8])\b", str(t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--provider", default="openai", choices=["openai", "gemini", "anthropic"])
    ap.add_argument("--effort", default="minimal")   # openai reasoning models only
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/open/localize_detection.csv")
    args = ap.parse_args()
    rb.REASONING_EFFORT = args.effort
    primer = open(PRIMER, encoding="utf-8").read()
    det_map = json.load(open(DET, encoding="utf-8"))
    df = pd.read_parquet(DATA)
    loc_idx = localize_idx(df)
    loc_imgs = sorted(df[df["index"].isin(loc_idx)].image_name.unique())
    miss = [im for im in loc_imgs if im not in det_map]
    if miss:
        print(f"ABORT: {len(miss)}/{len(loc_imgs)} LOCALIZE images still lack a detection map.")
        return
    print(f"{args.model}: {len(loc_idx)} LOCALIZE items across {len(loc_imgs)} images; BOTH arms (batched)...")

    # both arms fresh for this model: control (no map) and +detection map, same batching
    def work(im):
        g = df[df.image_name == im].sort_values("index")
        b64, qs = g.iloc[0]["image"], g["question"].tolist()
        a_ctrl, _ = rb.answer_image(b64, qs, primer, COAX_SYSTEM, args.provider, args.model, detection_text=None)
        a_det, _ = rb.answer_image(b64, qs, primer, COAX_SYSTEM, args.provider, args.model,
                                   detection_text=det_map.get(im))
        return {int(idx): (c, d) for idx, c, d in zip(g["index"], a_ctrl, a_det)}

    both = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(work, im) for im in loc_imgs]):
            both.update(fut.result())
            print(f"  answered ~{len(both)//6}/{len(loc_imgs)} images")

    rec = []
    for idx in sorted(loc_idx):
        row = df[df["index"] == idx].iloc[0]
        a_ctrl, a_det = both.get(idx, ("", ""))
        s_ctrl, _ = grade(build_grading_prompt(row.question, row["answer"], a_ctrl, "original"), judge="gpt-4o")
        s_det, _ = grade(build_grading_prompt(row.question, row["answer"], a_det, "original"), judge="gpt-4o")
        rec.append({"index": idx, "question": row.question, "gt": str(row["answer"]),
                    "ans_ctrl": a_ctrl, "ans_det": a_det, "score_ctrl": s_ctrl, "score_det": s_det})
    out = pd.DataFrame(rec)
    os.makedirs("results/open", exist_ok=True)
    out.to_csv(args.out, index=False)

    from scipy.stats import wilcoxon
    d = out.score_det - out.score_ctrl
    p = wilcoxon(d, zero_method="zsplit").pvalue if d.abs().sum() else 1.0
    print("\n" + "=" * 60)
    print(f"LOCALIZE: baseline (no det, batched) vs +detection map, gpt-5-mini minimal, n={len(out)}")
    print(f"  control (baseline)  : {out.score_ctrl.mean()*100:.1f}%")
    print(f"  + detection map     : {out.score_det.mean()*100:.1f}%")
    print(f"  paired delta        : {d.mean()*100:+.1f} pts  (Wilcoxon p={p:.3g}; "
          f"up {int((d>0).sum())}, down {int((d<0).sum())}, same {int((d==0).sum())})")
    print("=" * 60)


if __name__ == "__main__":
    main()
