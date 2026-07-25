"""Finish the 4 images the capped run didn't reach, merge with the 142 already done,
recompute the net. Reuses test_detector_inject; writes/prints incrementally; catches a
429 so a still-capped Gemini fails fast without losing anything.

  python -m eval_open.finish_inject
"""
import json
import pandas as pd
from dotenv import load_dotenv

import eval_open.run_batched as rb
import eval_open.test_detector_inject as T
from eval_open.rubrics import build_grading_prompt
from eval_open.judges import grade

load_dotenv(".env")
rb.THINKING_BUDGET = 8192
CSV = "results/open/detector_inject_gemini35flash_all.csv"
REMAIN = ["018477.jpg", "018494.jpg", "018539.jpg", "019277.jpg"]


def main():
    op = pd.read_parquet("data/open_ended.parquet")
    tmap = json.load(open("reference/mmoral_map.json"))
    tgt = T.select_targets(op, wrong_only=False)
    target_idx = set(tgt["index"])
    dtype_of = dict(zip(tgt["index"], tgt.dtype))
    base = dict(zip(tgt["index"], tgt.score))

    new = []
    try:
        for im in REMAIN:
            sub = op[op.image_name == im].sort_values("index")
            qs = sub.question.tolist(); gts = sub.answer.tolist(); idxs = sub["index"].tolist()
            b64 = T.strip_b64(sub.image.iloc[0])
            ansB = T.run_image("gemini", "gemini-3.5-flash", b64, qs, T.build_chart(tmap[im]))
            for i, (idx, q, gt) in enumerate(zip(idxs, qs, gts)):
                if idx not in target_idx:
                    continue
                sB = grade(build_grading_prompt(q, gt, ansB[i], "original"))[0]
                new.append({"image": im, "idx": idx, "dtype": dtype_of[idx], "base": base[idx], "B": sB})
                print(f"  {im} [{idx}] {dtype_of[idx]:10s} base {base[idx]:.1f}  B {sB:.1f}  | {q[:40]}", flush=True)
    except Exception as e:
        print(f"\nSTOPPED (Gemini still unavailable?): {str(e)[:180]}", flush=True)

    old = pd.read_csv(CSV)[["image", "idx", "dtype", "base", "B"]]
    full = pd.concat([old, pd.DataFrame(new, columns=old.columns)], ignore_index=True) if new else old
    full.to_csv(CSV, index=False)

    print(f"\n=== {len(full)} of 148 questions ({len(new)} new this run) ===")
    print(f"baseline {full.base.mean()*100:.1f}%  ->  B (+chart) {full.B.mean()*100:.1f}%   NET {(full.B.mean()-full.base.mean())*100:+.1f} pts")
    resc, reg = full[full.base <= 0.4], full[full.base > 0.4]
    print(f"  rescue        (n={len(resc)}): {resc.base.mean()*100:.0f}% -> {resc.B.mean()*100:.0f}%   ({(resc.B.mean()-resc.base.mean())*100:+.0f})")
    print(f"  already-right (n={len(reg)}): {reg.base.mean()*100:.0f}% -> {reg.B.mean()*100:.0f}%   ({(reg.B.mean()-reg.base.mean())*100:+.0f})")
    for t, g in full.groupby("dtype"):
        print(f"  {t:11s} n={len(g):2d}  {g.base.mean()*100:5.1f}% -> {g.B.mean()*100:5.1f}%  ({(g.B.mean()-g.base.mean())*100:+.1f})")


if __name__ == "__main__":
    main()
