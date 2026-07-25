"""Measure the chart's effect on the 139 'other-concrete' open-ended questions (the ones
the router does NOT touch). Split by tooth-mentioning vs non-tooth. Writes each row to CSV
immediately (flushed) so progress is watchable; catches a 429 so partial data survives.

  python -u -m eval_open.run_inject_other
"""
import csv
import json
import os
import re
import sys
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, "paper_analysis")
from question_concreteness import classify   # noqa: E402
import eval_open.run_batched as rb
import eval_open.test_detector_inject as T
from eval_open.rubrics import build_grading_prompt
from eval_open.judges import grade

load_dotenv(".env")
rb.THINKING_BUDGET = 8192
OUT = "results/open/detector_inject_other139.csv"
TOOTH = r"teeth|tooth|dentition|wisdom|molar|premolar|incisor|canine|restorat|implant|crown|filling|prosthes"


def main():
    op = pd.read_parquet("data/open_ended.parquet")
    sc = pd.read_csv("results/open/batched_gemini35_plain578_scores.csv")
    tmap = json.load(open("reference/mmoral_map.json"))
    base = dict(zip(sc["index"], sc.score))

    q = op[["index", "image_name", "question"]].copy()
    q["bucket"] = q.question.map(classify)
    ql = q.question.str.lower()
    q["dtype"] = "other"
    q.loc[ql.str.contains("how many"), "dtype"] = "count"
    q.loc[ql.str.contains("missing|absent"), "dtype"] = "missing"
    q.loc[ql.str.contains(r"which (?:tooth|teeth)") & ql.str.contains(T.FINDING), "dtype"] = "whichtooth"
    q.loc[ql.str.contains(r"#?\b[1-4][1-8]\b") & ql.str.contains(T.FINDING), "dtype"] = "cond#N"
    other = q[(q.bucket == "Concrete") & (q.dtype == "other")].copy()
    other_idx = set(other["index"])
    grp = {i: ("tooth" if re.search(TOOTH, str(qq), re.I) else "nontooth")
           for i, qq in zip(other["index"], other.question)}
    imgs = [im for im in dict.fromkeys(other.image_name) if im in tmap]
    print(f"{len(other_idx)} other-concrete questions across {len(imgs)} images "
          f"(tooth {sum(v=='tooth' for v in grp.values())}, non-tooth {sum(v=='nontooth' for v in grp.values())})", flush=True)

    f = open(OUT, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["image", "index", "group", "base", "B"])
    w.writeheader(); f.flush()
    try:
        for im in imgs:
            sub = op[op.image_name == im].sort_values("index")
            qs = sub.question.tolist(); gts = sub.answer.tolist(); idxs = sub["index"].tolist()
            b64 = T.strip_b64(sub.image.iloc[0])
            ansB = T.run_image("gemini", "gemini-3.5-flash", b64, qs, T.build_chart(tmap[im]))
            for i, (idx, qq, gt) in enumerate(zip(idxs, qs, gts)):
                if idx not in other_idx:
                    continue
                sB = grade(build_grading_prompt(qq, gt, ansB[i], "original"))[0]
                w.writerow({"image": im, "index": idx, "group": grp[idx],
                            "base": base.get(idx, ""), "B": sB}); f.flush()
                print(f"  {im} [{idx}] {grp[idx]:8s} base {base.get(idx):.1f}  B {sB:.1f}  | {qq[:38]}", flush=True)
    except Exception as e:
        print(f"\nSTOPPED (Gemini cap?): {str(e)[:150]}", flush=True)
    f.close()

    d = pd.read_csv(OUT)
    print(f"\n=== {len(d)} of {len(other_idx)} other-concrete graded ===", flush=True)
    print(f"baseline {d.base.mean()*100:.1f}%  ->  B (+chart) {d.B.mean()*100:.1f}%   NET {(d.B.mean()-d.base.mean())*100:+.1f} pts", flush=True)
    for g, gg in d.groupby("group"):
        print(f"  {g:8s} n={len(gg):3d}  {gg.base.mean()*100:.1f}% -> {gg.B.mean()*100:.1f}%  ({(gg.B.mean()-gg.base.mean())*100:+.1f})", flush=True)


if __name__ == "__main__":
    main()
