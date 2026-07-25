"""
Targeted test: does injecting the TRAINED detector's FDI map into the prompt help the
VLM on the concrete questions it currently gets WRONG?

Prior injection failed (§9.1) but used an INACCURATE, VLM-generated tooth map. This uses
the trained YOLO map (reference/mmoral_map.json, from detector/infer_mmoral.py --map-out)
PLUS a thinking budget, and restricts to gemini-wrong (score<=0.4), detector-relevant
concrete questions (count / missing / which-tooth-finding / condition-of-#N) — the cases
where knowing which tooth is which could conceivably fix the answer.

Two arms, same session, same thinking budget, so the only difference is the chart:
  A = no chart          (isolates the thinking budget)
  B = + detector TOOTH CHART   (the injection under test)
The committed plain578 score (thinking off, no chart) is shown as the outside baseline.

Questions are batched per image (matching the plain578 protocol); only the target
questions are scored. Grading = GPT-4o judge, rubric='original' (unchanged).

Run:
  python -m eval_open.test_detector_inject --provider gemini --model gemini-3.5-flash --think 8192
  python -m eval_open.test_detector_inject --provider openai --model gpt-5 --effort high
"""
import argparse
import json
import re
import sys
import pandas as pd
from dotenv import load_dotenv

import eval_open.run_batched as rb
from eval_open.run_batched import build_user, parse_numbered, PROVIDERS
from eval_open.prompts_open import COAX_SYSTEM
from eval_open.rubrics import build_grading_prompt
from eval_open.judges import grade

sys.path.insert(0, "paper_analysis")
from question_concreteness import classify   # noqa: E402

load_dotenv(".env")
DATA = "data/open_ended.parquet"
MAP = "reference/mmoral_map.json"
PRIMER = open("reference/opg_primer.txt", encoding="utf-8").read()
SCORES = "results/open/batched_gemini35_plain578_scores.csv"

QUAD = {"1": "upper-right", "2": "upper-left", "3": "lower-left", "4": "lower-right"}
TOOTH = {"1": "central incisor", "2": "lateral incisor", "3": "canine", "4": "first premolar",
         "5": "second premolar", "6": "first molar", "7": "second molar", "8": "third molar (wisdom)"}
FINDING = (r"caries|cariou|filling|restor|crown|implant|bridge|root canal|endodon|lesion|"
           r"impact|periapical|bone loss|patholog|decay|interven|condition of")


def anat(f):
    return f"{QUAD[f[0]]} {TOOTH[f[1]]}"


def build_chart(entry):
    """entry = {'count': int, 'teeth': [{fdi,box,conf}]} -> a plain-text tooth chart.
    Honest about the detector's partial numbering: state the reliable total count and only
    the confidently-numbered teeth; do NOT infer 'missing' (numbering is noisy on MMOral)."""
    teeth = sorted(entry["teeth"], key=lambda t: t["fdi"])
    lines = [f"  #{t['fdi']} = {anat(t['fdi'])}" for t in teeth]
    s = (f"Total teeth detected on this X-ray: {entry['count']}.\n"
         f"Of these, {len(teeth)} were confidently numbered (FDI code = position):\n"
         + "\n".join(lines)
         + "\n(The other detected teeth were located but not confidently numbered.)")
    return s


def strip_b64(s):
    return re.sub(r"^data:image/\w+;base64,", "", str(s))


def select_targets(op, wrong_only=True):
    sc = pd.read_csv(SCORES)
    m = op.merge(sc, on="index")
    m["bucket"] = m.question.map(classify)
    ql = m.question.str.lower()
    m["dtype"] = "other"
    m.loc[ql.str.contains("how many"), "dtype"] = "count"
    m.loc[ql.str.contains("missing|absent"), "dtype"] = "missing"
    m.loc[ql.str.contains(r"which (?:tooth|teeth)") & ql.str.contains(FINDING), "dtype"] = "whichtooth"
    m.loc[ql.str.contains(r"#?\b[1-4][1-8]\b") & ql.str.contains(FINDING), "dtype"] = "cond#N"
    rel = m.bucket.eq("Concrete") & m.dtype.isin(["count", "missing", "whichtooth", "cond#N"])
    sel = m[rel]
    if wrong_only:
        sel = sel[sel.score <= 0.4]
    return sel.copy()


def run_image(provider, model, b64, questions, chart):
    user = build_user(PRIMER, questions, detection_text=chart)   # chart=None -> no injection
    raw = PROVIDERS[provider](b64, COAX_SYSTEM, user, model)
    return parse_numbered(raw, len(questions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--think", type=int, default=8192, help="Gemini thinking budget")
    ap.add_argument("--effort", default="high", help="OpenAI reasoning effort")
    ap.add_argument("--n-images", type=int, default=10, help="max target images (ignored with --all)")
    ap.add_argument("--all", action="store_true",
                    help="all 148 detector-relevant concrete Qs (net effect), not just gemini-wrong")
    ap.add_argument("--chart-only", action="store_true",
                    help="skip the no-chart arm; compare chart directly to the committed baseline")
    args = ap.parse_args()

    rb.THINKING_BUDGET = args.think
    rb.REASONING_EFFORT = args.effort

    op = pd.read_parquet(DATA)
    tmap = json.load(open(MAP))
    tgt = select_targets(op, wrong_only=not args.all)
    target_idx = set(tgt["index"])
    dtype_of = dict(zip(tgt["index"], tgt.dtype))
    base = dict(zip(tgt["index"], tgt.score))
    imgs = [im for im in dict.fromkeys(tgt.image_name) if im in tmap]
    target_images = imgs if args.all else imgs[: args.n_images]
    arms = "B=+chart only" if args.chart_only else "A=no chart, B=+chart"
    print(f"{len(target_idx)} target questions across {tgt.image_name.nunique()} images; "
          f"running {len(target_images)} images.")
    print(f"arms: {arms} | model={args.model} think={args.think} effort={args.effort}\n")

    rows = []
    for imname in target_images:
        sub = op[op.image_name == imname].sort_values("index")
        qs = sub.question.tolist(); gts = sub.answer.tolist(); idxs = sub["index"].tolist()
        b64 = strip_b64(sub.image.iloc[0])
        chart = build_chart(tmap[imname])
        ansB = run_image(args.provider, args.model, b64, qs, chart)
        ansA = None if args.chart_only else run_image(args.provider, args.model, b64, qs, None)
        for i, (idx, q, gt) in enumerate(zip(idxs, qs, gts)):
            if idx not in target_idx:
                continue
            sB = grade(build_grading_prompt(q, gt, ansB[i], "original"))[0]
            row = {"image": imname, "index": idx, "dtype": dtype_of[idx], "q": q[:52],
                   "base": base[idx], "B_chart": sB, "ansB": ansB[i][:60]}
            atag = ""
            if not args.chart_only:
                row["A_nochart"] = grade(build_grading_prompt(q, gt, ansA[i], "original"))[0]
                row["ansA"] = ansA[i][:60]
                atag = f"A {row['A_nochart']:.1f}  "
            rows.append(row)
            print(f"  {imname} [{idx}] {dtype_of[idx]:10s} base {base[idx]:.1f}  {atag}B {sB:.1f}  | {q[:40]}")

    df = pd.DataFrame(rows)
    tagall = "_all" if args.all else ""
    out = f"results/open/detector_inject_{args.model.replace('.', '').replace('-', '')}{tagall}.csv"
    df.to_csv(out, index=False)
    print(f"\n=== {len(df)} target questions ===")
    print(f"baseline (committed plain578):         {df.base.mean()*100:.1f}%")
    if not args.chart_only:
        print(f"A (no chart, think {args.think}):         {df.A_nochart.mean()*100:.1f}%")
    print(f"B (+ detector chart, think {args.think}):  {df.B_chart.mean()*100:.1f}%")
    print(f"NET (B - baseline): {(df.B_chart.mean()-df.base.mean())*100:+.1f} pts")
    resc, reg = df[df.base <= 0.4], df[df.base > 0.4]
    if len(resc):
        print(f"  rescue        (base<=0.4, n={len(resc):2d}): {resc.base.mean()*100:4.0f}% -> {resc.B_chart.mean()*100:4.0f}%")
    if len(reg):
        print(f"  already-right (base >0.4, n={len(reg):2d}): {reg.base.mean()*100:4.0f}% -> {reg.B_chart.mean()*100:4.0f}%   <- regression check")
    print("\nby type (n | baseline | B+chart | delta):")
    for t, g in df.groupby("dtype"):
        print(f"  {t:10s} n={len(g):2d}  {g.base.mean()*100:5.1f}%  {g.B_chart.mean()*100:5.1f}%  "
              f"{(g.B_chart.mean()-g.base.mean())*100:+.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
