"""
Build the question-quality survey: every question we do NOT recommend dropping, put in
front of a dentist for one three-way judgement.

WHAT IS ASKED. For each question the dentist sees the X-ray, the question, the options
(multiple-choice) or the reference answer (free-text), and the keyed answer, then picks:

    Correct  /  Incorrect  /  Not enough information to be sure

The third button is the question-quality measure. If a practising dentist looking at the
actual radiograph cannot determine the answer, the question does not have a determinable
answer, and no model should be scored right or wrong on it (PAPER_DRAFT §7.4).

SCOPE. Closed KEEP+FLAG and open KEEP+FLAG+REPAIR, minus anything the dentist already
answered in round 1. Dropped, caption-track, malformed and location questions are NOT
asked: §7.2 settles those without an expert, and spending a clinician's time re-deciding
them is exactly what we said we would not do.

BATCHING. One image is one worksheet: the radiograph is read once and its ~9 questions
follow, which is where nearly all the saving comes from. Images are grouped into surveys
of --images-per-survey (default 25) and ordered so the highest-risk images come first, so
that a partly finished survey still yields the most informative items.

BLIND. The dentist never sees which bucket an item is in, what any model answered, or
whether we flagged it. They see the X-ray, the question and the recorded answer.

Run:    python -m dataio.make_quality_survey
Reads:  data/closed_ended.parquet, data/open_ended.parquet,
        results/corrected_benchmark/manifest_{closed,open}.csv,
        results/corrected_benchmark/repaired_references.csv,
        results/dentist_audit/{survey_manifest.csv,submission_sandeep.json}  (to skip)
        results/closed_ended/position_bias/*__shuffled__n491.csv             (risk order)
Writes: results/dentist_audit/quality_manifest.csv
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "results/dentist_audit/quality_manifest.csv"


def p(*r):
    return os.path.join(REPO, *r)


def already_answered():
    """(closed indices, open indices) the dentist answered in round 1. Never re-ask."""
    sub, man = p("results/dentist_audit/submission_sandeep.json"), p("results/dentist_audit/survey_manifest.csv")
    if not (os.path.exists(sub) and os.path.exists(man)):
        return set(), set()
    a = pd.DataFrame(json.load(open(sub, encoding="utf-8"))["answers"])[["item_id"]]
    m = pd.read_csv(man).merge(a, on="item_id")
    return (set(m[m.task_type == "closed"]["index"].astype(int)),
            set(m[m.task_type == "open"]["index"].astype(int)))


def risk_by_image(cl, keep_closed):
    """Images where the key is most likely to be wrong go first.

    Two signals, both already committed: how many of the image's questions we flagged
    (§7.2.6), and on how many our models disagree with the key. Model disagreement decides
    ONLY the order of review, never what stays in the benchmark (§7.4.2)."""
    risk = {}
    for _, r in cl[cl["index"].isin(keep_closed)].iterrows():
        img = str(r["file_name"]).split(".")[0]
        risk[img] = risk.get(img, 0)
    preds = []
    for f in glob.glob(p("results/closed_ended/position_bias/*__shuffled__n491.csv")):
        d = pd.read_csv(f)
        if {"index", "correct"} <= set(d.columns):
            preds.append(d.set_index("index")["correct"].astype(int))
    if preds:
        wrong = (1 - pd.concat(preds, axis=1).mean(axis=1))     # fraction of runs that missed
        for _, r in cl.iterrows():
            img = str(r["file_name"]).split(".")[0]
            if img in risk and r["index"] in wrong.index:
                risk[img] += float(wrong.loc[r["index"]])
    return risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-per-survey", type=int, default=25)
    args = ap.parse_args()

    cl = pd.read_parquet(p("data/closed_ended.parquet"))
    op = pd.read_parquet(p("data/open_ended.parquet"))
    mc = pd.read_csv(p("results/corrected_benchmark/manifest_closed.csv")).set_index("index")["disposition"]
    mo = pd.read_csv(p("results/corrected_benchmark/manifest_open.csv")).set_index("index")["disposition"]
    rep = pd.read_csv(p("results/corrected_benchmark/repaired_references.csv")).set_index("index")["repaired_reference"]
    sh = pd.read_parquet(p("data/closed_ended_shuffled.parquet")).set_index("index")["answer"]

    done_c, done_o = already_answered()
    keep_c = set(mc[mc.isin(["KEEP", "FLAG"])].index) - done_c
    keep_o = set(mo[mo.isin(["KEEP", "FLAG", "REPAIR"])].index) - done_o

    risk = risk_by_image(cl, keep_c)
    order = sorted(risk, key=lambda k: (-risk[k], k))
    rank = {img: i for i, img in enumerate(order)}

    rows = []
    for _, r in cl[cl["index"].isin(keep_c)].iterrows():
        img = str(r["file_name"]).split(".")[0]
        # the balanced key is what a corrected benchmark ships, so it is what we verify
        letter = sh.get(r["index"], r["answer"])
        opts = {L: str(r[f"option{i}"]) for i, L in zip(range(1, 5), "ABCD")}
        rows.append({"image": img, "index": int(r["index"]), "task_type": "closed",
                     "question": str(r["question"]), "A": opts["A"], "B": opts["B"],
                     "C": opts["C"], "D": opts["D"], "keyed_answer": letter,
                     "keyed_text": opts.get(letter, ""), "reference": "",
                     "risk_rank": rank.get(img, 999)})
    for _, r in op[op["index"].isin(keep_o)].iterrows():
        img = str(r["image_name"]).split(".")[0]
        ref = rep.get(r["index"])
        rows.append({"image": img, "index": int(r["index"]), "task_type": "open",
                     "question": str(r["question"]), "A": "", "B": "", "C": "", "D": "",
                     "keyed_answer": "", "keyed_text": "",
                     "reference": str(ref) if isinstance(ref, str) and ref.strip() else str(r["answer"]),
                     "risk_rank": rank.get(img, 999)})

    df = pd.DataFrame(rows).sort_values(["risk_rank", "image", "task_type", "index"])
    imgs = list(dict.fromkeys(df.image))
    batch = {im: i // args.images_per_survey + 1 for i, im in enumerate(imgs)}
    df["survey"] = df.image.map(batch)
    df["item_id"] = ["q%04d" % i for i in range(1, len(df) + 1)]
    os.makedirs(p("results/dentist_audit"), exist_ok=True)
    df.to_csv(p(OUT), index=False)

    print(f"skipped as already answered in round 1: {len(done_c)} closed, {len(done_o)} open")
    print(f"to review: {len(df)} questions over {len(imgs)} images "
          f"({(df.task_type=='closed').sum()} multiple-choice, {(df.task_type=='open').sum()} free-text)")
    print(f"\n{args.images_per_survey} images per survey -> {df.survey.max()} surveys")
    for s, g in df.groupby("survey"):
        print(f"  survey {s}: {g.image.nunique():3d} images, {len(g):3d} questions "
              f"(~{len(g)*12/60 + g.image.nunique()*1.25:.0f} min)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
