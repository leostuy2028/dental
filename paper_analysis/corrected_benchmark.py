"""
GENERATOR for: PAPER_DRAFT.md §7 "Toward a more trustworthy benchmark" (Tables 7.2, 7.3)
and for the corrected-benchmark release itself (pillar (d)).

Answers one question exactly: **item by item, what does a corrected MMOral-Bench keep,
what does it repair, what does it move to a separate track, and what does it drop?**

The §7 concreteness analysis alone suggests the free-text half has a trustworthy core of
287 "concrete" questions. That is too generous. 97 of those 287 have a reference answer
that is not a clinical statement at all but a raw list of pixel boxes, so a model cannot
be scored on reading them and a grader that only reads text cannot check them (§6.1).
Netting those out, the genuinely checkable core is smaller, and this script computes it.

DISPOSITIONS (disjoint; precedence in this order, so every item lands in exactly one)
  DROP        — unanswerable as posed. Closed: the question quotes pixel coordinates that
                are nowhere on the image (the dentist audit's "I don't see any coordinates",
                §5.8). Nothing to salvage.
  SEPARATE    — no determinate answer: whole-image caption/summary questions. Keep them,
                but score them as writing quality on their own track, never as reading
                accuracy, because §7 shows models score HIGHEST here precisely because
                they cannot be marked wrong.
  REPAIR      — the question is checkable but its reference answer is a raw coordinate
                box list. Replace the boxes with the plain finding they encode; drop only
                if unrepairable.
  FLAG        — keep and score, but mark for expert re-adjudication: references asserting
                "no apparent bone loss" (the one content error the dentist audit confirmed,
                §5.8) and items whose tooth codes are ambiguous between FDI and US Universal
                (§5.4).
  KEEP        — checkable, no known defect. This is the core to evaluate and improve on.

The closed half additionally gets its answer key REBALANCED (every kept item, §5.3), which
is a transformation rather than a disposition, so it is reported separately.

Run:    python -m paper_analysis.corrected_benchmark
Reads:  data/closed_ended.parquet, data/open_ended.parquet
Writes: paper_analysis/_generated/corrected_benchmark_{closed,open}_table.md
        paper_analysis/_generated/corrected_benchmark.values.json
        results/corrected_benchmark/manifest_{closed,open}.csv   <- the release artifact:
                                                                    every item + disposition
"""
import datetime
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "_generated")
REL_DIR = os.path.join(REPO, "results", "corrected_benchmark")
sys.path.insert(0, HERE)
from question_concreteness import classify   # noqa: E402  (the §7 bucket classifier)

COORD_IN_QUESTION = r"\[\s*\d+\s*,\s*\d+"        # "...within the bounding box [1377, 36, ...]"
COORD_REFERENCE = r"box_2d|point_2d"             # reference answer is a raw box list
BONE_LOSS_DEFAULT = r"no apparent bone loss"     # the confirmed key blind spot (§5.8)


def tooth_codes(*texts):
    out = []
    for t in texts:
        out += [int(c) for c in re.findall(r"#(\d{1,2})", str(t))]
    return out


def closed_dispositions(cl):
    """Disjoint disposition per closed item, in precedence order."""
    q = cl.question.astype(str)
    drop = q.str.contains(COORD_IN_QUESTION, regex=True)
    codes = cl.apply(lambda r: tooth_codes(r.question, r.option1, r.option2, r.option3, r.option4), axis=1)
    ambiguous = codes.map(lambda cs: len(cs) > 0 and all(11 <= c <= 32 for c in cs))
    boneloss = q.str.contains("bone loss", case=False)
    d = pd.Series("KEEP", index=cl.index)
    d[ambiguous | boneloss] = "FLAG"
    d[drop] = "DROP"                              # highest precedence
    return d, {"coord_in_question": int(drop.sum()), "ambiguous_codes": int(ambiguous.sum()),
               "bone_loss_items": int(boneloss.sum())}


def open_dispositions(op):
    """Disjoint disposition per free-text item, in precedence order."""
    bucket = op.question.map(classify)
    ref = op.answer.astype(str)
    coordref = ref.str.contains(COORD_REFERENCE, case=False, regex=True)
    boneloss = ref.str.contains(BONE_LOSS_DEFAULT, case=False)
    d = pd.Series("KEEP", index=op.index)
    d[boneloss] = "FLAG"                          # lowest precedence of the defects
    d[coordref] = "REPAIR"
    d[bucket == "Vague"] = "SEPARATE"             # highest: no determinate answer at all
    return d, bucket, {"coord_reference": int(coordref.sum()),
                       "bone_loss_reference": int(boneloss.sum()),
                       "vague": int((bucket == "Vague").sum())}


ORDER = ["KEEP", "FLAG", "REPAIR", "SEPARATE", "DROP"]
MEANING = {
    "KEEP": "checkable, no known defect; the core to evaluate and improve on",
    "FLAG": "kept and scored, but marked for expert re-adjudication",
    "REPAIR": "checkable question, but the reference answer is a raw coordinate box list",
    "SEPARATE": "no determinate answer; score as writing quality on its own track",
    "DROP": "unanswerable as posed; the question quotes coordinates not on the image",
}


def table(d, total, title):
    lines = [f"| {title} | items | share | disposition |", "|:--|--:|--:|:--|"]
    for k in ORDER:
        n = int((d == k).sum())
        if n:
            lines.append(f"| **{k.title()}** | {n} | {100*n/total:.1f}% | {MEANING[k]} |")
    lines.append(f"| *total* | {total} | 100% | |")
    return lines


def main():
    cl = pd.read_parquet(os.path.join(REPO, "data/closed_ended.parquet"))
    op = pd.read_parquet(os.path.join(REPO, "data/open_ended.parquet"))
    cd, cinfo = closed_dispositions(cl)
    od, bucket, oinfo = open_dispositions(op)

    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    prov = (f"<!-- GENERATED by paper_analysis/corrected_benchmark.py on {stamp}. "
            f"Do not hand-edit; run `python -m paper_analysis.corrected_benchmark`. -->")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REL_DIR, exist_ok=True)

    closed_md = "\n".join([prov] + table(cd, len(cl), "Multiple-choice half (491)")) + "\n"
    open_md = "\n".join([prov] + table(od, len(op), "Free-text half (578)")) + "\n"
    open(os.path.join(OUT_DIR, "corrected_benchmark_closed_table.md"), "w", encoding="utf-8").write(closed_md)
    open(os.path.join(OUT_DIR, "corrected_benchmark_open_table.md"), "w", encoding="utf-8").write(open_md)

    # the release artifact: every item with its disposition
    pd.DataFrame({"index": cl["index"], "disposition": cd.values,
                  "rebalanced_key": True}).to_csv(os.path.join(REL_DIR, "manifest_closed.csv"), index=False)
    pd.DataFrame({"index": op["index"], "concreteness": bucket.values,
                  "disposition": od.values}).to_csv(os.path.join(REL_DIR, "manifest_open.csv"), index=False)

    vals = {
        "closed": {"total": len(cl), **{k: int((cd == k).sum()) for k in ORDER}, "detail": cinfo,
                   "note": "every KEEP/FLAG item is additionally re-keyed onto the position-balanced key (§5.3)"},
        "open": {"total": len(op), **{k: int((od == k).sum()) for k in ORDER}, "detail": oinfo,
                 "concreteness": {k: int((bucket == k).sum()) for k in ["Concrete", "Broad", "Vague"]}},
        "_generator": "paper_analysis/corrected_benchmark.py", "_generated_utc": stamp,
        "_source": ["data/closed_ended.parquet", "data/open_ended.parquet"],
    }
    # the headline: how much of the free-text half is actually checkable and defect-free
    concrete_clean = int(((bucket == "Concrete") & (od == "KEEP")).sum())
    vals["open"]["concrete_and_clean"] = concrete_clean
    vals["open"]["concrete_but_coordinate_reference"] = int(((bucket == "Concrete") & (od == "REPAIR")).sum())

    json.dump(vals, open(os.path.join(OUT_DIR, "corrected_benchmark.values.json"), "w"), indent=2)
    print(closed_md); print(open_md)
    print(f"free-text: {vals['open']['concreteness']['Concrete']} questions are 'concrete', but "
          f"{vals['open']['concrete_but_coordinate_reference']} of those have a raw coordinate-box "
          f"reference, leaving {concrete_clean} concrete AND defect-free "
          f"({100*concrete_clean/len(op):.0f}% of the half).")
    print(f"\nwrote {OUT_DIR}/corrected_benchmark_*_table.md, .values.json")
    print(f"wrote {REL_DIR}/manifest_{{closed,open}}.csv  (the release artifact)")


if __name__ == "__main__":
    main()
