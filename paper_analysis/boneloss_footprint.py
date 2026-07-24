"""
GENERATOR for: PAPER_DRAFT.md §5.8 (dentist audit, round 2) — how many benchmark
questions the bone-loss key weakness can touch. Dataset-derived (counts over the
released files), not model outputs.

The round-2 dentist survey confirmed the key UNDER-reports bone loss. The at-risk
items are those the key answers "no (apparent) bone loss": in the closed half, the
yes/no bone-loss MCQs keyed to "No"; in the open half, references asserting "no
apparent bone loss". (Footprint = upper bound; the demonstrated errors are the
smaller model-flagged slice inside it.)

Run:   python paper_analysis/boneloss_footprint.py
Reads: data/open_ended.parquet, data/closed_ended.parquet
Writes: paper_analysis/_generated/boneloss_footprint.values.json
"""
import os
import re
import json
import datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "_generated")
BONE = re.compile(r"bone loss|bone architecture|alveolar bone|resorption|periodont|\bbone\b", re.I)
NOLOSS = re.compile(r"no (?:apparent )?(?:bone loss|resorption|significant bone)", re.I)


def main():
    op = pd.read_parquet(os.path.join(REPO, "data/open_ended.parquet"))
    open_bone = int((op.question.str.contains(BONE) | op.answer.astype(str).str.contains(BONE)).sum())
    open_noloss = int(op.answer.astype(str).str.contains(NOLOSS).sum())

    cl = pd.read_parquet(os.path.join(REPO, "data/closed_ended.parquet"))
    optcols = ["option1", "option2", "option3", "option4"]
    letter2col = {"A": "option1", "B": "option2", "C": "option3", "D": "option4"}
    cl = cl.copy()
    cl["bone"] = cl.apply(lambda r: bool(BONE.search(" ".join(str(r[c]) for c in ["question"] + optcols))), axis=1)
    cl["correct"] = cl.apply(lambda r: str(r[letter2col[r["answer"]]]), axis=1)
    closed_bone = int(cl["bone"].sum())
    closed_noloss = int(cl["bone"].sum() and cl[cl["bone"]]["correct"].str.contains(NOLOSS).sum())

    vals = {
        "open": {"n": len(op), "bone_mentioned": open_bone,
                 "keyed_no_bone_loss": open_noloss,
                 "keyed_no_bone_loss_pct": round(100 * open_noloss / len(op), 1)},
        "closed": {"n": len(cl), "bone_mentioned": closed_bone,
                   "keyed_no_bone_loss": closed_noloss,
                   "keyed_no_bone_loss_pct": round(100 * closed_noloss / len(cl), 1)},
        "_generator": "paper_analysis/boneloss_footprint.py",
        "_note": "dataset-derived footprint (at-risk items), not confirmed errors",
        "_generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "boneloss_footprint.values.json"), "w", encoding="utf-8") as f:
        json.dump(vals, f, indent=2)
    print(f"OPEN  (n={len(op)}): bone-mentioned {open_bone}, keyed 'no bone loss' {open_noloss} "
          f"({vals['open']['keyed_no_bone_loss_pct']}%)")
    print(f"CLOSED(n={len(cl)}): bone-mentioned {closed_bone}, keyed 'no bone loss' {closed_noloss} "
          f"({vals['closed']['keyed_no_bone_loss_pct']}%)")


if __name__ == "__main__":
    main()
