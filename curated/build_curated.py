"""
Build the curated MMOral-Bench release.

WHAT THIS SHIPS, AND WHY IT IS A PATCH RATHER THAN A FORK. The X-rays belong to
MMOral-Bench. Re-uploading them would duplicate a large dataset and muddy its provenance,
so this release carries no images and no copied questions. It carries the CHANGES: for
every one of the 1,069 released items, what we decided and why, plus the corrected answer
key and the repaired reference answers. A user downloads MMOral-Bench from its own source
and applies this patch, which keeps the original dataset canonical and makes every one of
our edits inspectable side by side with what it replaced.

WHAT CHANGES (paper §7):
  * the multiple-choice answer key is rebalanced, so always guessing one letter drops
    from 44.0% to 27.9%
  * 82 free-text reference answers that were lists of pixel boxes become sentences
  * 7 multiple-choice questions are removed as unanswerable
  * 107 free-text questions move to a separate writing-quality track
  * 6 free-text questions are marked as needing an overlap metric, not a text judge
  * 13 malformed references and 100 risky items are flagged, not changed

Run:    python curated/build_curated.py
Reads:  data/closed_ended.parquet, data/closed_ended_shuffled.parquet,
        data/open_ended.parquet,
        results/corrected_benchmark/manifest_{closed,open}.csv,
        results/corrected_benchmark/repaired_references.csv
Writes: curated/mmoral_curated_closed.csv
        curated/mmoral_curated_open.csv
        curated/VERSION.json
"""
import datetime
import json
import os
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VERSION = "0.1.0"

REASON = {
    "KEEP": "no defect found by inspection of the released files",
    "REPAIR": "reference answer was a coordinate list; rewritten as the sentence it encodes",
    "FLAG": "kept and scored, but sits on ground known to be unreliable (see flag_reason)",
    "SPATIAL": "question asks for a location; score by box overlap, not by a text judge",
    "MALFORMED": "reference answer is not valid JSON in the released file; left unchanged",
    "SEPARATE": "no determinate answer; score as writing quality, never as reading accuracy",
    "DROP": "unanswerable: asks about a box at coordinates that do not appear on the image",
}


def p(*rel):
    return os.path.join(REPO, *rel)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def main():
    cl = pd.read_parquet(p("data/closed_ended.parquet"))
    sh = pd.read_parquet(p("data/closed_ended_shuffled.parquet"))
    op = pd.read_parquet(p("data/open_ended.parquet"))
    mc = pd.read_csv(p("results/corrected_benchmark/manifest_closed.csv")).set_index("index")
    mo = pd.read_csv(p("results/corrected_benchmark/manifest_open.csv")).set_index("index")
    rep = pd.read_csv(p("results/corrected_benchmark/repaired_references.csv")).set_index("index")

    # ---- multiple choice: disposition + the rebalanced key -------------------------
    key_orig = cl.set_index("index")["answer"]
    key_bal = sh.set_index("index")["answer"]
    q = cl.set_index("index")["question"]
    closed = pd.DataFrame({
        "index": mc.index,
        "disposition": mc["disposition"].values,
        "reason": [REASON[d] for d in mc["disposition"]],
        "original_answer": key_orig.reindex(mc.index).values,
        "balanced_answer": key_bal.reindex(mc.index).values,
    })
    ql = q.reindex(mc.index).astype(str).str.lower()
    closed["flag_reason"] = [
        "bone loss: the one topic the dentist audit found the key repeatedly wrong on"
        if "bone loss" in t else
        ("tooth codes are valid in BOTH the FDI and US Universal systems and mean different "
         "teeth in each; the benchmark never states which it uses" if d == "FLAG" else "")
        for t, d in zip(ql, closed["disposition"])]
    closed["in_reading_score"] = closed["disposition"] != "DROP"
    closed.to_csv(os.path.join(HERE, "mmoral_curated_closed.csv"), index=False)

    # ---- free text: disposition + repaired references ------------------------------
    o = pd.DataFrame({
        "index": mo.index,
        "concreteness": mo["concreteness"].values,
        "disposition": mo["disposition"].values,
        "reason": [REASON[d] for d in mo["disposition"]],
    })
    o["repaired_reference"] = rep["repaired_reference"].reindex(mo.index).fillna("").values
    o["reference_changed"] = o["repaired_reference"] != ""
    # The reading score excludes three groups: the caption questions (no determinate answer),
    # the location questions (need an overlap metric), and the malformed ones. The malformed
    # matter here because their reference is BOTH a raw box list and unparseable, so the
    # §7.2.1 repair could not run on them; leaving them in would score plain-English answers
    # against coordinates, which is exactly the mismatch this release exists to remove.
    o["in_reading_score"] = ~o["disposition"].isin(["SEPARATE", "SPATIAL", "MALFORMED"])
    o["score_track"] = ["writing" if d == "SEPARATE" else
                        ("overlap" if d == "SPATIAL" else
                         ("excluded" if d == "MALFORMED" else "reading"))
                        for d in o["disposition"]]
    o.to_csv(os.path.join(HERE, "mmoral_curated_open.csv"), index=False)

    meta = {
        "name": "MMOral-Bench (curated)",
        "version": VERSION,
        "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": "OralGPT/MMOral-OPG-Bench (HuggingFace, MIT licence)",
        "source_commit_hash": "[verify: record the HF dataset commit at download time]",
        "built_from_code_commit": git_commit(),
        "form": "patch, not a fork: dispositions + corrected key + repaired references. No images.",
        "closed": {"total": int(len(closed)),
                   **{k: int((closed.disposition == k).sum()) for k in sorted(set(closed.disposition))},
                   "in_reading_score": int(closed.in_reading_score.sum())},
        "open": {"total": int(len(o)),
                 **{k: int((o.disposition == k).sum()) for k in sorted(set(o.disposition))},
                 "in_reading_score": int(o.in_reading_score.sum()),
                 "references_rewritten": int(o.reference_changed.sum())},
        "generators": ["paper_analysis/corrected_benchmark.py",
                       "dataio/repair_coordinate_refs.py",
                       "dataio/prepare_datasets.py",
                       "curated/build_curated.py"],
    }
    with open(os.path.join(HERE, "VERSION.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"MMOral-Bench (curated) v{VERSION}")
    print(f"  multiple choice: {meta['closed']['in_reading_score']} of {len(closed)} scored "
          f"({dict((k, v) for k, v in meta['closed'].items() if k.isupper())})")
    print(f"  free text      : {meta['open']['in_reading_score']} of {len(o)} in the reading score, "
          f"{meta['open']['references_rewritten']} references rewritten")
    print(f"  wrote curated/mmoral_curated_closed.csv, mmoral_curated_open.csv, VERSION.json")


if __name__ == "__main__":
    main()
