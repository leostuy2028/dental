"""
Analyze a dentist-audit submission (from the Web3Forms viewer) against the survey
manifest. No API. See dental_research/DENTIST_AUDIT.md for the design.

Reports, per model-agreement bucket (T1/T2/T3/C):
  - agree-with-key / disagree / cannot-determine / none-of-these counts
  - certified key-error rate (second-pass verdict == ERROR) with a Wilson 95% CI
Then the headline learning checks:
  - does model-disagreement predict key errors?  (T1+T2 error rate vs the C control)
  - a re-weighted whole-key error estimate (per-bucket rate x full-491 bucket size)
  - open-ended reference endorsement (Agree/Partial/Disagree/Cannot)

Usage: python paper_analysis/dentist_audit.py <submission.json> [manifest.csv]
"""
import os
import sys
import json
import pandas as pd
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from dataio.make_dentist_survey import buckets, SHUF, ORIG, CLOSED, load  # reuse bucket logic

MANIFEST = "results/dentist_audit/survey_manifest.csv"
LETTERS = set("ABCD")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (100 * (c - h) / d, 100 * (c + h) / d)


def full_bucket_sizes(repo):
    """Full-491 bucket sizes (order-invariant T1), for the re-weighted estimate."""
    dsh = {k: load(v, repo) for k, v in SHUF.items()}
    dor = {k: load(v, repo) for k, v in ORIG.items()}
    idx = dsh["gpt4o"].index
    closed = pd.read_parquet(os.path.join(repo, CLOSED)).set_index("index")
    none_opt = set(closed.index[closed[["option1", "option2", "option3", "option4"]].eq("None").any(axis=1)])
    tsh, tor = buckets(dsh, idx), buckets(dor, idx)
    valid = set(idx) - none_opt
    t1 = (set(tsh[tsh == "T1"].index) & set(tor[tor == "T1"].index)) & valid
    sizes = {"T1": len(t1)}
    for b in ("T2", "T3", "C"):
        sizes[b] = len((set(tsh[tsh == b].index) & valid) - t1)
    return sizes


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python paper_analysis/dentist_audit.py <submission.json> [manifest.csv]")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub = json.load(open(sys.argv[1], encoding="utf-8"))
    man = pd.read_csv(os.path.join(repo, sys.argv[2] if len(sys.argv) > 2 else MANIFEST),
                      keep_default_na=False, na_values=[]).set_index("item_id")
    verdict = {a["item_id"]: a["verdict"] for a in sub.get("adjudications", [])}

    is_test = "test" in str(sub.get("token", "")).lower()
    print("=" * 66)
    print(f"Dentist audit — token '{sub.get('token')}'"
          + ("   *** TEST DATA — numbers meaningless ***" if is_test else ""))
    print("=" * 66)

    # ---- closed, per bucket ----
    order = ["T1", "T2", "T3", "C"]
    agg = {b: Counter() for b in order}
    xcheck_ok = True
    for a in sub["answers"]:
        if a["task_type"] != "closed":
            continue
        iid = a["item_id"]
        if iid not in man.index:
            continue
        b = man.loc[iid, "bucket"]
        key = man.loc[iid, "answer_key"]
        ch = a["choice"]
        agg[b]["n"] += 1
        if ch in LETTERS and ch != key:
            agg[b]["disagree"] += 1
            v = verdict.get(iid)
            if v is None:
                xcheck_ok = False  # a disagreement with no adjudication = viewer bug
            if v == "ERROR":
                agg[b]["error"] += 1
            elif v == "DEFENSIBLE":
                agg[b]["defensible"] += 1
            elif v == "UNSURE":
                agg[b]["unsure"] += 1
        elif ch in LETTERS:
            agg[b]["agree"] += 1
        elif ch == "CANT":
            agg[b]["cant"] += 1
        elif ch == "NONE":
            agg[b]["none"] += 1

    # cross-check: adjudications must be exactly the definite disagreements
    disagree_ids = {a["item_id"] for a in sub["answers"] if a["task_type"] == "closed"
                    and a["choice"] in LETTERS and a["choice"] != man.loc[a["item_id"], "answer_key"]}
    match = disagree_ids == set(verdict) and xcheck_ok
    print(f"\n[pipeline check] {len(disagree_ids)} definite disagreements, "
          f"{len(verdict)} adjudications, exact match: {match}")

    print("\nPer bucket (certified error = dentist ruled the key WRONG in the 2nd pass):")
    print(f"  {'bucket':7}{'n':>4}{'agree':>7}{'disagree':>10}{'ERROR':>7}{'cant':>6}{'none':>6}"
          f"{'error rate [95% CI]':>26}")
    for b in order:
        c = agg[b]
        n, err = c["n"], c["error"]
        lo, hi = wilson(err, n)
        rate = f"{100*err/n:.0f}% [{lo:.0f}-{hi:.0f}]" if n else "-"
        print(f"  {b:7}{n:>4}{c['agree']:>7}{c['disagree']:>10}{err:>7}{c['cant']:>6}{c['none']:>6}{rate:>26}")

    # ---- headline: does model-disagreement predict key errors? ----
    susp_n = agg["T1"]["n"] + agg["T2"]["n"]
    susp_e = agg["T1"]["error"] + agg["T2"]["error"]
    c_n, c_e = agg["C"]["n"], agg["C"]["error"]
    print(f"\n[predicts errors?] suspect T1+T2: {susp_e}/{susp_n} "
          f"({100*susp_e/susp_n:.0f}%) vs control C: {c_e}/{c_n} ({100*c_e/c_n if c_n else 0:.0f}%)")

    # ---- re-weighted whole-key estimate ----
    sizes = full_bucket_sizes(repo)
    est = sum(sizes[b] * (agg[b]["error"] / agg[b]["n"] if agg[b]["n"] else 0) for b in order)
    tot = sum(sizes.values())
    print(f"[re-weighted] full-491 bucket sizes {sizes}; estimated key errors in those "
          f"{tot} items ≈ {est:.0f} ({100*est/tot:.0f}%)")

    # ---- open endorsement ----
    op = Counter(a["choice"] for a in sub["answers"] if a["task_type"] == "open")
    print(f"\nOpen-ended reference endorsement (n={sum(op.values())}): {dict(op)}")
    if is_test:
        print("\n*** Reminder: TEST DATA — do not report any of the above. ***")


if __name__ == "__main__":
    main()
