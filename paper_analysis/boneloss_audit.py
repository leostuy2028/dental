"""
Analyze the bone-loss key-bias survey (dentist audit, round 2). NO API.

Reads a returned submission JSON + results/dentist_audit/boneloss_manifest.csv and
adjudicates, per image, the three-way disagreement between the benchmark key, our two
models, and the dentist. Answers one question: does the Jaw key systematically
UNDER-report alveolar bone loss?

Rating scale (from the blind survey):
  1 = no pathologic bone loss   -> AGREES with a "no bone loss" key
  2/3/4 = mild / localized / moderate-severe bone loss -> DISAGREES (loss present)
  5 = cannot assess             -> excluded from the denominator

Buckets (hidden from the dentist; in the manifest):
  DISAGREE   key="none", both models="loss"  -> adjudication set (the test)
  AGREE_NORM key="none", both models="none"  -> negative control (dentist not over-calling)
  KEY_POS    key="loss"                       -> positive control (dentist detects real loss)

Decision (one-sided exact binomial on DISAGREE, k of n confirm loss, vs rater-noise p0):
  k>=5 of 12 (p0=.15) -> key systematically under-reports bone loss (dataset bias)
  k<=3               -> key fine; the models over-call (model bias)

Run:  python -m paper_analysis.boneloss_audit [results/dentist_audit/boneloss_submission_<token>.json]
"""
import sys
import glob
import json
from math import comb
import pandas as pd

MANIFEST = "results/dentist_audit/boneloss_manifest.csv"


def binom_tail(n, k, p):
    """P(X >= k) for X ~ Binom(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def loss_call(rating):
    """dentist rating -> True(loss) / False(no loss) / None(cannot assess)."""
    if rating in (None, "", "5"):
        return None
    return str(rating) in ("2", "3", "4")


def main():
    subs = sys.argv[1:] or sorted(glob.glob("results/dentist_audit/boneloss_submission_*.json"))
    if not subs:
        print("No submission JSON yet. Expected: results/dentist_audit/boneloss_submission_<token>.json")
        print("(The dentist's survey emails/downloads that file; drop it in results/dentist_audit/ then re-run.)")
        return
    sub = subs[0]
    man = pd.read_csv(MANIFEST).set_index("item_id")
    data = json.load(open(sub, encoding="utf-8"))
    resp = {r["item_id"]: r for r in data["responses"]}
    print(f"submission: {sub}  |  dentist: {data.get('dentist_name','?')}  "
          f"|  answered {data.get('n_answered','?')}/{len(man)}\n")

    rows = []
    for iid, r in man.iterrows():
        rr = resp.get(iid, {})
        rows.append({"item_id": iid, "bucket": r["bucket"], "image": r["image_name"],
                     "key": r["key_stance"], "rating": rr.get("rating"),
                     "dentist_loss": loss_call(rr.get("rating")),
                     "key_count": r.get("gt_count"), "dentist_count": rr.get("teeth_count"),
                     "conf": rr.get("confidence"), "note": rr.get("note", "")})
    df = pd.DataFrame(rows)

    def rate(bucket):
        b = df[(df.bucket == bucket) & (df.dentist_loss.notna())]
        k = int(b.dentist_loss.sum()); n = len(b)
        return k, n

    # --- controls ---
    kp, np_ = rate("KEY_POS")
    an, nn = rate("AGREE_NORM")
    print("CONTROLS (calibration):")
    print(f"  KEY_POS   (key says loss)   -> dentist sees loss on {kp}/{np_}   [want HIGH: dentist detects real loss]")
    print(f"  AGREE_NORM(all say normal)  -> dentist sees loss on {an}/{nn}   [want LOW: dentist not over-calling]")

    # --- the test ---
    kd, nd = rate("DISAGREE")
    p15 = binom_tail(nd, kd, 0.15) if nd else 1.0
    p10 = binom_tail(nd, kd, 0.10) if nd else 1.0
    print("\nADJUDICATION (DISAGREE: key='no loss', both models='loss'):")
    print(f"  dentist CONFIRMS bone loss on {kd}/{nd} images")
    print(f"  one-sided binomial vs rater noise:  P(>= {kd} | p0=.15) = {p15:.3f}   P(>= {kd} | p0=.10) = {p10:.3f}")
    if nd:
        if kd / nd >= 5 / 12 and p15 < 0.05:
            verdict = "KEY BIAS: the benchmark key under-reports bone loss (models were right on these)."
        elif kd <= max(1, int(0.25 * nd)):
            verdict = "MODEL OVER-CALL: the key looks sound; our models over-call bone loss."
        else:
            verdict = "MIXED / underpowered: neither hypothesis is clearly supported."
        print(f"  => {verdict}")

    # --- tooth-count key audit ---
    def as_num(x):
        try:
            return float(str(x).strip())
        except (ValueError, TypeError):
            return None
    cc = df.copy()
    cc["kc"] = cc.key_count.map(as_num)
    cc["dc"] = cc.dentist_count.map(as_num)
    cc = cc.dropna(subset=["kc", "dc"])
    if len(cc):
        cc["diff"] = cc.dc - cc.kc
        big = cc[cc["diff"].abs() >= 4]
        print("\nTOOTH-COUNT KEY AUDIT (dentist's count vs the benchmark's key count):")
        print(f"  mean |dentist - key| = {cc['diff'].abs().mean():.1f} teeth over {len(cc)} images")
        print(f"  images differing by >= 4 teeth (key likely over-counts): {len(big)}/{len(cc)}")
        for _, r in cc.sort_values("diff").iterrows():
            flag = "  <== KEY LIKELY WRONG" if abs(r["diff"]) >= 4 else ""
            tag = "[count-test] " if r.bucket == "COUNT_TEST" else ""
            print(f"    {tag}{r.image:14s} key={int(r.kc):2d}  dentist={int(r.dc):2d}  diff={r['diff']:+.0f}{flag}")

    print("\nper-image (bone):")
    for _, r in df.sort_values(["bucket", "item_id"]).iterrows():
        d = {True: "LOSS", False: "none", None: "n/a"}[r.dentist_loss]
        print(f"  {r.item_id} {r.bucket:11s} key={r.key:4s} dentist={d:4s} conf={r.conf or '-':6s} {str(r.note)[:50]}")


if __name__ == "__main__":
    main()
