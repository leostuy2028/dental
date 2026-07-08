"""
Validate 768px downscaling by reproducing the §5.4 (E11) primer experiment at low
resolution. No API. Compares the two new 768px arms to the committed full-res arms.

Success = 768px reproduces the baseline accuracy and the rescue signal within noise,
with high per-item agreement and no pathology-specific collapse. Then exploration
runs can use --max-image-px 768 (~8x cheaper Gemini images); paper numbers stay full-res.

Usage: python paper_analysis/resolution_check.py
"""
import os
import sys
import math
import pandas as pd
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIR = "results/closed_ended/knowledge_context"
F = {
    "full_noctx": f"{DIR}/gemini-3.5-flash__coax-direct-noctx__e11sel__n130.csv",
    "full_ctx":   f"{DIR}/gemini-3.5-flash__coax-direct-ctx-opgprimer__e11sel__n130.csv",
    "px768_noctx": f"{DIR}/gemini-3.5-flash__coax-direct-noctx-px768__e11sel__n130.csv",
    "px768_ctx":   f"{DIR}/gemini-3.5-flash__coax-direct-ctx-opgprimer-px768__e11sel__n130.csv",
}


def load(repo, k):
    return pd.read_csv(os.path.join(repo, F[k])).set_index("index")


def gem_tokens(w, h):
    if max(w, h) <= 384:
        return 258
    return math.ceil(w / 768) * math.ceil(h / 768) * 258


def signal(noctx, ctx, idx):
    a = noctx.loc[idx, "correct"].astype(bool)
    b = ctx.loc[idx, "correct"].astype(bool)
    rescued = int((~a & b).sum())
    broke = int((a & ~b).sum())
    return {"noctx_acc": round(100 * a.mean(), 1), "ctx_acc": round(100 * b.mean(), 1),
            "rescued": rescued, "broke": broke, "net": rescued - broke}


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = {k: load(repo, k) for k in F}
    idx = sorted(set.intersection(*[set(v.index) for v in d.values()]))
    print("=" * 60)
    print(f"768px vs full-res — §5.4 E11 reproduction ({len(idx)} items)")
    print("=" * 60)

    print("\nAccuracy:")
    for arm in ["noctx", "ctx"]:
        fa = d[f"full_{arm}"].loc[idx, "correct"].mean() * 100
        pa = d[f"px768_{arm}"].loc[idx, "correct"].mean() * 100
        print(f"  {arm:6}: full-res {fa:5.1f}%   768px {pa:5.1f}%   (Δ {pa-fa:+.1f})")

    print("\nSignal (noctx -> +primer):")
    for res in ["full", "px768"]:
        s = signal(d[f"{res}_noctx"], d[f"{res}_ctx"], idx)
        print(f"  {res:6}: {s['noctx_acc']}% -> {s['ctx_acc']}%   rescued {s['rescued']} / broke {s['broke']} (net {s['net']:+d})")

    print("\nPer-item agreement with full-res (does downscaling change the answer?):")
    cat = d["full_noctx"].loc[idx, "category"]
    for arm in ["noctx", "ctx"]:
        f_pred = d[f"full_{arm}"].loc[idx, "predicted"].astype(str)
        p_pred = d[f"px768_{arm}"].loc[idx, "predicted"].astype(str)
        agree = (f_pred == p_pred)
        print(f"  {arm:6}: {100*agree.mean():.0f}% ({int(agree.sum())}/{len(idx)})")
        if arm == "noctx":
            for dim in ["Teeth", "Patho", "HisT", "Jaw"]:
                m = cat.str.contains(dim)
                if m.sum():
                    print(f"          {dim:6} {100*agree[m].mean():.0f}% (n={int(m.sum())})")

    # ---- 1024px cap on the E11 no-context arm (within-session clean agreement) ----
    p1024 = os.path.join(repo, DIR, "gemini-3.5-flash__coax-direct-noctx-px1024__e11sel__n130.csv")
    if os.path.exists(p1024):
        x = pd.read_csv(p1024).set_index("index")
        cx = sorted(set(x.index) & set(idx))
        f_pred = d["full_noctx"].loc[cx, "predicted"].astype(str)
        agree = (f_pred == x.loc[cx, "predicted"].astype(str))
        print(f"\nE11 no-context agreement vs full-res (within-session, clean):")
        print(f"  1024px: {100*agree.mean():.0f}% ({int(agree.sum())}/{len(cx)})   "
              f"acc {x.loc[cx,'correct'].mean()*100:.1f}%   [768px was 76%, acc 33.8%]")

    # ---- whole-491: 768px vs committed full-res 58.0% (1-day-old baseline -> drift caveat) ----
    fw = os.path.join(repo, "results/closed_ended/position_bias/gemini-3.5-flash__coax-direct-k0__shuffled__n491.csv")
    pw = os.path.join(repo, "results/closed_ended/resolution/gemini-3.5-flash__coax-direct-noctx-px768__shuffled__n491.csv")
    if os.path.exists(pw):
        a = pd.read_csv(fw).set_index("index")
        b = pd.read_csv(pw).set_index("index")
        ci = sorted(set(a.index) & set(b.index))
        agree = (a.loc[ci, "predicted"].astype(str) == b.loc[ci, "predicted"].astype(str))
        print(f"\nWHOLE-491 (unbiased), 768px vs full-res:")
        print(f"  accuracy: full-res {a.loc[ci,'correct'].mean()*100:.1f}%  ->  768px {b.loc[ci,'correct'].mean()*100:.1f}%")
        print(f"  per-item agreement: {100*agree.mean():.0f}% ({int(agree.sum())}/{len(ci)})  "
              f"[CAVEAT: full-res baseline is 1 day old, so this mixes resolution + drift]")

    # cost saving estimate on the actual images
    cl = pd.read_parquet(os.path.join(repo, "data/closed_ended_e11_sel.parquet"))
    import base64, io
    from PIL import Image
    full_t = px_t = 0
    for b in cl["image"]:
        w, h = Image.open(io.BytesIO(base64.b64decode(str(b)))).size
        full_t += gem_tokens(w, h)
        s = min(1, 768 / max(w, h))
        px_t += gem_tokens(int(w * s), int(h * s))
    print(f"\nEst. Gemini image tokens over the {len(cl)} items: full {full_t} -> 768px {px_t} "
          f"({full_t/px_t:.1f}x cheaper images)")


if __name__ == "__main__":
    main()
