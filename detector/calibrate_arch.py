"""
Calibrate the arch-position prior on DENTEX's real expert FDI labels, then compare the two
numbering methods on images that were never used to calibrate.

Why this exists. Ordinal numbering (number_teeth.assign_fdi) counts outward from the midline,
so one undetected tooth or a slightly-off midline shifts every index behind it and the third
molar lands on 7. It scores 39% on MMOral wisdom teeth, and better detection did not move it
(37% -> 39% from v1 to v2), which rules out box quality as the cause.

Positional numbering (assign_fdi_positional) instead asks where a tooth sits: arc length from
the midline, in tooth-widths. A third molar sits at its own distance whether or not the
premolar in front of it is present.

The prior is eight numbers per quadrant and comes from DENTEX set (b) — real expert
annotations, no assistant-authored labels anywhere. Only the 5.9 MB annotation JSON is
fetched, not the 10.9 GB of images.

Split hygiene: this rebuilds prepare_data.py's QC gate and its seeded shuffle exactly, so the
prior is fitted on the 557 TRAIN images and scored on the 61 VAL images the detector never
trained on either.

Run:  python detector/calibrate_arch.py
"""
import argparse
import json
import os
import random
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from number_teeth import (QUADRANT, _arc_len, _centre, _fit_quadratic, assign_fdi,
                          assign_fdi_positional, find_midline, split_arches)
from prepare_data import ENUM, HF_URL, qc_image

ANN = f"{ENUM}/train_quadrant_enumeration.json"


def load_annotations(cache, source):
    if os.path.exists(cache):
        return json.load(open(cache))
    from remotezip import RemoteZip
    print(f"streaming {ANN} out of the DENTEX zip (~6 MB, images not touched)...", flush=True)
    with RemoteZip(source) as z:
        raw = z.read(ANN)
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with open(cache, "wb") as f:
        f.write(raw)
    return json.loads(raw)


def per_image(ann):
    """{image_id: [(fdi_code, xyxy_box), ...]} plus the clean-image id list, QC applied."""
    q = {c["id"]: str(c["name"]) for c in ann["categories_1"]}
    t = {c["id"]: str(c["name"]) for c in ann["categories_2"]}
    per = defaultdict(list)
    for a in ann["annotations"]:
        code = q[a["category_id_1"]] + t[a["category_id_2"]]
        x, y, w, h = a["bbox"]
        per[a["image_id"]].append((code, (x, y, x + w, y + h)))
    clean = [i for i in sorted(per) if qc_image([c for c, _ in per[i]])[0] == "ok"]
    return per, clean


def positions_with_truth(items):
    """[(true_code, quadrant, distance in tooth-widths), ...] for one image.

    Geometry uses the SAME estimators inference uses (fitted arches, estimated midline), so
    the prior is calibrated in the frame it will be applied in, absorbing the estimator's own
    bias rather than assuming a perfect midline.
    """
    boxes = [b for _, b in items]
    codes = [c for c, _ in items]
    centres = [_centre(b) for b in boxes]
    side = split_arches(centres)
    mid = find_midline(centres, side)
    widths = sorted(abs(b[2] - b[0]) for b in boxes)
    unit = widths[len(widths) // 2] or 1.0
    fits = {}
    for arch in ("upper", "lower"):
        pts = [p for p, s in zip(centres, side) if s == arch]
        fits[arch] = _fit_quadratic(pts) if len(pts) >= 3 else (0.0, 0.0, 0.0)
    out = []
    for code, (cx, _), s in zip(codes, centres, side):
        hand = "left" if cx < mid else "right"
        d = abs(_arc_len(fits[s], mid, cx)) / unit
        out.append((code, QUADRANT[(s, hand)], d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="reference/dentex_quadrant_enumeration.json")
    ap.add_argument("--source", default=HF_URL)
    ap.add_argument("--out", default="detector/arch_prior.json")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ann = load_annotations(args.cache, args.source)
    per, clean = per_image(ann)
    rng = random.Random(args.seed)
    order = clean[:]
    rng.shuffle(order)
    n_val = int(len(order) * args.val_frac)
    val_ids, train_ids = set(order[:n_val]), order[n_val:]
    print(f"{len(clean)} clean images -> {len(train_ids)} calibrate / {len(val_ids)} held out")

    # ---- fit the prior on TRAIN only --------------------------------------------
    by_index = defaultdict(list)
    agree = tot = 0
    for i in train_ids:
        for code, q, d in positions_with_truth(per[i]):
            by_index[code[1]].append(d)
            tot += 1
            agree += int(code[0]) == q          # did our geometry recover the true quadrant?
    prior = {k: round(st.median(v), 4) for k, v in sorted(by_index.items())}
    spread = {k: round(st.pstdev(v), 3) for k, v in sorted(by_index.items())}
    print(f"\nquadrant recovered from geometry on the calibration set: "
          f"{100*agree/tot:.1f}% of {tot} teeth")
    print("\nprior — distance from the midline, in tooth-widths:")
    for k in map(str, range(1, 9)):
        print(f"   index {k}:  median {prior[k]:5.2f}   sd {spread[k]:4.2f}   n={len(by_index[k])}")
    gaps = [prior[str(k + 1)] - prior[str(k)] for k in range(1, 8)]
    print(f"   step between neighbours: {[round(g,2) for g in gaps]}")
    if min(gaps) <= 0:
        print("   !! non-monotonic prior — positional numbering cannot work with this")

    json.dump({"unit": "tooth-widths of arc length from the midline",
               "source": "DENTEX set (b) quadrant_enumeration, expert labels",
               "calibrated_on": len(train_ids), "median": prior, "sd": spread},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")

    # ---- score both methods on the HELD-OUT images ------------------------------
    print(f"\nnumbering accuracy on {len(val_ids)} held-out DENTEX images "
          f"(scored against expert FDI labels, on the TRUE boxes — this isolates the "
          f"numbering method from detector error):")
    res = {}
    for label, fn in (("ordinal   (current)", lambda b: assign_fdi(b)),
                      ("positional (new)  ", lambda b: assign_fdi_positional(b, prior))):
        hit = tot_t = 0
        w_hit = w_tot = 0
        for i in val_ids:
            items = per[i]
            boxes = [b for _, b in items]
            truth = [c for c, _ in items]
            got = fn(boxes)
            for t, g in zip(truth, got):
                tot_t += 1
                hit += (t == g)
                if t.endswith("8"):
                    w_tot += 1
                    w_hit += (t == g)
        res[label] = (100 * hit / tot_t, 100 * w_hit / max(w_tot, 1))
        print(f"   {label}:  every tooth {100*hit/tot_t:5.1f}%    "
              f"third molars only {100*w_hit/max(w_tot,1):5.1f}%  (n={w_tot})")
    a, b = res["ordinal   (current)"], res["positional (new)  "]
    print(f"\n   change: every tooth {b[0]-a[0]:+.1f} pts, third molars {b[1]-a[1]:+.1f} pts")


if __name__ == "__main__":
    main()
