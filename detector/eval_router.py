"""
Step 3 — measure what the router buys, across base models of different strength.

NO API CALLS. Every model's per-item correctness is already in a committed CSV, and routing
does not change the prompt for the questions it does not route, so those answers are unchanged
BY CONSTRUCTION. The routed score is therefore exact, not an approximation:

    routed items  -> the detector's correctness
    everything else -> that model's committed correctness

The point of the design is the CURVE, not any single number. Testing only against our strongest
config is the hardest possible test and is why an earlier routing attempt bought +0.97. The
expected shape is a large lift on a weak model and a small one on a strong model, i.e. the
component supplies exactly what small models lack.

Run:  python detector/eval_router.py --weights best2.pt
"""
import argparse
import base64
import glob
import io
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from number_teeth import assign_fdi_positional
from router import answer_mcq, route

OPTS = ["option1", "option2", "option3", "option4"]


def detector_readings(args, images):
    """{image: {'count': int, 'wisdom': [codes]}} — one detector pass per image, cached."""
    if os.path.exists(args.cache) and not args.refresh:
        return json.load(open(args.cache))
    from PIL import Image
    from ultralytics import YOLO
    prior = json.load(open(args.prior))["median"]
    model = YOLO(args.weights)
    out = {}
    for name, b64 in images.items():
        blob = base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", str(b64)))
        im = Image.open(io.BytesIO(blob)).convert("RGB")
        r = model.predict(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                          agnostic_nms=True, verbose=False)[0]
        boxes = [tuple(round(v) for v in b) for b in r.boxes.xyxy.tolist()]
        codes = assign_fdi_positional(boxes, prior)
        out[name] = {"count": len(boxes),
                     "wisdom": sorted(c for c in codes if c and c.endswith("8"))}
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    json.dump(out, open(args.cache, "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best2.pt")
    ap.add_argument("--prior", default="detector/arch_prior.json")
    ap.add_argument("--closed", default="data/closed_ended_shuffled.parquet",
                    help="shuffled key is the project default (RESEARCH_PLAN §1.0)")
    ap.add_argument("--open", default="data/open_ended.parquet")
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--cache", default="results/detector/router_readings.json")
    ap.add_argument("--out", default="results/detector/router_lift.csv")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    op = pd.read_parquet(args.open)
    images = op.drop_duplicates("image_name").set_index("image_name")["image"]
    det = detector_readings(args, images)

    cl = pd.read_parquet(args.closed)
    img_col = "file_name" if "file_name" in cl.columns else "image_id"
    cl["kind"] = cl.question.map(route)

    # ---- what the router selects -------------------------------------------------
    sel = cl[cl.kind.notna()].copy()
    print(f"router selects {len(sel)} of {len(cl)} closed questions "
          f"({100*len(sel)/len(cl):.1f}%)")
    print(sel.kind.value_counts().to_string())

    # ---- the detector's answer on those --------------------------------------------
    dec = {}
    for _, r in sel.iterrows():
        d = det.get(r[img_col])
        if d is None:
            continue
        pick = answer_mcq(r.kind, d, [r[c] for c in OPTS])
        if pick is None:                      # non-numeric options -> hand back to the VLM
            continue
        dec[r["index"]] = "ABCD"[pick] == str(r.answer).strip().upper()
    print(f"detector answers {len(dec)} of them; "
          f"correct on {100*sum(dec.values())/max(len(dec),1):.1f}%")
    handed_back = len(sel) - len(dec)
    if handed_back:
        print(f"({handed_back} handed back to the VLM: no detector reading, or "
              f"non-numeric options)")

    # ---- combine with every committed model run ------------------------------------
    rows = []
    for f in sorted(glob.glob("results/closed_ended/**/*.csv", recursive=True)):
        if "superseded" in f or "n491" not in f:
            continue
        d = pd.read_csv(f, keep_default_na=False)
        if "correct" not in d.columns or "index" not in d.columns:
            continue
        d["base_ok"] = d.correct.astype(str).str.lower().isin(["true", "1"])
        d["routed_ok"] = [dec.get(i, b) for i, b in zip(d["index"], d.base_ok)]
        touched = d[d["index"].isin(dec)]
        rows.append(dict(
            run=os.path.basename(f).replace(".csv", ""),
            n=len(d),
            base=100 * d.base_ok.mean(),
            routed=100 * d.routed_ok.mean(),
            lift=100 * (d.routed_ok.mean() - d.base_ok.mean()),
            n_routed=len(touched),
            model_on_routed=100 * touched.base_ok.mean() if len(touched) else float("nan"),
            det_on_routed=100 * touched.routed_ok.mean() if len(touched) else float("nan"),
            api_calls_saved=len(touched),
        ))
    res = pd.DataFrame(rows).sort_values("base")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res.to_csv(args.out, index=False)

    print("\nLIFT BY BASE-MODEL STRENGTH  (weakest first — the curve is the result)")
    print(f"{'run':58}{'base':>7}{'routed':>8}{'lift':>7}{'model/det on routed':>22}")
    for _, r in res.iterrows():
        print(f"{r.run[:56]:58}{r.base:6.1f}%{r.routed:7.1f}%{r.lift:+6.1f}"
              f"{r.model_on_routed:12.0f}% ->{r.det_on_routed:4.0f}%")
    print(f"\nEvery routed question is an API call not made: "
          f"{res.api_calls_saved.iloc[0]}/{res.n.iloc[0]} "
          f"({100*res.api_calls_saved.iloc[0]/res.n.iloc[0]:.0f}% of the closed half)")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
