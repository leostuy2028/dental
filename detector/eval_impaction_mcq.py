"""
Head to head on the benchmark's IMPACTION multiple-choice questions: the detector answering
directly, against what gemini-3.5-flash scores on the identical items.

The detector answers two shapes of question and refuses everything else:

  numeric options ("2" / "3" / "4")      -> how many teeth it detected as impacted
  FDI-code options ("#38 and #48")       -> the option whose code set best matches the set it
                                            detected, scored by overlap so a near-miss still
                                            picks the closest option rather than guessing

Anything whose options are neither (prose like "All wisdom teeth are erupted") is handed back
to the VLM, because the detector has no way to express that answer.

NO API CALLS: gemini's per-item correctness comes from a committed CSV, and the detector runs
on CPU.

Run:  python detector/eval_impaction_mcq.py
"""
import argparse
import base64
import io
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from number_teeth import assign_fdi_positional

OPTS = ["option1", "option2", "option3", "option4"]
IMPACT_Q = re.compile(r"impact", re.I)
NUM_RE = re.compile(r"^\s*(\d+)\s*$")
CODE_RE = re.compile(r"#?\b([1-4][1-8])\b")


def centre(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tooth-weights", default="best2.pt")
    ap.add_argument("--disease-weights", default="best_disease.pt")
    ap.add_argument("--prior", default="detector/arch_prior.json")
    ap.add_argument("--closed", default="data/closed_ended_shuffled.parquet")
    ap.add_argument("--open", default="data/open_ended.parquet")
    ap.add_argument("--baseline",
                    default="results/closed_ended/knowledge_context/"
                            "gemini-3.5-flash__coax-direct-primerv1+visualex-v2__shuffled__n491.csv")
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--dis-conf", type=float, default=0.25)
    ap.add_argument("--out", default="results/detector/impaction_mcq.csv")
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    cl = pd.read_parquet(args.closed)
    img_col = "file_name" if "file_name" in cl.columns else "image_id"
    q = cl[cl.question.str.contains(IMPACT_Q)].copy()
    print(f"{len(q)} closed questions mention impaction")

    # ---- detector readings, one pass per image ---------------------------------
    op = pd.read_parquet(args.open)
    imgs = op.drop_duplicates("image_name").set_index("image_name")["image"]
    prior = json.load(open(args.prior))["median"]
    tooth, disease = YOLO(args.tooth_weights), YOLO(args.disease_weights)
    reading = {}
    for name in sorted(set(q[img_col])):
        blob = base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", str(imgs[name])))
        im = Image.open(io.BytesIO(blob)).convert("RGB")
        t = tooth.predict(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                          agnostic_nms=True, verbose=False)[0]
        tb = [tuple(round(v) for v in b) for b in t.boxes.xyxy.tolist()]
        codes = assign_fdi_positional(tb, prior)
        d = disease.predict(im, imgsz=args.imgsz, conf=args.dis_conf, verbose=False)[0]
        imp = [tuple(round(v) for v in b)
               for b, c in zip(d.boxes.xyxy.tolist(), d.boxes.cls.tolist())
               if disease.names[int(c)] == "Impacted"]
        got = set()
        for ib in imp:
            cx, cy = centre(ib)
            best, bd = None, 1e18
            for b, code in zip(tb, codes):
                if not code:
                    continue
                tx, ty = centre(b)
                dd = (tx - cx) ** 2 + (ty - cy) ** 2
                if dd < bd:
                    best, bd = code, dd
            if best:
                got.add(best)
        reading[name] = got

    # ---- the detector answers what it can --------------------------------------
    rows = []
    for _, r in q.iterrows():
        got = reading.get(r[img_col], set())
        opts = [str(r[c]).strip() for c in OPTS]
        pick, mode = None, None
        if all(NUM_RE.match(o) for o in opts):
            vals = [int(NUM_RE.match(o).group(1)) for o in opts]
            pick = min(range(4), key=lambda i: (abs(vals[i] - len(got)), i))
            mode = "count"
        elif any(CODE_RE.search(o) for o in opts):
            sets = [set(CODE_RE.findall(o)) for o in opts]
            if any(sets):
                # overlap minus spurious codes, so the closest option wins rather than a guess
                pick = max(range(4), key=lambda i: (len(sets[i] & got) - 0.5 * len(sets[i] - got), -i))
                mode = "codes"
        rows.append(dict(index=r["index"], image=r[img_col], mode=mode,
                         detector_pred="ABCD"[pick] if pick is not None else "",
                         key=str(r.answer).strip().upper(),
                         detector_ok=(pick is not None and "ABCD"[pick] == str(r.answer).strip().upper())))
    det = pd.DataFrame(rows)

    # ---- gemini on the same items ----------------------------------------------
    G = pd.read_csv(args.baseline, keep_default_na=False)
    G["gemini_ok"] = G.correct.astype(str).str.lower().isin(["true", "1"])
    m = det.merge(G[["index", "gemini_ok"]], on="index")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    m.to_csv(args.out, index=False)

    ans = m[m["mode"].notna()]
    print(f"\ndetector can express an answer for {len(ans)} of {len(m)} "
          f"({len(m)-len(ans)} have prose options and go to the VLM)")
    print(f"\nHEAD TO HEAD on those {len(ans)}:")
    print(f"   detector          : {100*ans.detector_ok.mean():.1f}%")
    print(f"   gemini-3.5-flash  : {100*ans.gemini_ok.mean():.1f}%")
    print(f"   difference        : {100*(ans.detector_ok.mean()-ans.gemini_ok.mean()):+.1f} pts")
    b = int((ans.gemini_ok & ~ans.detector_ok).sum())
    c = int(((~ans.gemini_ok) & ans.detector_ok).sum())
    if b + c:
        from scipy import stats
        print(f"   McNemar p={stats.binomtest(c, b + c, 0.5).pvalue:.3f} "
              f"(detector-only-right {c}, gemini-only-right {b})")
    for mode in ("count", "codes"):
        s = ans[ans["mode"] == mode]
        if len(s):
            print(f"\n   {mode:6} questions  n={len(s):3d}   "
                  f"detector {100*s.detector_ok.mean():5.1f}%   gemini {100*s.gemini_ok.mean():5.1f}%")
    print(f"\nper-item -> {args.out}")


if __name__ == "__main__":
    main()
