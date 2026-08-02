"""
Does the disease detector's IMPACTED class survive the move from DENTEX to MMOral?

On held-out DENTEX it is strong (mAP50 0.941, recall 0.939) — nearly the tooth detector's
level. The other three classes are not worth routing (Caries 0.563, Deep Caries 0.534,
Periapical Lesion 0.328), so only Impacted is scored here.

Cross-scanner transfer is genuinely open for this project. Counting transferred cleanly;
appearance-based FDI numbering collapsed. Impaction is a GEOMETRIC fact — an unerupted tooth
sitting below the occlusal plane at an odd angle — so it should behave like counting rather
than like numbering. That is a prediction, not a result, which is why this exists.

Ground truth is the benchmark's OWN reference answers, which name impacted teeth by FDI code
("#38 and #48 are impacted"). No assistant-authored labels anywhere.

Two readouts:
  A. WHICH TEETH are impacted — compose the impaction box with the tooth numbering
     (detector/number_teeth.assign_fdi_positional) by asking which numbered tooth each
     impaction box sits on. This is the composition the router would use.
  B. HOW MANY are impacted — the count alone, which needs no numbering and is therefore the
     more robust of the two.

Run:  python detector/eval_impaction.py
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

# Extracting the truth is where this measurement is easiest to get wrong, and it did go wrong
# once. A first attempt split on sentence-enders only, leaving whole multi-line report BLOCKS
# intact, so every FDI-looking number in a block containing "impacted" was swept in:
#   "26 teeth visualized ..."     -> read as tooth #26
#   "Deep caries in #36 and #47"  -> joined the impaction set because the same block later
#                                    said "Impacted wisdom teeth"
# On 016649 that produced truth {18,23,26,28,38} for an image with three wisdom teeth. Recall
# is scored against this set, so an inflated truth makes the detector look far worse than it
# is. Now: split on LINES, require the impaction claim in the same line, and refuse a bare
# number followed by "teeth"/"tooth" (that is a count, not a code).
FDI_RE = re.compile(r"#\s*([1-4][1-8])\b|\b([1-4][1-8])\b(?!\s*(?:teeth|tooth))")
LINE_SPLIT = re.compile(r"[\n;]|(?<=[.])\s+")
# one keyword set per DENTEX diagnosis class, matched against the benchmark's own answers
CLASS_RE = {
    "Impacted":          re.compile(r"impact", re.I),
    "Caries":            re.compile(r"caries|decay|cavit", re.I),
    "Deep Caries":       re.compile(r"deep caries", re.I),
    "Periapical Lesion": re.compile(r"periapical|lesion|abscess", re.I),
}
IMPACT_RE = CLASS_RE["Impacted"]
NEG_RE = re.compile(r"\bno\b[^.]{0,50}\bimpact|not impacted|non-impacted|all .{0,20}erupted", re.I)


def centre(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tooth-weights", default="best2.pt")
    ap.add_argument("--disease-weights", default="best_disease.pt")
    ap.add_argument("--prior", default="detector/arch_prior.json")
    ap.add_argument("--data", default="data/open_ended.parquet")
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--dis-conf", type=float, default=0.25,
                    help="confidence for the disease detector (its own scale, not the tooth one)")
    ap.add_argument("--cls", default="Impacted", choices=list(CLASS_RE),
                    help="which DENTEX diagnosis class to score")
    ap.add_argument("--out", default="results/detector/disease_mmoral.csv")
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    op = pd.read_parquet(args.data)
    kw = CLASS_RE[args.cls]

    # ---- ground truth from the benchmark's own answers --------------------------
    truth = {}
    for _, r in op.iterrows():
        a = str(r.answer)
        if not kw.search(a) or "box_2d" in a:
            continue
        # only sentences that actually assert impaction, so a passing mention does not count
        for sent in LINE_SPLIT.split(a):
            if not kw.search(sent):
                continue
            if NEG_RE.search(sent):
                truth.setdefault(r.image_name, set())
                continue
            codes = {g1 or g2 for g1, g2 in FDI_RE.findall(sent) if (g1 or g2)}
            if codes:
                truth.setdefault(r.image_name, set()).update(codes)
    print(f"[{args.cls}] {len(truth)} images whose reference answers name affected teeth by FDI code")
    n_pos = sum(1 for v in truth.values() if v)
    print(f"   of these, {n_pos} assert at least one, {len(truth)-n_pos} assert none")

    imgs = op.drop_duplicates("image_name").set_index("image_name")["image"]
    prior = json.load(open(args.prior))["median"]
    tooth = YOLO(args.tooth_weights)
    disease = YOLO(args.disease_weights)

    rows = []
    for name, gt in sorted(truth.items()):
        blob = base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", str(imgs[name])))
        im = Image.open(io.BytesIO(blob)).convert("RGB")

        t = tooth.predict(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                          agnostic_nms=True, verbose=False)[0]
        tboxes = [tuple(round(v) for v in b) for b in t.boxes.xyxy.tolist()]
        codes = assign_fdi_positional(tboxes, prior)

        d = disease.predict(im, imgsz=args.imgsz, conf=args.dis_conf, verbose=False)[0]
        imp = [tuple(round(v) for v in b)
               for b, c in zip(d.boxes.xyxy.tolist(), d.boxes.cls.tolist())
               if disease.names[int(c)] == args.cls]

        # compose: each impaction box takes the code of the numbered tooth nearest its centre
        pred = set()
        for ib in imp:
            cx, cy = centre(ib)
            best, bd = None, 1e9
            for tb, code in zip(tboxes, codes):
                if not code:
                    continue
                tx, ty = centre(tb)
                dd = (tx - cx) ** 2 + (ty - cy) ** 2
                if dd < bd:
                    best, bd = code, dd
            if best:
                pred.add(best)

        rows.append(dict(image=name, n_impact_boxes=len(imp),
                         truth="|".join(sorted(gt)), pred="|".join(sorted(pred)),
                         n_truth=len(gt), n_pred=len(pred),
                         count_ok=len(pred) == len(gt), exact_set=pred == gt,
                         hit=len(pred & gt), miss=len(gt - pred), extra=len(pred - gt)))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    n = len(df)
    tp, fn, fp = df.hit.sum(), df.miss.sum(), df.extra.sum()
    print(f"\nA. WHICH TEETH ARE IMPACTED (detection composed with numbering), {n} images")
    print(f"   exact set match : {df.exact_set.sum()}/{n} ({100*df.exact_set.mean():.0f}%)")
    print(f"   per-tooth  precision {tp/(tp+fp) if tp+fp else 0:.2f}  "
          f"recall {tp/(tp+fn) if tp+fn else 0:.2f}")
    print(f"\nB. HOW MANY ARE IMPACTED (detection alone, no numbering)")
    print(f"   right n findings: {df.count_ok.sum()}/{n} ({100*df.count_ok.mean():.0f}%)")
    print(f"   mean |error|    : {(df.n_pred-df.n_truth).abs().mean():.2f} findings")
    print(f"\n   worst rows:")
    for _, r in df[~df.exact_set].head(6).iterrows():
        print(f"     {r.image}  truth[{r.truth}]  pred[{r.pred}]")
    print(f"\nper-image -> {args.out}")


if __name__ == "__main__":
    main()
