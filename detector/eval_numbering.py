"""
Score the geometric FDI numbering (detector/number_teeth.py) against the benchmark's own
answers. No hand labels are involved anywhere — the ground truth here is the benchmark's
reference text, which names wisdom teeth by FDI code, and in two cases gives their boxes.

Two readouts:

  A. WISDOM-TOOTH SET, over every image whose reference answer names the third molars
     present ("Four wisdom teeth are detected: #18, #28, #38, and #48"). We predict the set
     of codes ending in 8 and compare. This is the number that decides whether step 1
     unlocks the 65 wisdom-teeth benchmark questions.

  B. BOX-LEVEL CODE, on the images whose reference answer carries `box_2d` + `tooth_id` for
     each wisdom tooth. Match each reference box to our nearest detection by IoU and ask
     whether we gave it the same code. Precise, but only a couple of images have it.

Run:  python detector/eval_numbering.py --weights best.pt
"""
import argparse
import base64
import io
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from number_teeth import assign_fdi

BOX_RE = re.compile(r'box_2d"?\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
ID_RE = re.compile(r'tooth_id"?\s*:\s*"?(\d{2})')
# "#18, #28, #38, and #48" / "(18, #28, #38)" -> the third molars the reference says are there
CODE_RE = re.compile(r"#?\b([1-4]8)\b")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--data", default="data/open_ended.parquet")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="results/detector/numbering_wisdom.csv")
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    op = pd.read_parquet(args.data)
    wis = op[op.question.str.contains("wisdom", case=False)]

    # A reference states the wisdom teeth present only when it lists codes; "no wisdom teeth"
    # answers are kept as the empty set, which is a real and scoreable case.
    truth = {}
    for _, r in wis.iterrows():
        a = str(r.answer)
        if "box_2d" in a:
            continue
        codes = set(CODE_RE.findall(a))
        neg = re.search(r"\bno\b[^.]{0,40}\bwisdom\b|\bwisdom teeth\b[^.]{0,20}\b(absent|missing|not)\b", a, re.I)
        if codes:
            truth.setdefault(r.image_name, set()).update(codes)
        elif neg:
            truth.setdefault(r.image_name, set())
    print(f"{len(truth)} images whose reference answer settles which wisdom teeth are present")

    imgs = op.drop_duplicates("image_name").set_index("image_name")["image"]
    model = YOLO(args.weights)

    rows = []
    for name, gt in sorted(truth.items()):
        blob = base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", str(imgs[name])))
        im = Image.open(io.BytesIO(blob)).convert("RGB")
        r = model.predict(im, imgsz=args.imgsz, conf=args.conf, agnostic_nms=True, verbose=False)[0]
        boxes = [tuple(round(v) for v in b) for b in r.boxes.xyxy.tolist()]
        pred = {c for c in assign_fdi(boxes) if c and c.endswith("8")}
        rows.append(dict(image=name, n_boxes=len(boxes),
                         truth="|".join(sorted(gt)), pred="|".join(sorted(pred)),
                         exact_set=pred == gt, n_truth=len(gt), n_pred=len(pred),
                         count_ok=len(pred) == len(gt),
                         hit=len(pred & gt), miss=len(gt - pred), extra=len(pred - gt)))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    n = len(df)
    tp, fn, fp = df.hit.sum(), df.miss.sum(), df.extra.sum()
    print(f"\nA. WISDOM-TOOTH SET, {n} images")
    print(f"   exact set match      : {df.exact_set.sum()}/{n}  ({100*df.exact_set.mean():.0f}%)")
    print(f"   right COUNT of them  : {df.count_ok.sum()}/{n}  ({100*df.count_ok.mean():.0f}%)"
          f"   <- what the 'how many wisdom teeth' questions need")
    print(f"   per-tooth  precision {tp/(tp+fp) if tp+fp else 0:.2f}   recall {tp/(tp+fn) if tp+fn else 0:.2f}")
    print(f"   worst rows:")
    for _, r in df[~df.exact_set].head(6).iterrows():
        print(f"     {r.image}  truth[{r.truth}]  pred[{r.pred}]")

    # ---- B. box-level ------------------------------------------------------------
    box_rows = []
    for _, r in wis[wis.answer.astype(str).str.contains("box_2d")].iterrows():
        a = str(r.answer)
        pairs = list(zip([tuple(map(int, g)) for g in BOX_RE.findall(a)], ID_RE.findall(a)))
        if not pairs:
            continue
        blob = base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", str(imgs[r.image_name])))
        im = Image.open(io.BytesIO(blob)).convert("RGB")
        rr = model.predict(im, imgsz=args.imgsz, conf=args.conf, agnostic_nms=True, verbose=False)[0]
        boxes = [tuple(round(v) for v in b) for b in rr.boxes.xyxy.tolist()]
        codes = assign_fdi(boxes)
        for gb, gid in pairs:
            j = max(range(len(boxes)), key=lambda k: iou(boxes[k], gb)) if boxes else None
            box_rows.append(dict(image=r.image_name, gt_code=gid,
                                 iou=round(iou(boxes[j], gb), 2) if j is not None else 0.0,
                                 ours=codes[j] if j is not None else None))
    if box_rows:
        b = pd.DataFrame(box_rows)
        matched = b[b.iou >= 0.3]
        print(f"\nB. BOX-LEVEL, {len(b)} reference wisdom-tooth boxes on {b.image.nunique()} images")
        print(f"   detector found the tooth (IoU>=0.3): {len(matched)}/{len(b)}")
        if len(matched):
            print(f"   and gave it the right code        : {(matched.ours == matched.gt_code).sum()}/{len(matched)}")
        print(b.to_string(index=False))
    print(f"\nper-image -> {args.out}")


if __name__ == "__main__":
    main()
