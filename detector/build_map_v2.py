"""
Build the v2 tooth map for the 100 MMOral panoramics: boxes from the single-class detector,
FDI codes from the position-calibrated numbering.

This replaces reference/mmoral_map.json, which came from v1's 32-class head. That map is what
the earlier injection experiments were fed, and its numbering was ~75% wrong on MMOral -- the
reason those experiments could not tell whether a model can use a tooth map or merely could
not use a WRONG one. This map is the corrected input for that question.

Run:  python detector/build_map_v2.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best2.pt")
    ap.add_argument("--prior", default="detector/arch_prior.json")
    ap.add_argument("--data", default="data/open_ended.parquet")
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--out", default="reference/mmoral_map_v2.json")
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    prior = json.load(open(args.prior))["median"]
    op = pd.read_parquet(args.data)
    images = op.drop_duplicates("image_name").set_index("image_name")["image"]
    model = YOLO(args.weights)

    out = {}
    for name, b64 in images.items():
        blob = base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", str(b64)))
        im = Image.open(io.BytesIO(blob)).convert("RGB")
        r = model.predict(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                          agnostic_nms=True, verbose=False)[0]
        boxes = [[round(v) for v in b] for b in r.boxes.xyxy.tolist()]
        confs = [round(float(c), 3) for c in r.boxes.conf.tolist()]
        codes = assign_fdi_positional([tuple(b) for b in boxes], prior)
        teeth = [{"fdi": c, "conf": cf, "box": b}
                 for c, cf, b in zip(codes, confs, boxes) if c]
        out[name] = {"count": len(boxes), "teeth": sorted(teeth, key=lambda t: t["fdi"])}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    n_num = sum(len(v["teeth"]) for v in out.values())
    n_box = sum(v["count"] for v in out.values())
    print(f"{len(out)} images, {n_box} boxes, {n_num} numbered "
          f"({100*n_num/max(n_box,1):.0f}%) -> {args.out}")
    print(f"provenance: {args.weights} @ imgsz={args.imgsz} conf={args.conf} iou={args.iou}, "
          f"positional numbering from {args.prior}")


if __name__ == "__main__":
    main()
