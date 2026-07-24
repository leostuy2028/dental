"""
Stage 4 — run the trained detector on the 100 MMOral panoramics and check
cross-scanner transfer: does the detector's tooth count agree with the counts we
trust (the dentist-confirmed reference counts)?

Runs locally (CPU is fine, seconds for 100 images) from the dental repo, using the
weights you trained on Colab and downloaded from Drive.

Run (from the dental repo root):
  python detector/infer_mmoral.py --weights best.pt
"""
import argparse
import base64
import io
import os
import re


def gt_count(text):
    m = re.search(r"\b(\d{1,2})\s+teeth\s+(?:are\s+)?(?:visualized|present|detected)", str(text), re.I)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/open_ended.parquet")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="results/detector/mmoral_counts.csv")
    ap.add_argument("--map-out", default=None,
                    help="also dump the full FDI map (per tooth: fdi+box) as JSON here")
    args = ap.parse_args()

    import pandas as pd
    from PIL import Image
    from ultralytics import YOLO
    from detect_utils import count_teeth, detect_map

    op = pd.read_parquet(args.data)
    # per-image trusted reference count (from any reference that states one)
    op["gtc"] = op.answer.map(gt_count)
    ref = (op.dropna(subset=["gtc"]).groupby("image_name")["gtc"]
           .agg(lambda s: int(s.mode().iloc[0])).to_dict())
    images = op.drop_duplicates("image_name").set_index("image_name")["image"]

    model = YOLO(args.weights)
    rows, tmap = [], {}
    for name, b64 in images.items():
        raw = base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", str(b64)))
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        pred_n = count_teeth(model, img, args.imgsz, args.conf)   # agnostic NMS, physical teeth
        if args.map_out:
            tmap[name] = {"count": pred_n,   # validated physical count (agnostic NMS)
                          "teeth": detect_map(model, img, args.imgsz, args.conf)}
        rows.append({"image": name, "detector_count": pred_n,
                     "ref_count": ref.get(name, ""),
                     "diff": (pred_n - ref[name]) if name in ref else ""})

    import pandas as pd
    df = pd.DataFrame(rows)
    scored = df[df.ref_count != ""].copy()
    scored["absdiff"] = (scored.detector_count - scored.ref_count).abs()
    n = len(scored)
    print(f"MMOral cross-scanner check ({n} images with a trusted reference count):")
    print(f"  exact match:        {(scored.absdiff == 0).sum()}/{n} "
          f"({(scored.absdiff == 0).mean() * 100:.0f}%)")
    print(f"  within 1 tooth:     {(scored.absdiff <= 1).sum()}/{n} "
          f"({(scored.absdiff <= 1).mean() * 100:.0f}%)")
    print(f"  mean |detector - reference|: {scored.absdiff.mean():.2f} teeth")
    print("\n  If this is poor, the DENTEX->MMOral scanner gap is real; "
          "fine-tune on a few hand/dentist-labeled MMOral images before trusting it.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nper-image detector vs reference counts -> {args.out}")
    if args.map_out:
        import json
        with open(args.map_out, "w") as f:
            json.dump(tmap, f)
        print(f"full FDI map ({len(tmap)} images) -> {args.map_out}")


if __name__ == "__main__":
    main()
