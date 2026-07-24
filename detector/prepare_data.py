"""
Stage 0 — build a clean, YOLO-format DENTEX tooth-detection dataset (set (b),
quadrant_enumeration) and write it to disk (typically a Google Drive path so it
persists across Colab sessions).

- Streams set (b) out of the 10.9 GB DENTEX training zip via HTTP range requests
  (no full download), OR reads a locally downloaded training_data.zip if you pass one.
- Cleans the ~3% annotation noise: de-duplicates repeated FDI boxes, drops images
  with an impossible tooth count.
- Converts COCO boxes -> YOLO format (class + normalized xywh), 32 FDI classes.
- Splits train/val and writes dentex.yaml.

Run:
  python detector/prepare_data.py --out /content/drive/MyDrive/dentex_yolo
  python detector/prepare_data.py --out ./dentex_yolo --source /path/to/training_data.zip
"""
import argparse
import io
import json
import os
import random
from collections import defaultdict

# The 32 FDI codes in a fixed order -> class ids 0..31 (11..18, 21..28, 31..38, 41..48)
FDI = [f"{q}{t}" for q in (1, 2, 3, 4) for t in range(1, 9)]
FDI2CLS = {code: i for i, code in enumerate(FDI)}
ENUM = "training_data/quadrant_enumeration"
HF_URL = ("https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/"
          "DENTEX/training_data.zip")


def open_zip(source):
    """Return a zip-like object with .read(name). URL -> streamed; local path -> zipfile."""
    if source.startswith("http"):
        from remotezip import RemoteZip
        return RemoteZip(source)
    import zipfile
    return zipfile.ZipFile(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dataset dir (e.g. a Google Drive path)")
    ap.add_argument("--source", default=HF_URL, help="DENTEX training zip URL or local .zip path")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-teeth", type=int, default=10, help="skip images with fewer boxes (broken)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image
    z = open_zip(args.source)
    ann = json.loads(z.read(f"{ENUM}/train_quadrant_enumeration.json"))
    q = {c["id"]: str(c["name"]) for c in ann["categories_1"]}   # quadrant 1-4
    t = {c["id"]: str(c["name"]) for c in ann["categories_2"]}   # tooth 1-8
    info = {i["id"]: i for i in ann["images"]}
    per = defaultdict(list)
    for a in ann["annotations"]:
        per[a["image_id"]].append((q[a["category_id_1"]] + t[a["category_id_2"]], a["bbox"]))

    ids = sorted(per)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    val_ids = set(ids[: int(len(ids) * args.val_frac)])

    for split in ("train", "val"):
        os.makedirs(f"{args.out}/images/{split}", exist_ok=True)
        os.makedirs(f"{args.out}/labels/{split}", exist_ok=True)

    kept = skipped = 0
    for img_id in ids:
        fn = info[img_id]["file_name"]
        # de-duplicate repeated FDI codes (the 3% annotation noise), keep first box per code
        seen, boxes = set(), []
        for code, bbox in per[img_id]:
            if code in seen or code not in FDI2CLS:
                continue
            seen.add(code)
            boxes.append((code, bbox))
        if not (args.min_teeth <= len(boxes) <= 32):
            skipped += 1
            continue

        split = "val" if img_id in val_ids else "train"
        raw = z.read(f"{ENUM}/xrays/{fn}")
        W, H = Image.open(io.BytesIO(raw)).size
        with open(f"{args.out}/images/{split}/{fn}", "wb") as f:
            f.write(raw)

        lines = []
        for code, (x, y, w, h) in boxes:
            xc, yc, ww, hh = (x + w / 2) / W, (y + h / 2) / H, w / W, h / H
            lines.append(f"{FDI2CLS[code]} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
        stem = os.path.splitext(fn)[0]
        with open(f"{args.out}/labels/{split}/{stem}.txt", "w") as f:
            f.write("\n".join(lines))
        kept += 1
        if kept % 50 == 0:
            print(f"  prepared {kept} images...", flush=True)

    with open(f"{args.out}/dentex.yaml", "w") as f:
        f.write(f"path: {args.out}\ntrain: images/train\nval: images/val\n"
                f"nc: {len(FDI)}\nnames: {FDI}\n")
    n_val = sum(1 for i in ids if i in val_ids)
    print(f"\ndone: {kept} images kept ({kept - n_val + skipped} skipped-or-train), "
          f"{skipped} skipped for bad count.")
    print(f"wrote {args.out}/dentex.yaml  ({len(FDI)} classes)")


if __name__ == "__main__":
    main()
