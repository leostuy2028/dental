"""
Stage 0 — build a clean, YOLO-format DENTEX tooth-detection dataset (set (b),
quadrant_enumeration) and write it to disk (typically a Google Drive path so it
persists across Colab sessions).

DATA HYGIENE (the point of this script):
  Every image is run through qc_image(). Images with an ERRONEOUS tooth count —
  a duplicated FDI code (the same tooth boxed twice) or >32 boxes — are EXCLUDED
  entirely, not silently patched, so no bad labels enter training OR validation.
  The exclusions are recorded in qc_report.csv + excluded.json (the single blocklist
  any other process can read), and verify_build() then re-derives the invariant from what
  was actually written. Legit sparse/edentulous mouths are kept (a counter must handle
  them); only genuinely broken annotations are dropped.

  --single-class flattens every tooth to class 0, which DESTROYS the FDI codes, so the
  written labels can no longer prove "no tooth boxed twice" on their own. The codes are
  therefore recorded in fdi_codes.json and verify_build() checks them from there. The gate
  itself is unaffected either way: qc_image() runs on the raw codes before any flattening,
  and only clean images reach the write loop. detector/test_prepare_qc.py plants a
  duplicate tooth in both modes and asserts the build refuses it.

Run:
  python detector/prepare_data.py --out /content/drive/MyDrive/dentex_yolo
  python detector/prepare_data.py --out ./dentex_yolo --source /path/to/training_data.zip
"""
import argparse
import io
import json
import os
import random
from collections import Counter, defaultdict

# The 32 FDI codes in a fixed order -> class ids 0..31 (11..18, 21..28, 31..38, 41..48)
FDI = [f"{q}{t}" for q in (1, 2, 3, 4) for t in range(1, 9)]
FDI2CLS = {code: i for i, code in enumerate(FDI)}
ENUM = "training_data/quadrant_enumeration"
HF_URL = ("https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/"
          "DENTEX/training_data.zip")
MIN_BOXES = 4          # below this, the annotation is almost certainly broken (not a real sparse mouth)


def qc_image(codes):
    """codes = list of FDI strings for one image. Return (status, reason).
    'excluded' if the tooth count is erroneous; 'ok' otherwise."""
    dups = sorted(c for c, k in Counter(codes).items() if k > 1)
    if dups:
        return "excluded", "duplicate_fdi:" + ",".join(dups)
    if len(codes) > 32:                 # impossible without a dup; belt-and-suspenders
        return "excluded", "over_32_boxes"
    if len(codes) < MIN_BOXES:          # near-empty => broken annotation, not a real sparse mouth
        return "excluded", "too_few_boxes"
    return "ok", ""


def open_zip(source):
    if source.startswith("http"):
        from remotezip import RemoteZip
        return RemoteZip(source)
    import zipfile
    return zipfile.ZipFile(source)


def verify_build(out, single_class):
    """Re-derive the QC invariant from what was actually written, independently of qc_image.

    Flattening to one class destroys the FDI codes, so the written labels alone can no longer
    prove "no tooth boxed twice". The codes are therefore recorded in fdi_codes.json and
    checked from there, which keeps the SAME invariant enforced in both modes: every shipped
    image has MIN_BOXES..32 boxes, no repeated FDI code, and exactly as many label lines as
    it has codes.

    Returns a list of problems (empty when the build is clean).
    """
    import glob
    codes_by_stem = {os.path.splitext(fn)[0]: cs
                     for fn, cs in json.load(open(f"{out}/fdi_codes.json")).items()}
    bad = []
    for lf in sorted(glob.glob(f"{out}/labels/*/*.txt")):
        stem = os.path.splitext(os.path.basename(lf))[0]
        cls = [ln.split()[0] for ln in open(lf) if ln.strip()]
        codes = codes_by_stem.get(stem)
        if codes is None:
            bad.append(f"{lf}: shipped but not in the kept set")
            continue
        if len(set(codes)) != len(codes):
            dup = sorted(c for c, k in Counter(codes).items() if k > 1)
            bad.append(f"{lf}: duplicate tooth {dup} survived QC")
        if not (MIN_BOXES <= len(codes) <= 32):
            bad.append(f"{lf}: {len(codes)} boxes, outside {MIN_BOXES}..32")
        if len(cls) != len(codes):
            bad.append(f"{lf}: {len(cls)} label lines vs {len(codes)} codes")
        if single_class and set(cls) - {"0"}:
            bad.append(f"{lf}: non-zero class in single-class mode")
        if not single_class and len(cls) != len(set(cls)):
            bad.append(f"{lf}: duplicate class")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dataset dir (e.g. a Google Drive path)")
    ap.add_argument("--source", default=HF_URL, help="DENTEX training zip URL or local .zip path")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--single-class", action="store_true",
                    help="write every tooth as class 0 (nc=1). Numbering then comes from detector/number_teeth.py, not from the network. This is the step-2 build.")
    args = ap.parse_args()

    from PIL import Image
    z = open_zip(args.source)
    ann = json.loads(z.read(f"{ENUM}/train_quadrant_enumeration.json"))
    q = {c["id"]: str(c["name"]) for c in ann["categories_1"]}
    t = {c["id"]: str(c["name"]) for c in ann["categories_2"]}
    info = {i["id"]: i for i in ann["images"]}
    per = defaultdict(list)
    for a in ann["annotations"]:
        per[a["image_id"]].append((q[a["category_id_1"]] + t[a["category_id_2"]], a["bbox"]))

    # --- QC every image FIRST; split only the clean ones -------------------------
    report, clean_ids = [], []
    for img_id in sorted(per):
        codes = [c for c, _ in per[img_id]]
        status, reason = qc_image(codes)
        report.append({"file": info[img_id]["file_name"], "n_boxes": len(codes),
                       "status": status, "reason": reason})
        if status == "ok":
            clean_ids.append(img_id)

    rng = random.Random(args.seed)
    rng.shuffle(clean_ids)
    val_ids = set(clean_ids[: int(len(clean_ids) * args.val_frac)])

    # rebuild the split from scratch so a re-run never leaves stale (e.g. previously
    # kept) images behind; only touches images/ + labels/, never runs/ (weights).
    import shutil
    for split in ("train", "val"):
        shutil.rmtree(f"{args.out}/images/{split}", ignore_errors=True)
        shutil.rmtree(f"{args.out}/labels/{split}", ignore_errors=True)
        os.makedirs(f"{args.out}/images/{split}", exist_ok=True)
        os.makedirs(f"{args.out}/labels/{split}", exist_ok=True)

    for img_id in clean_ids:
        fn = info[img_id]["file_name"]
        split = "val" if img_id in val_ids else "train"
        raw = z.read(f"{ENUM}/xrays/{fn}")
        W, H = Image.open(io.BytesIO(raw)).size
        with open(f"{args.out}/images/{split}/{fn}", "wb") as f:
            f.write(raw)
        lines = []
        for code, (x, y, w, h) in per[img_id]:
            cls = 0 if args.single_class else FDI2CLS[code]
            lines.append(f"{cls} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                         f"{w / W:.6f} {h / H:.6f}")
        stem = os.path.splitext(fn)[0]
        with open(f"{args.out}/labels/{split}/{stem}.txt", "w") as f:
            f.write("\n".join(lines))

    # --- write the QC record (the blocklist any other process can read) ----------
    import csv
    with open(f"{args.out}/qc_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "n_boxes", "status", "reason"])
        w.writeheader(); w.writerows(report)
    excluded = [r for r in report if r["status"] == "excluded"]
    with open(f"{args.out}/excluded.json", "w") as f:
        json.dump({"excluded_files": [r["file"] for r in excluded],
                   "reasons": {r["file"]: r["reason"] for r in excluded}}, f, indent=1)

    with open(f"{args.out}/dentex.yaml", "w") as f:
        f.write(f"path: {args.out}\ntrain: images/train\nval: images/val\n")
        if args.single_class:
            f.write("nc: 1\nnames: ['tooth']\n")
        else:
            f.write(f"nc: {len(FDI)}\nnames: {FDI}\n")

    # --- VERIFY that the QC gate actually held ------------------------------------
    with open(f"{args.out}/fdi_codes.json", "w") as f:
        json.dump({info[i]["file_name"]: [c for c, _ in per[i]] for i in clean_ids}, f)
    bad = verify_build(args.out, args.single_class)
    assert not bad, f"VERIFY FAILED: {bad[:3]} — bad labels leaked through"

    reasons = Counter(r["reason"].split(":")[0] for r in excluded)
    print(f"kept {len(clean_ids)} clean images "
          f"({len(clean_ids) - len(val_ids)} train / {len(val_ids)} val)")
    print(f"EXCLUDED {len(excluded)} images -> {dict(reasons)}  (see qc_report.csv, excluded.json)")
    print("VERIFY OK: " + ("box counts match QC, all class 0" if args.single_class
                          else "no training/val label has a duplicate tooth."))
    print(f"wrote {args.out}/dentex.yaml")


if __name__ == "__main__":
    main()
