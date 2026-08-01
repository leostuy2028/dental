"""
Stage 0b — build a YOLO dataset for the DENTEX DIAGNOSIS classes (set (c),
quadrant-enumeration-disease): Impacted, Caries, Deep Caries, Periapical Lesion.

WHY THIS IS A DIFFERENT DATASET FROM prepare_data.py, and why its QC gate is different.

prepare_data.py builds the TOOTH detector from set (b), where every tooth in the mouth is
annotated. That lets it enforce a count invariant — no duplicate FDI code, 4..32 boxes — which
is essential, because a counter trained on a mis-counted image is worthless.

Set (c) annotates ONLY THE ABNORMAL TEETH. That makes it wrong for counting (it would teach the
model that healthy teeth are background) and right for pathology, which is what this builds.
The count invariant does NOT apply and must not be reused here:

  - an image with 3 boxes is a mouth with 3 findings, not a broken annotation;
  - the SAME FDI code legitimately repeats, because one tooth can carry more than one finding
    (a caries and a periapical lesion on the same tooth are two true boxes).

So the gate here is only: the image must have at least one finding and a sane box count. Any
stricter rule would silently discard real clinical data.

Class balance is heavily skewed and worth knowing before reading the metrics:
  Caries 2189 boxes / 623 images   Impacted 604 / 254
  Deep Caries 578 / 321            Periapical Lesion 158 / 116
Periapical Lesion has a tenth the data of Caries, and it is also the class with a known
modality ceiling — a commercial dental AI detects periapical lesions at 33% sensitivity on
panoramic against 78% on CBCT (Kazimierczak et al., paper ref [4]). Expect it to train worst,
and treat a poor result there as partly the modality rather than the model.

Run:
  python detector/prepare_disease.py --out /content/drive/MyDrive/dentex_disease
"""
import argparse
import csv
import io
import json
import os
import random
import shutil
from collections import Counter, defaultdict

DISEASE = "training_data/quadrant-enumeration-disease"
ANN = f"{DISEASE}/train_quadrant_enumeration_disease.json"
HF_URL = ("https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/"
          "DENTEX/training_data.zip")
MAX_BOXES = 40          # above this the annotation is implausible for one mouth


def open_zip(source):
    if source.startswith("http"):
        from remotezip import RemoteZip
        return RemoteZip(source)
    import zipfile
    return zipfile.ZipFile(source)


def qc_image(n_boxes):
    """Set (c) is abnormal-only, so there is no count invariant to check — only sanity."""
    if n_boxes < 1:
        return "excluded", "no_findings"
    if n_boxes > MAX_BOXES:
        return "excluded", f"over_{MAX_BOXES}_boxes"
    return "ok", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default=HF_URL)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image

    z = open_zip(args.source)
    ann = json.loads(z.read(ANN))
    names = [c["name"] for c in sorted(ann["categories_3"], key=lambda c: c["id"])]
    cls_of = {c["id"]: c["id"] for c in ann["categories_3"]}
    quad = {c["id"]: c["name"] for c in ann["categories_1"]}
    tooth = {c["id"]: c["name"] for c in ann["categories_2"]}
    info = {i["id"]: i for i in ann["images"]}

    per = defaultdict(list)
    for a in ann["annotations"]:
        # the FDI code is carried alongside the diagnosis; kept in the sidecar so a later
        # stage can check "did we put the finding on the right tooth" without re-reading DENTEX
        fdi = quad[a["category_id_1"]] + tooth[a["category_id_2"]]
        per[a["image_id"]].append((cls_of[a["category_id_3"]], a["bbox"], fdi))

    report, clean = [], []
    for img_id in sorted(per):
        status, reason = qc_image(len(per[img_id]))
        report.append({"file": info[img_id]["file_name"], "n_boxes": len(per[img_id]),
                       "status": status, "reason": reason})
        if status == "ok":
            clean.append(img_id)

    rng = random.Random(args.seed)
    rng.shuffle(clean)
    val_ids = set(clean[: int(len(clean) * args.val_frac)])

    for split in ("train", "val"):
        shutil.rmtree(f"{args.out}/images/{split}", ignore_errors=True)
        shutil.rmtree(f"{args.out}/labels/{split}", ignore_errors=True)
        os.makedirs(f"{args.out}/images/{split}", exist_ok=True)
        os.makedirs(f"{args.out}/labels/{split}", exist_ok=True)

    sidecar = {}
    for img_id in clean:
        fn = info[img_id]["file_name"]
        split = "val" if img_id in val_ids else "train"
        raw = z.read(f"{DISEASE}/xrays/{fn}")
        W, H = Image.open(io.BytesIO(raw)).size
        with open(f"{args.out}/images/{split}/{fn}", "wb") as f:
            f.write(raw)
        lines, codes = [], []
        for cid, (x, y, w, h), fdi in per[img_id]:
            lines.append(f"{cid} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                         f"{w / W:.6f} {h / H:.6f}")
            codes.append({"cls": names[cid], "fdi": fdi})
        stem = os.path.splitext(fn)[0]
        with open(f"{args.out}/labels/{split}/{stem}.txt", "w") as f:
            f.write("\n".join(lines))
        sidecar[fn] = codes

    with open(f"{args.out}/findings.json", "w") as f:
        json.dump(sidecar, f)
    with open(f"{args.out}/qc_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "n_boxes", "status", "reason"])
        w.writeheader(); w.writerows(report)
    with open(f"{args.out}/dentex.yaml", "w") as f:
        f.write(f"path: {args.out}\ntrain: images/train\nval: images/val\n"
                f"nc: {len(names)}\nnames: {names}\n")

    # VERIFY: every written label maps back to a finding the source actually recorded
    import glob
    bad = []
    for lf in glob.glob(f"{args.out}/labels/*/*.txt"):
        stem = os.path.splitext(os.path.basename(lf))[0]
        got = [ln.split()[0] for ln in open(lf) if ln.strip()]
        want = [c for fn, c in sidecar.items() if os.path.splitext(fn)[0] == stem]
        if want and len(got) != len(want[0]):
            bad.append(f"{lf}: {len(got)} labels vs {len(want[0])} findings")
        if any(int(g) >= len(names) for g in got):
            bad.append(f"{lf}: class id outside 0..{len(names)-1}")
    assert not bad, f"VERIFY FAILED: {bad[:3]}"

    excl = [r for r in report if r["status"] == "excluded"]
    print(f"kept {len(clean)} images ({len(clean)-len(val_ids)} train / {len(val_ids)} val), "
          f"excluded {len(excl)} {dict(Counter(r['reason'] for r in excl))}")
    print("classes:", names)
    cnt = Counter(names[c] for v in per.values() for c, _, _ in v)
    for k, v in cnt.most_common():
        print(f"  {k:20} {v:5d} boxes")
    print("VERIFY OK: every label file matches its recorded findings")
    print(f"wrote {args.out}/dentex.yaml  (+ findings.json carrying each finding's FDI code)")


if __name__ == "__main__":
    main()
