"""
Build HisT/Jaw visual exemplars from the Zenodo panoramic-condition dataset
(CC-BY-4.0, zenodo.org/records/15487430) for the P13 v2 exemplar test. These cover
the dimensions DENTEX lacked: implants / crowns / fillings / root-canal (HisT) and
bone loss (Jaw) — the untaught half of the hard core.

For each target condition we pick the image with the most prominent (largest-area)
real box of that class and draw ONLY that condition's boxes, so the exemplar is a
focused "this is what X looks like". Labels come verbatim from the YOLO annotations
(no tooth numbers — this dataset doesn't annotate them, and we do not guess).

Run:   python dataio/make_zenodo_exemplars.py <path-to-zenodo-data-dir>
Writes:reference/zenodo_exemplars/<name>.jpg + manifest.json
"""
import os
import sys
import glob
import json
from PIL import Image, ImageDraw, ImageFont

# (class_id, friendly caption, output stem, how many images)
# For the HisT classes (implant/crown/filling/rct) the largest-box heuristic yields clean,
# unambiguous exemplars. For bone loss (class 5) the largest box selects whole-arch EDENTULOUS
# ridge atrophy — a different entity from the peri-dental bone loss the Jaw dimension asks about
# — so those two are PINNED to visually-QA'd dentate cases (see PIN below).
TARGETS = [
    (0, "a dental implant", "implant", 1),
    (1, "a prosthetic crown / restoration", "crown", 1),
    (2, "a dental filling (obturation)", "filling", 1),
    (3, "root canal treatment (endodontic filling)", "rct", 1),
    (5, "periodontal bone loss (bone resorption)", "boneloss", 2),
]
# class_id -> [label-file stems] to force specific images (bypasses largest-box pick).
# 57 (16 boxes) & 591 (8 boxes): dentate arches with interproximal bone-crest loss around
# teeth — the true periodontal pattern; verified by eye. Avoids edentulous-ridge false teach.
PIN = {5: ["57", "591"]}
MAX_PX = 1024


def main():
    root = sys.argv[1]
    lbls = glob.glob(os.path.join(root, "train", "labels", "*.txt"))
    # index: class_id -> [(area, label_path)]
    by_cls = {}
    for lp in lbls:
        for line in open(lp):
            p = line.split()
            if len(p) == 5:
                cid = int(p[0]); area = float(p[3]) * float(p[4])
                by_cls.setdefault(cid, []).append((area, lp))

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo, "reference", "zenodo_exemplars")
    os.makedirs(out, exist_ok=True)
    manifest = []
    used = set()
    for cid, caption, stem, n in TARGETS:
        if cid in PIN:
            # forced picks: resolve each pinned stem to its label path, in order
            lookup = {os.path.splitext(os.path.basename(p))[0]: (a, p)
                      for a, p in by_cls.get(cid, [])}
            cands = [lookup[s] for s in PIN[cid] if s in lookup]
        else:
            cands = sorted(by_cls.get(cid, []), reverse=True)  # largest box first
        picked = 0
        for _, lp in cands:
            if lp in used:
                continue
            img_stem = os.path.splitext(os.path.basename(lp))[0]
            imgs = glob.glob(os.path.join(root, "train", "images", img_stem + ".*"))
            if not imgs:
                continue
            used.add(lp); picked += 1
            img = Image.open(imgs[0]).convert("RGB")
            W, H = img.size
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 38)
            except Exception:
                font = ImageFont.load_default()
            for line in open(lp):
                p = line.split()
                if len(p) == 5 and int(p[0]) == cid:  # draw ONLY this condition
                    xc, yc, w, h = [float(v) for v in p[1:]]
                    x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
                    x2, y2 = (xc + w / 2) * W, (yc + h / 2) * H
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=6)
            lab = caption.split("(")[0].strip()
            draw.rectangle([8, 8, 12 + len(lab) * 19, 54], fill=(255, 40, 40))
            draw.text((12, 10), lab, fill=(255, 255, 255), font=font)
            if max(img.size) > MAX_PX:
                s = MAX_PX / max(img.size)
                img = img.resize((int(img.width * s), int(img.height * s)))
            name = f"{stem}{picked if n > 1 else ''}"
            img.save(os.path.join(out, name + ".jpg"), quality=90)
            manifest.append({"image": f"zenodo_exemplars/{name}.jpg",
                             "caption": f"This panoramic X-ray shows {caption} (red box)."})
            print(f"{name}.jpg -> {caption}")
            if picked >= n:
                break

    json.dump({"_source": "Zenodo 15487430 panoramic condition dataset, CC-BY-4.0; labels verbatim from YOLO annotations (no tooth numbers annotated)",
               "exemplars": manifest}, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"\nwrote {len(manifest)} exemplars + manifest.json to reference/zenodo_exemplars/")


if __name__ == "__main__":
    main()
