"""
Build labeled visual few-shot exemplars from the DENTEX validation set (CC-BY,
ibrahimhamamci/DENTEX) for the P13 visual-exemplar test. Draws each finding's real
bounding box + FDI-code/diagnosis label onto the image (so the exemplar points
unambiguously at the tooth), downscales, and writes a captions manifest. Captions
come ONLY from DENTEX's real annotations — nothing is invented.

Run:   python dataio/make_dentex_exemplars.py <path-to-dentex-scratch-dir>
Writes:reference/dentex_exemplars/<name>.jpg  + manifest.json
"""
import os
import sys
import json
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

CHOSEN = ["val_9", "val_20", "val_39", "val_34", "val_19", "val_37"]  # cover impacted/caries/deep-caries/periapical
MAX_PX = 1024


def main():
    src = sys.argv[1]
    ann = json.load(open(os.path.join(src, "validation_triple.json")))
    q = {c["id"]: c["name"] for c in ann["categories_1"]}
    t = {c["id"]: c["name"] for c in ann["categories_2"]}
    dx = {c["id"]: c["name"] for c in ann["categories_3"]}
    fname = {i["id"]: i["file_name"] for i in ann["images"]}
    per = defaultdict(list)
    for a in ann["annotations"]:
        fdi = f"{q[a['category_id_1']]}{t[a['category_id_2']]}"
        per[fname[a["image_id"]]].append((fdi, dx[a["category_id_3"]], a["bbox"]))

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo, "reference", "dentex_exemplars")
    os.makedirs(out, exist_ok=True)
    xrays = os.path.join(src, "valimg", "validation_data", "quadrant_enumeration_disease", "xrays")

    manifest = []
    for stem in CHOSEN:
        fn = stem + ".png"
        findings = per[fn]
        img = Image.open(os.path.join(xrays, fn)).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except Exception:
            font = ImageFont.load_default()
        for fdi, diag, (x, y, w, h) in findings:
            draw.rectangle([x, y, x + w, y + h], outline=(255, 40, 40), width=6)
            label = f"#{fdi} {diag}"
            draw.rectangle([x, max(0, y - 46), x + 12 + len(label) * 20, y], fill=(255, 40, 40))
            draw.text((x + 4, max(0, y - 44)), label, fill=(255, 255, 255), font=font)
        if max(img.size) > MAX_PX:
            s = MAX_PX / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)))
        img.save(os.path.join(out, stem + ".jpg"), quality=90)
        cap = "Labeled findings (red boxes, FDI numbering): " + \
              "; ".join(f"#{fdi} {diag.lower()}" for fdi, diag, _ in findings) + "."
        manifest.append({"image": f"dentex_exemplars/{stem}.jpg", "caption": cap})
        print(f"{stem}: {cap}")

    json.dump({"_source": "DENTEX (ibrahimhamamci/DENTEX), CC-BY; validation set; labels verbatim from annotations",
               "exemplars": manifest}, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"\nwrote {len(manifest)} exemplars + manifest.json to reference/dentex_exemplars/")


if __name__ == "__main__":
    main()
