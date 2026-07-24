"""
Stage 3 — the GATE. Standard detection metrics PLUS the count/FDI metrics that
actually matter for our use, and a list of the worst validation errors.

mAP tells you the boxes are well-placed; what we care about is:
  * COUNT accuracy  — does len(detections) equal the true tooth count?
  * FDI accuracy    — of the true teeth, how many did we detect with the right number?
If the detector can't count on DENTEX's own val split, stop before touching MMOral.

Run:
  python detector/validate.py --weights .../weights/best.pt --data .../dentex.yaml
"""
import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    ap.add_argument("--report", default=None, help="optional path to write a CSV of per-image counts")
    args = ap.parse_args()

    import yaml as pyyaml
    from ultralytics import YOLO
    from detect_utils import predict_boxes, dedup_fdi
    model = YOLO(args.weights)

    # 1) standard detection metrics (mAP) — writes plots next to the weights
    m = model.val(data=args.data, imgsz=args.imgsz, plots=True, verbose=False)
    print(f"DETECTION:  mAP50 = {m.box.map50:.3f}   mAP50-95 = {m.box.map:.3f}")

    # 2) count + FDI accuracy on the val images
    d = pyyaml.safe_load(open(args.data))
    root, valrel = d["path"], d["val"]
    val_split = os.path.basename(valrel.rstrip("/"))     # 'val'
    imgs = sorted(glob.glob(os.path.join(root, valrel, "*")))

    raw_exact = phys_exact = enum_exact = 0
    raw_err, phys_err, enum_err = [], [], []
    fdi_hit = fdi_tot = 0
    rows = []
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lab = os.path.join(root, "labels", val_split, stem + ".txt")
        true = [int(ln.split()[0]) for ln in open(lab)] if os.path.exists(lab) else []
        true_set, true_n = set(true), len(true)

        raw_n = len(predict_boxes(model, ip, args.imgsz, args.conf, agnostic=False))  # default NMS
        ag = predict_boxes(model, ip, args.imgsz, args.conf, agnostic=True)           # agnostic NMS
        phys_n = len(ag)                          # distinct physical teeth
        enum = set(dedup_fdi(ag))                 # one box per FDI code
        enum_n = len(enum)

        raw_err.append(abs(raw_n - true_n));   raw_exact += (raw_n == true_n)
        phys_err.append(abs(phys_n - true_n)); phys_exact += (phys_n == true_n)
        enum_err.append(abs(enum_n - true_n)); enum_exact += (enum_n == true_n)
        fdi_tot += true_n
        fdi_hit += len(true_set & enum)           # true teeth whose FDI we also predicted
        rows.append((stem, true_n, raw_n, phys_n, enum_n, phys_n - true_n))

    n = max(len(imgs), 1)
    print(f"COUNT raw      (per-class NMS):    exact {raw_exact:2d}/{n} ({raw_exact / n * 100:2.0f}%)   "
          f"mean |err| {sum(raw_err) / n:.2f} teeth")
    print(f"COUNT physical (agnostic NMS):     exact {phys_exact:2d}/{n} ({phys_exact / n * 100:2.0f}%)   "
          f"mean |err| {sum(phys_err) / n:.2f} teeth")
    print(f"COUNT enum     (agnostic + 1/FDI): exact {enum_exact:2d}/{n} ({enum_exact / n * 100:2.0f}%)   "
          f"mean |err| {sum(enum_err) / n:.2f} teeth")
    print(f"FDI:           {fdi_hit}/{fdi_tot} true teeth detected with the right number "
          f"({fdi_hit / max(fdi_tot, 1) * 100:.0f}%)")

    print("\nWorst PHYSICAL count errors (true -> predicted):")
    for stem, tn, rn, pn, en, e in sorted(rows, key=lambda x: -abs(x[5]))[:10]:
        print(f"  {stem}:  true {tn:2d}  physical {pn:2d}  ({e:+d})   [raw {rn}, enum {en}]")

    if args.report:
        import csv
        with open(args.report, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "true_count", "raw_count", "physical_count", "enum_count", "physical_error"])
            w.writerows(rows)
        print(f"\nper-image counts -> {args.report}")


if __name__ == "__main__":
    main()
