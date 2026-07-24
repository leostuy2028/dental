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
    model = YOLO(args.weights)

    # 1) standard detection metrics (mAP) — writes plots next to the weights
    m = model.val(data=args.data, imgsz=args.imgsz, plots=True, verbose=False)
    print(f"DETECTION:  mAP50 = {m.box.map50:.3f}   mAP50-95 = {m.box.map:.3f}")

    # 2) count + FDI accuracy on the val images
    d = pyyaml.safe_load(open(args.data))
    root, valrel = d["path"], d["val"]
    val_split = os.path.basename(valrel.rstrip("/"))     # 'val'
    imgs = sorted(glob.glob(os.path.join(root, valrel, "*")))

    exact = 0
    abs_err = []
    fdi_hit = fdi_tot = 0
    rows = []
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lab = os.path.join(root, "labels", val_split, stem + ".txt")
        true = [ln.split()[0] for ln in open(lab)] if os.path.exists(lab) else []
        true_set, true_n = set(true), len(true)

        r = model.predict(ip, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        pred = [str(int(c)) for c in r.boxes.cls.tolist()]
        pred_n = len(pred)

        abs_err.append(abs(pred_n - true_n))
        exact += (pred_n == true_n)
        fdi_tot += true_n
        fdi_hit += len(true_set & set(pred))      # true teeth whose FDI we also predicted
        rows.append((stem, true_n, pred_n, pred_n - true_n))

    n = max(len(imgs), 1)
    print(f"COUNT:      exact {exact}/{n} ({exact / n * 100:.0f}%)   "
          f"mean |error| = {sum(abs_err) / n:.2f} teeth")
    print(f"FDI:        {fdi_hit}/{fdi_tot} true teeth detected with the right number "
          f"({fdi_hit / max(fdi_tot, 1) * 100:.0f}%)")

    print("\nWorst count errors (true -> predicted):")
    for stem, tn, pn, e in sorted(rows, key=lambda x: -abs(x[3]))[:10]:
        print(f"  {stem}:  true {tn:2d}  pred {pn:2d}  ({e:+d})")

    if args.report:
        import csv
        with open(args.report, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "true_count", "pred_count", "error"])
            w.writerows(rows)
        print(f"\nper-image counts -> {args.report}")


if __name__ == "__main__":
    main()
