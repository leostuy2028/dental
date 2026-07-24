"""
Stage 2 — fine-tune a YOLO FDI tooth detector on the prepared DENTEX set (b).

Ultralytics prints per-epoch progress (box/cls/dfl loss, precision, recall, mAP50,
mAP50-95) and writes plots to <project>/<name>/: results.png (loss + metric curves),
confusion_matrix.png, and val_batch*_pred.jpg (predictions on val images). Point
--project at a Drive path so all of that survives the session.

Run (Colab, T4):
  python detector/train.py --data /content/drive/MyDrive/dentex_yolo/dentex.yaml \
      --project /content/drive/MyDrive/dentex_yolo/runs
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dentex.yaml (on Drive)")
    ap.add_argument("--model", default="yolov8n.pt", help="COCO-pretrained start, or an SSL backbone")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=1024, help="panoramics are wide; keep this large")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="dentex_yolov8n")
    ap.add_argument("--patience", type=int, default=30, help="early-stop patience")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its last.pt (Colab disconnected, etc.)")
    args = ap.parse_args()

    from ultralytics import YOLO
    if args.resume:
        ckpt = f"{args.project}/{args.name}/weights/last.pt"
        print(f"resuming from {ckpt} (other train args are read from the saved run)")
        model = YOLO(ckpt)
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            patience=args.patience,
            mosaic=0.4,          # ease off heavy augmentation for medical images
            cache=True,          # cache images so Drive I/O isn't the bottleneck after epoch 1
            plots=True,          # write results.png / confusion_matrix.png / val previews
            seed=0,
        )
    best = f"{args.project}/{args.name}/weights/best.pt"
    print(f"\nBest weights: {best}")
    print(f"Curves + confusion matrix + val-prediction images: {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
