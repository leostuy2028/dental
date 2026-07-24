# FDI tooth detector

A YOLO detector that finds and FDI-numbers every tooth on a panoramic X-ray. Its job
in this project is **counting/enumeration** — the models we audit can't count teeth,
and the benchmark's counts are dentist-confirmed accurate, so an accurate detector can
answer the count questions directly.

## The workflow (no copy-paste)

The **repo is the source of truth**; Colab is a thin runner that pulls it. You never
paste code between the chat/editor and Colab.

1. Edit a `.py` here (or locally, or on GitHub's web editor).
2. Commit + push.
3. In Colab, the notebook's first cell does `git pull`, so it runs the latest.
4. **Dataset + weights live on Google Drive** (passed as `--out` / `--project` paths),
   so a dead Colab session never loses them.

Open the notebook straight from GitHub (bookmark this one URL):

```
https://colab.research.google.com/github/leostuy2028/dental/blob/master/detector/train.ipynb
```

Runtime -> Change runtime type -> **T4 GPU**, then run the cells top to bottom.

## The pipeline

| Stage | Script | Where | Notes |
|------|--------|-------|-------|
| 0 data | `prepare_data.py` | Colab (once) | streams DENTEX set (b), cleans the 3% noise, converts to YOLO, splits, writes `dentex.yaml` to **Drive** |
| 1 pretrain *(optional)* | `pretrain_ssl.py` | Colab GPU | SimCLR / InfoNCE on unlabeled X-rays -> a ResNet backbone |
| 2 train | `train.py` | Colab GPU | YOLO fine-tune (~1 hr on T4); curves + confusion matrix + val previews saved to Drive |
| 3 validate | `validate.py` | Colab | mAP **plus** count + FDI accuracy and the worst errors — this is the gate |
| 4 infer | `infer_mmoral.py` | **your machine (CPU)** | runs on the 100 MMOral images, checks the count against dentist-confirmed references |

## Data lives on Drive

`prepare_data.py --out /content/drive/MyDrive/dentex_yolo` writes:
```
dentex_yolo/
  images/{train,val}/*.png
  labels/{train,val}/*.txt      # class x_center y_center w h  (normalized)
  dentex.yaml                   # 32 FDI classes (11..18, 21..28, 31..38, 41..48)
  runs/<name>/weights/best.pt   # trained weights (Stage 2)
```
The raw 10.9 GB DENTEX zip is **never** persisted — set (b) is streamed out of it.

## Watching progress / errors

Ultralytics prints per-epoch loss + mAP live, and writes to `runs/<name>/`:
- `results.png` — loss and metric curves,
- `confusion_matrix.png` — which FDI codes get confused (expect wisdom teeth 18/28/38/48 to be weakest, they're often missing),
- `val_batch*_pred.jpg` — predicted boxes on validation images.

`validate.py` adds the count-oriented view: exact-count match rate, mean |count error|,
FDI recall, and the 10 worst images.

## The optional SSL caveat
`pretrain_ssl.py` trains a **ResNet** backbone (SSL weights load cleanly into ResNet).
Loading it into Ultralytics YOLO's custom CSPDarknet backbone is non-trivial. If the SSL
component is central to your writeup, use a ResNet-backbone detector; otherwise skip
Stage 1 and rely on the COCO-pretrained start in `train.py` (the reliable path).

## Recommended order
Stage 0 -> 2 -> 3 first (get a working detector, see if it counts on DENTEX val).
Add Stage 1 only if you want the SSL depth. Then Stage 4 (MMOral).
