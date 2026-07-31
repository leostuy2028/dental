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
| 0 data | `prepare_data.py` | Colab (once) | streams DENTEX set (b), **excludes** the ~2.5% count-erroneous images, converts to YOLO, splits, writes `dentex.yaml` to **Drive** |
| 1 pretrain *(optional)* | `pretrain_ssl.py` | Colab GPU | SimCLR / InfoNCE on unlabeled X-rays -> a ResNet backbone |
| 2 train | `train.py` | Colab GPU | YOLO fine-tune (~1 hr on T4); curves + confusion matrix + val previews saved to Drive |
| 3 validate | `validate.py` | Colab | mAP **plus** count + FDI accuracy and the worst errors — this is the gate |
| 4 infer | `infer_mmoral.py` | **your machine (CPU)** | runs on the 100 MMOral images, checks the count against dentist-confirmed references |

**The trained weights are committed** at the repo root as `best.pt` (6 MB, YOLOv8-nano).
That is the exact checkpoint behind every committed detector number (DENTEX val mAP50
0.935; MMOral 51% exact / 86% within one tooth) and behind `reference/mmoral_map.json`,
so Stage 4 reproduces without retraining: `python detector/infer_mmoral.py --weights best.pt`.
Retraining produces a *different* model, so replace `best.pt` only alongside re-running the
downstream results.

---

## Two detectors live here. Know which one you want.

| | `best.pt` (v1) | `best2.pt` (v2) |
|---|---|---|
| model | YOLOv8-**nano**, 3M params | YOLOv8-**small**, 11M |
| classes | 32, one per FDI code | **1** (`tooth`) |
| trained at | imgsz 1024, mosaic 0.4 | imgsz **1536**, mosaic **0** |
| DENTEX val | mAP50 0.935 (32-class) | mAP50 0.991, recall 0.985 (1-class) |
| **inference settings** | `imgsz=1280, conf=0.25, iou=0.7` | **`imgsz=1536, conf=0.20, iou=0.45`** |
| MMOral exact count | 69.8% (0.66 err) | **81.4%** (0.36 err) |
| held out, 800 splits | 64.1% | **76.1%** |
| 28 counting MCQs | 67.9% | **75.0%** |

`best.pt` is kept because every number currently in the paper traces to it. **`best2.pt` is
the better counter** and is what new work should use.

The mAP figures are **not** comparable: v1's is over 32 classes, so a correctly-placed box
with the wrong FDI code counted against it. Compare the count rows, not the mAP row.

### The inference settings are not optional

`iou=0.45` is load-bearing for v2, not a nicety. Going single-class removed v1's "one box per
FDI code" rule, which had been suppressing duplicate boxes as a side effect. At ultralytics'
default `iou=0.7` v2 counts *worse* than v1 (64% vs 70%) because two boxes land on one tooth
and nothing collapses them. At 0.45 it reaches 81%. At 0.30 it falls to 42%, because it starts
merging genuinely adjacent teeth. There is a real optimum in the middle — see
`results/detector/inference_sweep_v2.csv` for the whole surface.

Same story for numbering: a duplicate box inflates its quadrant's index and shifts every code
behind it, so `eval_numbering.py` must be passed the same `--iou`.

### Reproduce the v2 numbers (CPU, no GPU, no retraining)

```bash
# counting: the grid, a held-out check, and the benchmark's 28 counting MCQs
python detector/tune_inference.py --weights best2.pt --res 1280,1536,1792 --iou 0.3,0.45,0.6,0.7 \
    --raw results/detector/inference_raw_confs_v2.json --out results/detector/inference_sweep_v2.csv

# numbering: wisdom teeth, scored against the benchmark's own reference answers
python detector/eval_numbering.py --weights best2.pt --imgsz 1536 --conf 0.20 --iou 0.45 \
    --out results/detector/numbering_wisdom_v2.csv

# the data-hygiene gate, both class modes
python detector/test_prepare_qc.py
```

Raw confidences are cached, so re-runs take seconds; pass `--refresh` to redo inference.
Both scripts default to `best.pt`, so **always pass `--weights best2.pt`** for v2.

### What the pipeline actually outputs

The network emits only boxes — one class, no tooth numbers:

```
box [1166, 268, 1242, 611]  conf 0.74  class 0 (tooth)
```

`number_teeth.py` then adds the FDI code from geometry: fit each arch, find the midline,
quadrant = arch x side, number outward from the midline. Final per-tooth record:

```
FDI 11  box [1166, 268, 1242, 611]  conf 0.74
```

From which the two answerable question types fall out: **how many teeth** = number of boxes;
**which wisdom teeth** = the codes ending in 8.

### Numbering is NOT solved — do not quote it as working

Wisdom-tooth accuracy is **39%** (per-tooth recall 0.69), against 37% for v1. Better detection
did not fix it, which rules out detection quality as the cause.

The cause is the method. Numbering counts ordinally outward from the midline, so *any* upstream
error — one missed tooth, or a midline off by a few percent — shifts every code behind it and
the third molar lands on 7 instead of 8. Worked example, `016640.jpg`: 30 teeth detected
(reference 31), all 30 numbered, midline at x=1264 against an image centre of 1236. Quadrants 1
and 4 reach index 8; quadrants 2 and 3 stop at 7. So the pipeline reports wisdom teeth
`[18, 48]` while the reference says all four are present. Nothing is missing from the picture —
the counting scheme simply ran short in two quadrants.

The fix is to assign a code from **normalised position along the arch**, calibrated on DENTEX's
real expert FDI labels, which is robust to a missing neighbour by construction. Not built yet.

## Data lives on Drive

`prepare_data.py --out /content/drive/MyDrive/dentex_yolo` writes:
```
dentex_yolo/
  images/{train,val}/*.png
  labels/{train,val}/*.txt      # class x_center y_center w h  (normalized)
  dentex.yaml                   # 32 FDI classes (11..18, 21..28, 31..38, 41..48)
  qc_report.csv                 # every image: n_boxes, status (ok/excluded), reason
  excluded.json                 # the blocklist of dropped files + reasons
  runs/<name>/weights/best.pt   # trained weights (Stage 2)
```
The raw 10.9 GB DENTEX zip is **never** persisted — set (b) is streamed out of it.

## Data hygiene — keeping count-erroneous images out

Some DENTEX annotations box a tooth twice, so the image's tooth count is wrong (the
worst has 40 boxes). Those must never reach the detector — training it on a bad count
would defeat the one thing it exists to do. So Stage 0 runs a single QC gate,
`qc_image()`, over the **raw** annotations *before* the train/val split:

- an image is **excluded** if any FDI code repeats (duplicate tooth), if it has >32
  boxes, or if it has fewer than 4 (a broken, near-empty annotation);
- genuine sparse/edentulous mouths are **kept** (the counter must handle them);
- exclusions are written to `qc_report.csv` + `excluded.json` — one auditable blocklist
  that is the *only* thing feeding `images/{train,val}`, so `train.py` and `validate.py`
  both read the already-filtered split and nothing else can slip past;
- after writing, a verify pass re-reads every emitted label and **asserts** no file
  contains a duplicate class — the build fails loudly if a bad label ever leaks through.

On the current DENTEX set that drops **16 of 634** images (all duplicate-tooth), leaving
618 clean. `infer_mmoral.py` runs on MMOral, not DENTEX, so it never touches these.

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
