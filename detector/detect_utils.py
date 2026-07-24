"""
Shared tooth-counting logic so validate.py and infer_mmoral.py count identically.

The raw detector over-counts: Ultralytics' default NMS is PER-CLASS, so when the model
is unsure whether a tooth is (say) 31 or 41 it emits both labels on the same physical
tooth and both survive -> counts can exceed 32, which is anatomically impossible. Two
corrections, both grounded in dental anatomy, fix this:

  1. class-agnostic NMS   — suppress overlapping boxes regardless of FDI class, so one
     physical tooth yields one box (the highest-confidence label wins);
  2. one box per FDI code — a mouth has at most one tooth per FDI position, so keep the
     highest-confidence box for each code. Caps the count at 32 and removes any
     same-code duplicates that agnostic NMS didn't merge.

These only remove impossible detections; they can never discard an anatomically valid
tooth. Report raw vs cleaned side by side so post-processing never hides a weak model.
"""


# class id (0..31) -> FDI code ("11".."48"), same order as detector/prepare_data.py
FDI = [f"{q}{t}" for q in (1, 2, 3, 4) for t in range(1, 9)]


def predict_boxes(model, source, imgsz=1024, conf=0.25, agnostic=True):
    """Return [(fdi_class_id, confidence), ...] for one image."""
    r = model.predict(source, imgsz=imgsz, conf=conf, agnostic_nms=agnostic, verbose=False)[0]
    return list(zip((int(c) for c in r.boxes.cls.tolist()),
                    (float(c) for c in r.boxes.conf.tolist())))


def detect_map(model, source, imgsz=1024, conf=0.25):
    """The labeled tooth map for one image: one box per FDI code (highest confidence),
    agnostic NMS. Returns [{'fdi': '26', 'conf': 0.9, 'box': [x1,y1,x2,y2]}, ...] sorted
    by FDI. Boxes are in original-image pixels. This is the full detector output used to
    ground a VLM (which tooth is where), not just the count."""
    r = model.predict(source, imgsz=imgsz, conf=conf, agnostic_nms=True, verbose=False)[0]
    best = {}
    for c, cf, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        c = int(c)
        if c not in best or cf > best[c][0]:
            best[c] = (float(cf), [round(v) for v in xy])
    return [{"fdi": FDI[c], "conf": round(cf, 3), "box": box}
            for c, (cf, box) in sorted(best.items())]


def dedup_fdi(boxes):
    """Keep the highest-confidence box per FDI code. boxes = [(cls, conf), ...] -> {cls: conf}."""
    best = {}
    for c, cf in boxes:
        if c not in best or cf > best[c]:
            best[c] = cf
    return best


def count_teeth(model, source, imgsz=1024, conf=0.25):
    """The DEPLOYED tooth count: agnostic NMS, count distinct physical detections.
    On DENTEX val this beats the FDI-code count (51% vs 39% exact, 0.70 vs 1.05 mean
    error) because it doesn't collapse two real teeth that share a confused code."""
    return len(predict_boxes(model, source, imgsz, conf, agnostic=True))


def tooth_codes(model, source, imgsz=1024, conf=0.25):
    """Enumeration view: agnostic NMS + one box per FDI code. Returns a set of class ids.
    Use for 'which teeth', not for counting (it under-counts on code confusion)."""
    return set(dedup_fdi(predict_boxes(model, source, imgsz, conf, agnostic=True)))
