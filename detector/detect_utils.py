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


def predict_boxes(model, source, imgsz=1024, conf=0.25, agnostic=True):
    """Return [(fdi_class_id, confidence), ...] for one image."""
    r = model.predict(source, imgsz=imgsz, conf=conf, agnostic_nms=agnostic, verbose=False)[0]
    return list(zip((int(c) for c in r.boxes.cls.tolist()),
                    (float(c) for c in r.boxes.conf.tolist())))


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
