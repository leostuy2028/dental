# Tests for detector/detect_utils.py
# Shared tooth-counting logic used by validate.py and infer_mmoral.py. Post-processes
# raw YOLO output two ways: class-agnostic NMS (merge overlapping boxes across classes)
# and "one box per FDI code" (keep only the highest-confidence box per code). No real
# model is used here: a tiny fake stands in for the ultralytics YOLO model.

from detect_utils import FDI, count_teeth, dedup_fdi, detect_map, predict_boxes, tooth_codes


class FakeTensor:
    """Stands in for a torch tensor: only .tolist() is ever called on it here."""
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = FakeTensor(cls)
        self.conf = FakeTensor(conf)
        self.xyxy = FakeTensor(xyxy)


class FakeResult:
    def __init__(self, cls, conf, xyxy):
        self.boxes = FakeBoxes(cls, conf, xyxy)


class FakeModel:
    """Records the predict() call it received and always returns the same boxes."""
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = cls, conf, xyxy
        self.calls = []

    def predict(self, source, imgsz, conf, agnostic_nms, verbose):
        self.calls.append(dict(source=source, imgsz=imgsz, conf=conf,
                               agnostic_nms=agnostic_nms, verbose=verbose))
        return [FakeResult(self.cls, self.conf, self.xyxy)]


def test_fdi_table_has_32_codes_in_quadrant_order():
    assert len(FDI) == 32
    assert FDI[0] == "11"
    assert FDI[8] == "21"
    assert FDI[-1] == "48"


# ---------- predict_boxes ----------

def test_predict_boxes_pairs_class_with_confidence():
    model = FakeModel(cls=[0.0, 5.0], conf=[0.9, 0.4], xyxy=[[0, 0, 1, 1], [2, 2, 3, 3]])
    out = predict_boxes(model, "img.png")
    assert out == [(0, 0.9), (5, 0.4)]


def test_predict_boxes_passes_agnostic_flag_through_to_the_model():
    model = FakeModel(cls=[], conf=[], xyxy=[])
    predict_boxes(model, "img.png", imgsz=640, conf=0.5, agnostic=False)
    assert model.calls[0]["agnostic_nms"] is False
    assert model.calls[0]["imgsz"] == 640
    assert model.calls[0]["conf"] == 0.5


def test_predict_boxes_defaults_to_agnostic_true():
    model = FakeModel(cls=[], conf=[], xyxy=[])
    predict_boxes(model, "img.png")
    assert model.calls[0]["agnostic_nms"] is True


# ---------- dedup_fdi ----------

def test_dedup_fdi_keeps_the_highest_confidence_per_class():
    assert dedup_fdi([(0, 0.5), (0, 0.9), (1, 0.3)]) == {0: 0.9, 1: 0.3}


def test_dedup_fdi_empty():
    assert dedup_fdi([]) == {}


# ---------- detect_map ----------

def test_detect_map_keeps_best_box_per_fdi_and_sorts_by_code():
    # class 5 ("16") appears twice; the higher-confidence box should win
    model = FakeModel(cls=[5, 0, 5], conf=[0.6, 0.95, 0.8],
                      xyxy=[[10.4, 10.6, 20.0, 20.0], [0, 0, 5, 5], [1, 1, 2, 2]])
    out = detect_map(model, "img.png")
    assert [t["fdi"] for t in out] == ["11", "16"]
    picked = next(t for t in out if t["fdi"] == "16")
    assert picked["conf"] == 0.8
    assert picked["box"] == [1, 1, 2, 2]


def test_detect_map_rounds_box_coordinates_and_confidence():
    model = FakeModel(cls=[0], conf=[0.12345], xyxy=[[1.4, 2.6, 3.5, 4.5]])
    out = detect_map(model, "img.png")
    assert out[0]["box"] == [1, 3, 4, 4]  # Python round(): .5 rounds to even
    assert out[0]["conf"] == 0.123


def test_detect_map_always_uses_agnostic_nms():
    model = FakeModel(cls=[0], conf=[0.5], xyxy=[[0, 0, 1, 1]])
    detect_map(model, "img.png")
    assert model.calls[0]["agnostic_nms"] is True


def test_detect_map_empty_result():
    model = FakeModel(cls=[], conf=[], xyxy=[])
    assert detect_map(model, "img.png") == []


# ---------- count_teeth / tooth_codes ----------

def test_count_teeth_counts_every_surviving_detection():
    model = FakeModel(cls=[0, 1, 2], conf=[0.9, 0.8, 0.7],
                      xyxy=[[0, 0, 1, 1]] * 3)
    assert count_teeth(model, "img.png") == 3


def test_count_teeth_empty():
    model = FakeModel(cls=[], conf=[], xyxy=[])
    assert count_teeth(model, "img.png") == 0


def test_tooth_codes_deduplicates_by_class_id():
    model = FakeModel(cls=[0, 0, 1], conf=[0.5, 0.9, 0.3], xyxy=[[0, 0, 1, 1]] * 3)
    assert tooth_codes(model, "img.png") == {0, 1}
