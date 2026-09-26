# Tests for detector/infer_mmoral.py
# Stage 4: runs the detector on the MMOral panoramics and checks its tooth COUNT against
# the dentist-confirmed reference count parsed out of the benchmark's own answer text.
# No network, no real weights: a tiny local parquet file stands in for
# data/open_ended.parquet, and a fake model stands in for ultralytics.YOLO.

import base64
import io
import json

import pandas as pd
import ultralytics
from PIL import Image

from infer_mmoral import gt_count


# ---------- gt_count ----------

def test_gt_count_extracts_the_number_before_visualized():
    assert gt_count("28 teeth are visualized in the radiograph.") == 28


def test_gt_count_extracts_before_present():
    assert gt_count("26 teeth present") == 26


def test_gt_count_extracts_before_detected():
    assert gt_count("30 teeth detected") == 30


def test_gt_count_returns_none_when_no_count_sentence():
    assert gt_count("no clear number here") is None


def test_gt_count_handles_none():
    assert gt_count(None) is None


def test_gt_count_requires_teeth_immediately_after_the_number():
    # CURRENT BEHAVIOR (looks like a bug): an adjective between the number and "teeth"
    # ("permanent teeth") breaks the match, so a real count sentence like this is missed
    assert gt_count("5 permanent teeth are visualized") is None


def test_gt_count_ignores_a_visible_count_not_using_a_matched_verb():
    # "visible" is not one of visualized/present/detected, so it doesn't count
    assert gt_count("28 teeth are visible") is None


# ---------- main() end to end ----------

class FakeTensor:
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
    """Keyed by the decoded image's pixel size, so the fake doesn't need to see a
    filename (the real code only ever hands the model a decoded PIL image)."""
    def __init__(self, by_size):
        self.by_size = by_size
        self.calls = []

    def predict(self, source, imgsz, conf, agnostic_nms=True, iou=None, verbose=False):
        cls = self.by_size[source.size]
        self.calls.append(dict(imgsz=imgsz, conf=conf, agnostic_nms=agnostic_nms))
        return [FakeResult(cls, [0.9] * len(cls), [[0, 0, 5, 5]] * len(cls))]


def b64_png(size):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(200, 200, 200)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_open_ended(tmp_path):
    rows = [
        {"image_name": "img1.png", "image": b64_png((20, 20)),
         "answer": "28 teeth are visualized in the radiograph."},
        {"image_name": "img1.png", "image": b64_png((20, 20)),
         "answer": "28 teeth are visualized here too (a repeated question, same image)."},
        {"image_name": "img2.png", "image": b64_png((10, 10)),
         "answer": "no count mentioned for this one"},
    ]
    path = tmp_path / "open_ended.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_main_scores_detector_count_against_reference_count(tmp_path, monkeypatch, capsys):
    data_path = make_open_ended(tmp_path)
    model = FakeModel({(20, 20): [0, 1, 2], (10, 10): [0, 1]})
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: model)
    out_csv = tmp_path / "mmoral_counts.csv"

    monkeypatch.setattr("sys.argv", ["infer_mmoral.py", "--weights", "fake.pt",
                                     "--data", str(data_path), "--out", str(out_csv)])
    import infer_mmoral
    infer_mmoral.main()

    rows = out_csv.read_text().splitlines()
    assert rows[0] == "image,detector_count,ref_count,diff"
    assert "img1.png,3,28,-25" in rows
    # img2 has no reference count, so ref_count and diff are both blank
    assert "img2.png,2,," in rows

    out = capsys.readouterr().out
    assert "MMOral cross-scanner check (1 images with a trusted reference count):" in out
    assert "exact match:        0/1 (0%)" in out


def test_main_writes_the_full_fdi_map_when_requested(tmp_path, monkeypatch):
    data_path = make_open_ended(tmp_path)
    model = FakeModel({(20, 20): [0, 1, 2], (10, 10): [0, 1]})
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: model)
    out_csv = tmp_path / "mmoral_counts.csv"
    map_out = tmp_path / "mmoral_map.json"

    monkeypatch.setattr("sys.argv", ["infer_mmoral.py", "--weights", "fake.pt",
                                     "--data", str(data_path), "--out", str(out_csv),
                                     "--map-out", str(map_out)])
    import infer_mmoral
    infer_mmoral.main()

    tmap = json.load(open(map_out))
    assert tmap["img1.png"]["count"] == 3
    assert [t["fdi"] for t in tmap["img1.png"]["teeth"]] == ["11", "12", "13"]
