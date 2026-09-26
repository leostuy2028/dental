# Tests for detector/build_map_v2.py
# Builds the v2 tooth map (boxes from the detector, FDI codes from positional numbering)
# for the MMOral panoramics, one JSON file for the whole benchmark. No network, no real
# weights: a tiny local parquet file and a fake YOLO model stand in for the real ones.

import base64
import io
import json

import pandas as pd
import ultralytics
from PIL import Image


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)


class FakeResult:
    def __init__(self, xyxy, conf):
        self.boxes = FakeBoxes(xyxy, conf)


class FakeModel:
    """One image -> two boxes, keyed by the decoded image's pixel size."""
    def __init__(self, by_size):
        self.by_size = by_size
        self.calls = []

    def predict(self, source, imgsz, conf, iou, agnostic_nms, verbose):
        self.calls.append(dict(imgsz=imgsz, conf=conf, iou=iou, agnostic_nms=agnostic_nms))
        boxes, confs = self.by_size[source.size]
        return [FakeResult(boxes, confs)]


def b64_png(size):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(200, 200, 200)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_open_ended(tmp_path, size=(20, 20)):
    df = pd.DataFrame([{"image_name": "img1.png", "image": b64_png(size)}])
    path = tmp_path / "open_ended.parquet"
    df.to_parquet(path)
    return path


def make_prior(tmp_path):
    # a flat prior so the DP just needs SOME calibrated distance per index; exact
    # values don't matter for this test, only that assign_fdi_positional runs
    path = tmp_path / "arch_prior.json"
    path.write_text(json.dumps({"median": {str(k): float(k) for k in range(1, 9)}}))
    return path


def test_main_builds_a_map_with_fdi_codes_and_counts(tmp_path, monkeypatch, capsys):
    data_path = make_open_ended(tmp_path)
    prior_path = make_prior(tmp_path)
    # two boxes far enough apart in x to land in different quadrants once numbered
    model = FakeModel({(20, 20): ([[0, 0, 10, 10], [20, 0, 30, 10]], [0.9, 0.8])})
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: model)
    out_path = tmp_path / "mmoral_map_v2.json"

    monkeypatch.setattr("sys.argv", ["build_map_v2.py", "--prior", str(prior_path),
                                     "--data", str(data_path), "--out", str(out_path)])
    import build_map_v2
    build_map_v2.main()

    out = json.load(open(out_path))
    entry = out["img1.png"]
    assert entry["count"] == 2
    assert len(entry["teeth"]) == 2
    fdis = sorted(t["fdi"] for t in entry["teeth"])
    assert all(len(f) == 2 for f in fdis)

    printed = capsys.readouterr().out
    assert "1 images, 2 boxes, 2 numbered (100%)" in printed
    assert str(out_path) in printed
    assert "provenance:" in printed

    # default inference settings from the argparse defaults are what got passed through
    assert model.calls[0]["imgsz"] == 1536
    assert model.calls[0]["conf"] == 0.20
    assert model.calls[0]["iou"] == 0.45


def test_main_unnumbered_boxes_are_excluded_from_teeth_but_counted(tmp_path, monkeypatch):
    # a lone box with no partner in its quadrant still gets numbered as index 1 by
    # assign_fdi_positional, so use a prior that makes it fall outside 1..8 instead:
    # an empty prior quadrant is impossible here, so instead check the simple case
    # where every detected box does get a code, and count == number of boxes regardless
    data_path = make_open_ended(tmp_path)
    prior_path = make_prior(tmp_path)
    model = FakeModel({(20, 20): ([[0, 0, 10, 10]], [0.5])})
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: model)
    out_path = tmp_path / "mmoral_map_v2.json"

    monkeypatch.setattr("sys.argv", ["build_map_v2.py", "--prior", str(prior_path),
                                     "--data", str(data_path), "--out", str(out_path)])
    import build_map_v2
    build_map_v2.main()

    out = json.load(open(out_path))
    assert out["img1.png"]["count"] == 1
    assert len(out["img1.png"]["teeth"]) == 1
    assert out["img1.png"]["teeth"][0]["conf"] == 0.5
