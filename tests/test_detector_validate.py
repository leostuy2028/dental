# Tests for detector/validate.py
# Stage 3, "the GATE": standard mAP plus the count/FDI metrics that matter for this
# project, on a YOLO val split. No real model or weights: a hand-written fake model
# stands in for ultralytics.YOLO, with canned boxes per image and per NMS mode.

import os

import ultralytics


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


class FakeValMetrics:
    def __init__(self, map50, map_):
        self.box = type("Box", (), {"map50": map50, "map": map_})()


class FakeModel:
    """detections, keyed by image basename and NMS mode ('raw' = per-class NMS,
    'ag' = agnostic NMS), plus a canned val() result."""
    def __init__(self, detections, map50=0.9, map_=0.8):
        self.detections = detections
        self.map50, self.map_ = map50, map_
        self.val_calls = []
        self.predict_calls = []

    def val(self, **kwargs):
        self.val_calls.append(kwargs)
        return FakeValMetrics(self.map50, self.map_)

    def predict(self, source, imgsz, conf, agnostic_nms, verbose):
        name = os.path.basename(source)
        key = "ag" if agnostic_nms else "raw"
        cls = self.detections[name][key]
        self.predict_calls.append((name, agnostic_nms))
        return [FakeResult(cls, [0.9] * len(cls), [[0, 0, 1, 1]] * len(cls))]


def make_yaml_dataset(tmp_path, nc, images):
    """images = {stem: true_classes_list}. Writes a tiny YOLO-style val split."""
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    for stem, classes in images.items():
        (tmp_path / "images" / "val" / f"{stem}.png").write_bytes(b"")
        lines = [f"{c} 0.1 0.1 0.1 0.1" for c in classes]
        (tmp_path / "labels" / "val" / f"{stem}.txt").write_text("\n".join(lines))
    yaml_path = tmp_path / "dentex.yaml"
    yaml_path.write_text(f"path: {tmp_path}\ntrain: images/train\nval: images/val\nnc: {nc}\n")
    return yaml_path


def test_main_multiclass_reports_count_and_fdi_accuracy(tmp_path, monkeypatch, capsys):
    yaml_path = make_yaml_dataset(tmp_path, nc=32, images={
        "img1": [0, 1, 2],   # true count 3
        "img2": [5, 6],      # true count 2
    })
    detections = {
        # raw (per-class NMS) over-counts img1 with a duplicated class 2, agnostic fixes it
        "img1.png": {"raw": [0, 1, 2, 2], "ag": [0, 1, 2]},
        "img2.png": {"raw": [5, 6], "ag": [5, 6]},
    }
    model = FakeModel(detections)
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: model)
    report = tmp_path / "report.csv"

    monkeypatch.setattr("sys.argv", ["validate.py", "--weights", "fake.pt",
                                     "--data", str(yaml_path), "--report", str(report)])
    import validate
    validate.main()

    out = capsys.readouterr().out
    assert "DETECTION:  mAP50 = 0.900   mAP50-95 = 0.800" in out
    assert "COUNT raw      (per-class NMS):    exact  1/2 (50%)" in out
    assert "COUNT physical (agnostic NMS):     exact  2/2 (100%)" in out
    assert "COUNT enum     (agnostic + 1/FDI): exact  2/2 (100%)" in out
    assert "FDI:           5/5 true teeth detected with the right number (100%)" in out

    rows = report.read_text().splitlines()
    assert rows[0] == "image,true_count,raw_count,physical_count,enum_count,physical_error"
    assert "img1,3,4,3,3,0" in rows
    assert "img2,2,2,2,2,0" in rows

    # both predict_boxes calls (raw then agnostic) were actually made for each image
    assert ("img1.png", False) in model.predict_calls
    assert ("img1.png", True) in model.predict_calls


def test_main_single_class_mode_suppresses_the_fdi_readout(tmp_path, monkeypatch, capsys):
    yaml_path = make_yaml_dataset(tmp_path, nc=1, images={"img1": [0, 0]})
    detections = {"img1.png": {"raw": [0, 0], "ag": [0, 0]}}
    model = FakeModel(detections, map50=0.5, map_=0.4)
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: model)

    monkeypatch.setattr("sys.argv", ["validate.py", "--weights", "fake.pt", "--data", str(yaml_path)])
    import validate
    validate.main()

    out = capsys.readouterr().out
    assert "<- THE GATE in single-class mode" in out
    assert "COUNT enum / FDI:                  n/a" in out
    assert "FDI:" not in out.split("n/a")[1].split("\n")[0]


def test_main_image_with_no_label_file_counts_as_zero_true_teeth(tmp_path, monkeypatch, capsys):
    # an image can exist without a matching .txt label; the true count is then 0
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    (tmp_path / "images" / "val" / "lonely.png").write_bytes(b"")
    yaml_path = tmp_path / "dentex.yaml"
    yaml_path.write_text(f"path: {tmp_path}\ntrain: images/train\nval: images/val\nnc: 32\n")

    model = FakeModel({"lonely.png": {"raw": [], "ag": []}})
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: model)
    monkeypatch.setattr("sys.argv", ["validate.py", "--weights", "fake.pt", "--data", str(yaml_path)])
    import validate
    validate.main()

    out = capsys.readouterr().out
    assert "exact  1/1 (100%)" in out  # 0 predicted == 0 true
