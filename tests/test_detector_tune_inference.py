# Tests for detector/tune_inference.py
# Chooses the detector's inference settings (image size, confidence, NMS iou) by sweeping
# against the benchmark's own dentist-confirmed tooth counts. The key trick this file
# relies on (documented in its own docstring) is that one low-conf inference pass can be
# re-thresholded in Python to simulate every higher conf, so tests work entirely off
# cached confidence lists -- no real model needed for most of it.

import base64
import io
import json

import pandas as pd
import ultralytics
from PIL import Image

from tune_inference import count_at, exact_rate, gt_count, load_raw, reference_counts, settings


# ---------- gt_count ----------

def test_gt_count_extracts_the_reference_count():
    assert gt_count("28 teeth are visualized in the radiograph.") == 28


def test_gt_count_none_when_absent():
    assert gt_count("nothing to see here") is None


# ---------- reference_counts ----------

def test_reference_counts_builds_a_name_to_count_dict(tmp_path):
    rows = [
        {"image_name": "img1.png", "answer": "28 teeth are visualized."},
        {"image_name": "img1.png", "answer": "28 teeth are visualized (asked twice)."},
        {"image_name": "img2.png", "answer": "no count here"},
    ]
    path = tmp_path / "open.parquet"
    pd.DataFrame(rows).to_parquet(path)
    op, ref = reference_counts(str(path))
    assert ref == {"img1.png": 28}
    assert "img2.png" not in ref
    assert len(op) == 3


# ---------- settings / count_at / exact_rate ----------

def test_settings_is_the_cross_product_of_res_conf_iou(monkeypatch):
    import tune_inference
    monkeypatch.setattr(tune_inference, "RES", [1024])
    monkeypatch.setattr(tune_inference, "IOU", [0.7])
    monkeypatch.setattr(tune_inference, "CONF", [0.1, 0.2])
    assert settings() == [(1024, 0.1, 0.7), (1024, 0.2, 0.7)]


def test_count_at_counts_confidences_at_or_above_threshold():
    raw = {(1024, 0.7, "img1.png"): [0.9, 0.5, 0.2]}
    assert count_at(raw, (1024, 0.5, 0.7), "img1.png") == 2
    assert count_at(raw, (1024, 0.95, 0.7), "img1.png") == 0
    assert count_at(raw, (1024, 0.1, 0.7), "img1.png") == 3


def test_exact_rate_is_the_percentage_of_exact_matches():
    raw = {(1024, 0.7, "a"): [0.9, 0.9], (1024, 0.7, "b"): [0.9, 0.9, 0.9]}
    ref = {"a": 2, "b": 2}   # "b" is off by one at this threshold
    assert exact_rate(raw, ref, (1024, 0.5, 0.7), ["a", "b"]) == 50.0


# ---------- load_raw ----------

def test_load_raw_reads_a_v2_cache_with_the_iou_axis(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps({"1024|0.45|img1.png": [0.9, 0.5]}))

    class Args:
        pass
    args = Args()
    args.raw, args.refresh = str(raw_path), False
    out = load_raw(args, op=None)
    assert out == {(1024, 0.45, "img1.png"): [0.9, 0.5]}


def test_load_raw_reads_a_v1_cache_written_before_iou_was_an_axis(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps({"1024|img1.png": [0.9, 0.5]}))

    class Args:
        pass
    args = Args()
    args.raw, args.refresh = str(raw_path), False
    out = load_raw(args, op=None)
    # a v1 cache predates the iou axis, so it's assumed to be ultralytics' default 0.7
    assert out == {(1024, 0.7, "img1.png"): [0.9, 0.5]}


def test_load_raw_runs_inference_and_writes_the_cache_when_missing(tmp_path, monkeypatch):
    class FakeTensor:
        def __init__(self, v):
            self.v = v
        def tolist(self):
            return self.v

    class FakeResult:
        def __init__(self, conf):
            self.boxes = type("B", (), {"conf": FakeTensor(conf)})()

    class FakeModel:
        def predict(self, source, imgsz, conf, iou, agnostic_nms, verbose):
            return [FakeResult([0.9, 0.5, 0.1])]

    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: FakeModel())

    def b64_png():
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    op = pd.DataFrame([{"image_name": "img1.png", "image": b64_png()}])
    raw_path = tmp_path / "raw.json"

    class Args:
        pass
    args = Args()
    args.raw, args.refresh, args.weights = str(raw_path), False, "fake.pt"

    import tune_inference
    monkeypatch.setattr(tune_inference, "RES", [1024])
    monkeypatch.setattr(tune_inference, "IOU", [0.7])

    out = load_raw(args, op)
    assert out == {(1024, 0.7, "img1.png"): [0.9, 0.5, 0.1]}
    assert json.loads(raw_path.read_text()) == {"1024|0.7|img1.png": [0.9, 0.5, 0.1]}


# ---------- main() end to end ----------

def b64_png():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(200, 200, 200)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    open_rows = [
        {"image_name": "img1.png", "image": b64_png(),
         "answer": "28 teeth are visualized in the radiograph."},
        {"image_name": "img2.png", "image": b64_png(),
         "answer": "26 teeth are visualized in the radiograph."},
    ]
    pd.DataFrame(open_rows).to_parquet("open.parquet")
    closed_rows = [
        {"index": 0, "file_name": "img1.png", "question": "How many teeth are visualized?",
         "option1": "26", "option2": "27", "option3": "28", "option4": "29", "answer": "C"},
    ]
    pd.DataFrame(closed_rows).to_parquet("closed.parquet")
    raw = {
        "1024|0.7|img1.png": sorted([0.9] * 28 + [0.3] * 2, reverse=True),
        "1024|0.7|img2.png": sorted([0.9] * 26 + [0.3] * 3, reverse=True),
    }
    (tmp_path / "raw.json").write_text(json.dumps(raw))
    (tmp_path / "results" / "closed_ended" / "modelA").mkdir(parents=True)
    pd.DataFrame([{"index": 0, "correct": "True"}]).to_csv(
        "results/closed_ended/modelA/gemini__n491.csv", index=False)


def test_main_sweeps_the_grid_and_reports_the_shipped_setting(tmp_path, monkeypatch, capsys):
    make_project(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", [
        "tune_inference.py", "--weights", "fake.pt", "--open", "open.parquet",
        "--closed", "closed.parquet", "--raw", "raw.json", "--out", "sweep.csv",
        "--splits", "2", "--res", "1024", "--iou", "0.7",
    ])
    import tune_inference
    tune_inference.main()

    out = capsys.readouterr().out
    assert "2 images carry a dentist-confirmed reference count" in out
    # 0.35 is the first conf threshold where both images' inflated confidence lists
    # drop back down to their true counts (28 and 26 real boxes, 0.3-conf noise boxes)
    assert "best     imgsz=1024, conf=0.35, iou=0.7:  exact 100.0%" in out
    assert "shipped  imgsz=1024, conf=0.25, iou=0.7:  exact 0.0%" in out
    assert "tuned setting, held out : 100.0%" in out
    assert "1 pure-count multiple-choice questions on 1 images." in out
    assert "detector, tuned   : 100.0%" in out
    assert "100.0%  gemini__n491.csv" in out
    assert "25.0%  chance" in out

    rows = (tmp_path / "sweep.csv").read_text().splitlines()
    assert rows[0] == "imgsz,conf,iou,n,exact,within1,mae"
    assert "1024,0.35,0.7,2,100.0,100.0,0.0" in rows


def test_main_ignores_non_n491_superseded_and_malformed_csvs(tmp_path, monkeypatch, capsys):
    make_project(tmp_path, monkeypatch)
    results_dir = tmp_path / "results" / "closed_ended" / "modelA"
    pd.DataFrame([{"index": 0, "correct": "True"}]).to_csv(results_dir / "old_run.csv", index=False)
    pd.DataFrame([{"index": 0, "correct": "True"}]).to_csv(
        results_dir / "superseded_run__n491.csv", index=False)
    pd.DataFrame([{"foo": 1}]).to_csv(results_dir / "malformed__n491.csv", index=False)

    monkeypatch.setattr("sys.argv", [
        "tune_inference.py", "--weights", "fake.pt", "--open", "open.parquet",
        "--closed", "closed.parquet", "--raw", "raw.json", "--out", "sweep.csv",
        "--splits", "1", "--res", "1024", "--iou", "0.7",
    ])
    import tune_inference
    tune_inference.main()

    out = capsys.readouterr().out
    # only the one well-formed, non-superseded, n491 run shows up in the score list
    assert out.count("gemini__n491.csv") == 1
    assert "old_run" not in out
    assert "superseded_run" not in out
    assert "malformed" not in out


def test_main_without_the_shipped_setting_in_range_omits_that_readout(tmp_path, monkeypatch, capsys):
    # SHIPPED is (1024, 0.25, 0.7); sweeping only imgsz=1536 means the "shipped" and
    # "detector, shipped" comparison lines must not be printed at all
    make_project(tmp_path, monkeypatch)
    raw = {
        "1536|0.7|img1.png": sorted([0.9] * 28, reverse=True),
        "1536|0.7|img2.png": sorted([0.9] * 26, reverse=True),
    }
    (tmp_path / "raw.json").write_text(json.dumps(raw))
    monkeypatch.setattr("sys.argv", [
        "tune_inference.py", "--weights", "fake.pt", "--open", "open.parquet",
        "--closed", "closed.parquet", "--raw", "raw.json", "--out", "sweep.csv",
        "--splits", "1", "--res", "1536", "--iou", "0.7",
    ])
    import tune_inference
    tune_inference.main()

    out = capsys.readouterr().out
    assert "shipped " not in out
    assert "detector, shipped" not in out
