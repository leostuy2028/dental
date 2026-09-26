# Tests for detector/eval_numbering.py
# Scores geometric FDI numbering (number_teeth.py) against the benchmark's own reference
# answers: which wisdom teeth (by FDI code) does the reference say are present, and does
# our numbering agree, both as a set (readout A) and box-by-box (readout B). No network,
# no real weights: a tiny local parquet file and a fake YOLO model.

import base64
import io
import json

import pandas as pd
import ultralytics
from PIL import Image

from eval_numbering import BOX_RE, CODE_RE, ID_RE, iou


# ---------- iou ----------

def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_of_partially_overlapping_boxes():
    assert round(iou((0, 0, 10, 10), (5, 5, 15, 15)), 4) == round(25 / 175, 4)


def test_iou_of_non_overlapping_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_of_two_zero_area_boxes_is_zero():
    # union area is 0, and the function must not divide by zero
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


# ---------- the reference-text regexes ----------

def test_box_re_extracts_four_ints():
    text = '"box_2d": [10, 10, 50, 50], "tooth_id": "18"'
    assert BOX_RE.findall(text) == [("10", "10", "50", "50")]


def test_id_re_extracts_the_tooth_code():
    text = '"box_2d": [10, 10, 50, 50], "tooth_id": "18"'
    assert ID_RE.findall(text) == ["18"]


def test_code_re_finds_all_third_molar_codes():
    text = "Four wisdom teeth are detected: #18, #28, #38, and #48."
    assert CODE_RE.findall(text) == ["18", "28", "38", "48"]


# ---------- main() end to end ----------

class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, xyxy):
        self.xyxy = FakeTensor(xyxy)


class FakeResult:
    def __init__(self, xyxy):
        self.boxes = FakeBoxes(xyxy)


class FakeModel:
    def __init__(self, by_size):
        self.by_size = by_size

    def predict(self, source, imgsz, conf, iou, agnostic_nms, verbose):
        return [FakeResult(self.by_size[source.size])]


def arch_boxes(y_base, curve, n_side=8, spacing=20, width=18):
    boxes = []
    for side in (-1, 1):
        for i in range(1, n_side + 1):
            x = side * i * spacing
            y = y_base + curve * (x ** 2) / 10000.0
            boxes.append([x - width / 2, y - 10, x + width / 2, y + 10])
    return boxes


def full_mouth():
    return arch_boxes(100, 1.0) + arch_boxes(300, -1.0)


def b64_png(size):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(200, 200, 200)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_open_ended(tmp_path):
    rows = [
        {"image_name": "img1.png", "image": b64_png((20, 20)),
         "question": "How many wisdom teeth are detected?",
         "answer": "Four wisdom teeth are detected: #18, #28, #38, and #48."},
        {"image_name": "img2.png", "image": b64_png((10, 10)),
         "question": "How many wisdom teeth are present?",
         "answer": "No wisdom teeth are present in this radiograph."},
        {"image_name": "img3.png", "image": b64_png((30, 30)),
         "question": "Where are this patient's wisdom teeth?",
         "answer": '{"box_2d": [0, 0, 10, 10], "tooth_id": "18"}'},
        {"image_name": "img4.png", "image": b64_png((40, 40)),
         "question": "How many teeth are visualized?",   # not a wisdom question at all
         "answer": "28 teeth are visualized."},
    ]
    path = tmp_path / "open_ended.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_main_ordinal_scores_the_wisdom_set_and_box_level(tmp_path, monkeypatch, capsys):
    data_path = make_open_ended(tmp_path)
    by_size = {
        (20, 20): full_mouth(),   # img1: full mouth -> ordinal numbering finds all 4
        (10, 10): [],             # img2: no boxes, matches the "no wisdom teeth" truth
        (30, 30): [[0, 0, 10, 10]],   # img3: one box for the box-level readout
        (40, 40): [],              # img4: never touched (not a wisdom question)
    }
    model = FakeModel(by_size)
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: model)
    out_csv = tmp_path / "numbering_wisdom.csv"

    monkeypatch.setattr("sys.argv", ["eval_numbering.py", "--weights", "fake.pt",
                                     "--data", str(data_path), "--out", str(out_csv)])
    import eval_numbering
    eval_numbering.main()

    out = capsys.readouterr().out
    assert "2 images whose reference answer settles which wisdom teeth are present" in out
    assert "A. WISDOM-TOOTH SET, 2 images" in out
    assert "exact set match      : 2/2  (100%)" in out
    assert "B. BOX-LEVEL, 1 reference wisdom-tooth boxes on 1 images" in out
    assert "detector found the tooth (IoU>=0.3): 1/1" in out

    rows = out_csv.read_text().splitlines()
    assert "image,n_boxes,truth,pred,exact_set,n_truth,n_pred,count_ok,hit,miss,extra" == rows[0]
    assert "img1.png,32,18|28|38|48,18|28|38|48,True,4,4,True,4,0,0" in rows
    assert "img2.png,0,,,True,0,0,True,0,0,0" in rows
    assert not any(row.startswith("img4.png") for row in rows)  # not a wisdom question


def test_main_prints_worst_rows_and_skips_a_malformed_box_2d_answer(tmp_path, monkeypatch, capsys):
    rows = [
        {"image_name": "img1.png", "image": b64_png((20, 20)),
         "question": "How many wisdom teeth are detected?",
         "answer": "Four wisdom teeth are detected: #18, #28, #38, and #48."},
        {"image_name": "img2.png", "image": b64_png((30, 30)),
         "question": "Where are this patient's wisdom teeth?",
         # has "box_2d" in the text but no bracketed coordinates for BOX_RE to find
         "answer": "box_2d is not available for this image"},
    ]
    path = tmp_path / "open_ended.parquet"
    pd.DataFrame(rows).to_parquet(path)

    by_size = {(20, 20): [], (30, 30): []}   # detector finds nothing -> mismatches img1's truth
    model = FakeModel(by_size)
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: model)
    out_csv = tmp_path / "numbering_wisdom.csv"

    monkeypatch.setattr("sys.argv", ["eval_numbering.py", "--weights", "fake.pt",
                                     "--data", str(path), "--out", str(out_csv)])
    import eval_numbering
    eval_numbering.main()

    out = capsys.readouterr().out
    assert "exact set match      : 0/1  (0%)" in out
    assert "img1.png  truth[18|28|38|48]  pred[]" in out
    # img2 never reaches the box-level CSV rows since its box_2d text had no coordinates
    assert "B. BOX-LEVEL" not in out


def test_main_positional_method_reads_the_prior_and_prints_its_source(tmp_path, monkeypatch, capsys):
    data_path = make_open_ended(tmp_path)
    by_size = {(20, 20): full_mouth(), (10, 10): [], (30, 30): [[0, 0, 10, 10]], (40, 40): []}
    model = FakeModel(by_size)
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: model)
    prior_path = tmp_path / "arch_prior.json"
    prior_path.write_text(json.dumps({"median": {str(k): k * 1.1111 for k in range(1, 9)}}))
    out_csv = tmp_path / "numbering_wisdom.csv"

    monkeypatch.setattr("sys.argv", ["eval_numbering.py", "--weights", "fake.pt",
                                     "--data", str(data_path), "--out", str(out_csv),
                                     "--method", "positional", "--prior", str(prior_path)])
    import eval_numbering
    eval_numbering.main()

    out = capsys.readouterr().out
    assert f"positional numbering, prior from {prior_path}" in out
    assert "exact set match      : 2/2  (100%)" in out
