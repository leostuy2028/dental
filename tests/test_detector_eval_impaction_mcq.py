# Tests for detector/eval_impaction_mcq.py
# Head-to-head: the detector answering the benchmark's IMPACTION multiple-choice
# questions directly (by composing the disease detector with tooth numbering), against
# a committed gemini baseline CSV. No API calls, no real weights: fake tooth/disease
# models and tiny local parquet/CSV files.

import base64
import io
import json

import pandas as pd
import ultralytics
from PIL import Image

from eval_impaction_mcq import CODE_RE, NUM_RE, centre
from number_teeth import assign_fdi


def test_centre_is_the_box_midpoint():
    assert centre((0, 0, 10, 10)) == (5.0, 5.0)


def test_num_re_matches_a_bare_number_with_surrounding_space():
    assert NUM_RE.match(" 3 ")


def test_num_re_rejects_a_number_with_extra_text():
    assert NUM_RE.match("3 teeth") is None


def test_code_re_finds_fdi_codes_with_or_without_a_hash():
    assert CODE_RE.findall("#18, 28, and #38") == ["18", "28", "38"]


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, xyxy, cls):
        self.xyxy = FakeTensor(xyxy)
        self.cls = FakeTensor(cls)


class FakeResult:
    def __init__(self, xyxy, cls):
        self.boxes = FakeBoxes(xyxy, cls)


class FakeToothModel:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, source, imgsz, conf, iou, agnostic_nms, verbose):
        return [FakeResult(self.boxes, [0] * len(self.boxes))]


class FakeDiseaseModel:
    names = {0: "Impacted"}

    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, source, imgsz, conf, verbose):
        return [FakeResult(self.boxes, [0] * len(self.boxes))]


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


def b64_png(size=(10, 10)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(200, 200, 200)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_main_scores_count_and_code_questions_against_a_baseline(tmp_path, monkeypatch, capsys):
    boxes = full_mouth()
    codes = assign_fdi(boxes)
    b38, b48 = boxes[codes.index("38")], boxes[codes.index("48")]

    closed_rows = [
        {"index": 0, "file_name": "img1.png", "question": "How many teeth are impacted?",
         "option1": "1", "option2": "2", "option3": "3", "option4": "4", "answer": "B"},
        {"index": 1, "file_name": "img1.png", "question": "Which teeth are impacted?",
         "option1": "#18", "option2": "#28", "option3": "#38", "option4": "#48", "answer": "C"},
        {"index": 2, "file_name": "img1.png", "question": "Are wisdom teeth impacted?",
         "option1": "Yes, all", "option2": "No, none", "option3": "Some",
         "option4": "Unclear", "answer": "A"},
        {"index": 3, "file_name": "img2.png", "question": "How many teeth are visualized?",
         "option1": "26", "option2": "27", "option3": "28", "option4": "29", "answer": "C"},
    ]
    closed_path = tmp_path / "closed.parquet"
    pd.DataFrame(closed_rows).to_parquet(closed_path)
    open_path = tmp_path / "open.parquet"
    pd.DataFrame([{"image_name": "img1.png", "image": b64_png()},
                 {"image_name": "img2.png", "image": b64_png()}]).to_parquet(open_path)
    prior_path = tmp_path / "arch_prior.json"
    prior_path.write_text(json.dumps({"median": {str(k): float(k) for k in range(1, 9)}}))
    baseline_path = tmp_path / "baseline.csv"
    pd.DataFrame([{"index": 0, "correct": "True"}, {"index": 1, "correct": "False"},
                 {"index": 2, "correct": "True"}]).to_csv(baseline_path, index=False)

    tooth = FakeToothModel(boxes)
    disease = FakeDiseaseModel([b38, b48])
    seq = iter([tooth, disease])
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: next(seq))

    out_csv = tmp_path / "impaction_mcq.csv"
    monkeypatch.setattr("sys.argv", ["eval_impaction_mcq.py", "--prior", str(prior_path),
                                     "--closed", str(closed_path), "--open", str(open_path),
                                     "--baseline", str(baseline_path), "--out", str(out_csv)])
    import eval_impaction_mcq
    eval_impaction_mcq.main()

    rows = out_csv.read_text().splitlines()
    assert rows[0] == "index,image,mode,detector_pred,key,detector_ok,gemini_ok"
    assert "0,img1.png,count,B,B,True,True" in rows
    assert "1,img1.png,codes,C,C,True,False" in rows
    # prose-options question: mode is empty and detector_pred is empty
    assert "2,img1.png,,,A,False,True" in rows
    # the non-impaction question (index 3) never made it into the impaction subset at all
    assert not any(row.startswith("3,") for row in rows)

    out = capsys.readouterr().out
    assert "3 closed questions mention impaction" in out
    assert "detector can express an answer for 2 of 3 (1 have prose options and go to the VLM)" in out
    assert "detector          : 100.0%" in out
    assert "gemini-3.5-flash  : 50.0%" in out
    assert "count  questions  n=  1   detector 100.0%   gemini 100.0%" in out
    assert "codes  questions  n=  1   detector 100.0%   gemini   0.0%" in out


def test_main_composing_skips_a_tooth_box_with_no_fdi_code(tmp_path, monkeypatch):
    # curve=0.0 keeps rounding/split_arches deterministic; 9 teeth on one flat row
    # means one per side has no valid index (max is 8) and gets code=None. An impaction
    # box centred on that uncoded tooth must still resolve to the nearest CODED tooth.
    boxes = arch_boxes(100, 0.0, n_side=9)
    prior = {str(k): float(k) for k in range(1, 9)}
    tb = [tuple(round(v) for v in b) for b in boxes]
    from number_teeth import assign_fdi_positional
    codes = assign_fdi_positional(tb, prior)
    none_idx = codes.index(None)
    imp_box = list(boxes[none_idx])

    closed_rows = [
        {"index": 0, "file_name": "img1.png", "question": "Which teeth are impacted?",
         "option1": "#41", "option2": "#42", "option3": "#43", "option4": "#44", "answer": "A"},
    ]
    closed_path = tmp_path / "closed.parquet"
    pd.DataFrame(closed_rows).to_parquet(closed_path)
    open_path = tmp_path / "open.parquet"
    pd.DataFrame([{"image_name": "img1.png", "image": b64_png()}]).to_parquet(open_path)
    prior_path = tmp_path / "arch_prior.json"
    prior_path.write_text(json.dumps({"median": prior}))
    baseline_path = tmp_path / "baseline.csv"
    pd.DataFrame([{"index": 0, "correct": "False"}]).to_csv(baseline_path, index=False)

    tooth = FakeToothModel(boxes)
    disease = FakeDiseaseModel([imp_box])
    seq = iter([tooth, disease])
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: next(seq))

    out_csv = tmp_path / "impaction_mcq.csv"
    monkeypatch.setattr("sys.argv", ["eval_impaction_mcq.py", "--prior", str(prior_path),
                                     "--closed", str(closed_path), "--open", str(open_path),
                                     "--baseline", str(baseline_path), "--out", str(out_csv)])
    import eval_impaction_mcq
    eval_impaction_mcq.main()

    rows = out_csv.read_text().splitlines()
    assert "0,img1.png,codes,A,A,True,False" in rows
