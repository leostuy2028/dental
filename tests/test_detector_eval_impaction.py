# Tests for detector/eval_impaction.py
# Does the disease detector's Impacted class survive DENTEX -> MMOral? Ground truth is
# parsed straight out of the benchmark's own reference sentences ("#38 and #48 are
# impacted"). No network, no real weights: fake tooth/disease YOLO models and a tiny
# local parquet file.

import base64
import io
import json

import pandas as pd
import ultralytics
from PIL import Image

from eval_impaction import CLASS_RE, FDI_RE, LINE_SPLIT, NEG_RE, centre


# ---------- small helpers / regexes ----------

def test_centre_is_the_box_midpoint():
    assert centre((0, 0, 10, 20)) == (5.0, 10.0)


def test_fdi_re_finds_hash_coded_teeth():
    assert FDI_RE.findall("Impacted teeth: #38 and #48.") == [("38", ""), ("48", "")]


def test_fdi_re_ignores_a_bare_number_before_the_word_teeth():
    # a plain FDI-shaped number immediately followed by "teeth"/"tooth" is a count, not
    # a code (this is the exact regression the module's docstring describes)
    assert FDI_RE.findall("26 teeth visualized") == []


def test_line_split_splits_on_newlines_semicolons_and_sentence_ends():
    assert LINE_SPLIT.split("Line one. Line two.\nLine three") == \
        ["Line one.", "Line two.", "Line three"]


def test_neg_re_detects_a_negative_impaction_statement():
    assert NEG_RE.search("No teeth are impacted.")


def test_neg_re_does_not_fire_on_a_positive_statement():
    assert NEG_RE.search("Tooth #38 is impacted.") is None


def test_class_re_has_one_pattern_per_diagnosis_class():
    assert set(CLASS_RE) == {"Impacted", "Caries", "Deep Caries", "Periapical Lesion"}


# ---------- main() end to end ----------

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
    def __init__(self, by_size):
        self.by_size = by_size

    def predict(self, source, imgsz, conf, iou, agnostic_nms, verbose):
        return [FakeResult(self.by_size[source.size], [0] * len(self.by_size[source.size]))]


class FakeDiseaseModel:
    names = {0: "Impacted", 1: "Caries"}

    def __init__(self, by_size):
        self.by_size = by_size

    def predict(self, source, imgsz, conf, verbose):
        xyxy, cls = self.by_size[source.size]
        return [FakeResult(xyxy, cls)]


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


def setup_models(monkeypatch, tooth, disease):
    seq = iter([tooth, disease])
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: next(seq))


def test_main_composes_impaction_boxes_with_tooth_numbering(tmp_path, monkeypatch, capsys):
    boxes = full_mouth()
    # find the real "38" and "48" boxes from the geometry itself, not hand-guessed
    import sys
    sys.path.insert(0, "detector")
    from number_teeth import assign_fdi
    codes = assign_fdi(boxes)
    b38 = boxes[codes.index("38")]
    b48 = boxes[codes.index("48")]

    rows = [{"image_name": "img1.png", "image": b64_png((20, 20)),
             "answer": "Impacted teeth: #38 and #48 are noted."}]
    data_path = tmp_path / "open_ended.parquet"
    pd.DataFrame(rows).to_parquet(data_path)
    prior_path = tmp_path / "arch_prior.json"
    prior_path.write_text(json.dumps({"median": {str(k): float(k) for k in range(1, 9)}}))

    tooth = FakeToothModel({(20, 20): boxes})
    disease = FakeDiseaseModel({(20, 20): ([b38, b48], [0, 0])})
    setup_models(monkeypatch, tooth, disease)

    out_csv = tmp_path / "disease_mmoral.csv"
    monkeypatch.setattr("sys.argv", ["eval_impaction.py", "--prior", str(prior_path),
                                     "--data", str(data_path), "--out", str(out_csv)])
    import eval_impaction
    eval_impaction.main()

    rows = out_csv.read_text().splitlines()
    header = "image,n_impact_boxes,truth,pred,n_truth,n_pred,count_ok,exact_set,hit,miss,extra"
    assert rows[0] == header
    assert "img1.png,2,38|48,38|48,2,2,True,True,2,0,0" in rows

    out = capsys.readouterr().out
    assert "[Impacted] 1 images whose reference answers name affected teeth by FDI code" in out
    assert "exact set match : 1/1 (100%)" in out
    assert "right n findings: 1/1 (100%)" in out


def test_main_negative_statement_gives_an_empty_truth_set(tmp_path, monkeypatch, capsys):
    rows = [{"image_name": "img1.png", "image": b64_png((10, 10)),
             "answer": "No teeth are impacted in this radiograph."}]
    data_path = tmp_path / "open_ended.parquet"
    pd.DataFrame(rows).to_parquet(data_path)
    prior_path = tmp_path / "arch_prior.json"
    prior_path.write_text(json.dumps({"median": {str(k): float(k) for k in range(1, 9)}}))

    tooth = FakeToothModel({(10, 10): []})
    disease = FakeDiseaseModel({(10, 10): ([], [])})
    setup_models(monkeypatch, tooth, disease)

    out_csv = tmp_path / "disease_mmoral.csv"
    monkeypatch.setattr("sys.argv", ["eval_impaction.py", "--prior", str(prior_path),
                                     "--data", str(data_path), "--out", str(out_csv)])
    import eval_impaction
    eval_impaction.main()

    out = capsys.readouterr().out
    assert "of these, 0 assert at least one, 1 assert none" in out
    rows = out_csv.read_text().splitlines()
    assert "img1.png,0,,,0,0,True,True,0,0,0" in rows


def test_main_skips_answers_without_the_keyword_or_with_box_2d(tmp_path, monkeypatch, capsys):
    rows = [
        {"image_name": "no_mention.png", "image": b64_png((10, 10)),
         "answer": "26 teeth are visualized, no findings."},
        {"image_name": "has_box2d.png", "image": b64_png((10, 10)),
         "answer": '#38 is impacted. {"box_2d": [0,0,5,5], "tooth_id": "38"}'},
        {"image_name": "multi_sentence.png", "image": b64_png((10, 10)),
         "answer": "Mild caries noted. #38 is impacted."},
    ]
    data_path = tmp_path / "open_ended.parquet"
    pd.DataFrame(rows).to_parquet(data_path)
    prior_path = tmp_path / "arch_prior.json"
    prior_path.write_text(json.dumps({"median": {str(k): float(k) for k in range(1, 9)}}))

    box = [0, 0, 10, 10]
    tooth = FakeToothModel({(10, 10): [box]})
    disease = FakeDiseaseModel({(10, 10): ([], [])})
    setup_models(monkeypatch, tooth, disease)

    out_csv = tmp_path / "disease_mmoral.csv"
    monkeypatch.setattr("sys.argv", ["eval_impaction.py", "--prior", str(prior_path),
                                     "--data", str(data_path), "--out", str(out_csv)])
    import eval_impaction
    eval_impaction.main()

    out = capsys.readouterr().out
    # only "multi_sentence.png" settles a truth: "no_mention" never mentions impaction,
    # and "has_box2d" is excluded because its answer carries box_2d
    assert "1 images whose reference answers name affected teeth by FDI code" in out


def test_main_composing_skips_a_tooth_box_with_no_fdi_code(tmp_path, monkeypatch):
    # a 9th tooth in a quadrant gets code=None from assign_fdi_positional (max index is
    # 8); an impaction box centred exactly on that uncoded tooth must still be assigned
    # to the next-nearest CODED tooth rather than being silently dropped or crashing
    import sys as _s
    _s.path.insert(0, "detector")
    from number_teeth import assign_fdi_positional

    # curve=0.0 (a flat row, not an arc) keeps split_arches/rounding deterministic:
    # 9 teeth means one per side has no valid index (max is 8) and gets code=None
    boxes = arch_boxes(100, 0.0, n_side=9)
    prior = {str(k): float(k) for k in range(1, 9)}
    codes = assign_fdi_positional([tuple(round(v) for v in b) for b in boxes], prior)
    none_idx = codes.index(None)
    imp_box = boxes[none_idx]

    rows = [{"image_name": "img1.png", "image": b64_png((10, 10)),
             "answer": "No teeth are impacted."}]  # NEG_RE match -> truth is the empty set
    data_path = tmp_path / "open_ended.parquet"
    pd.DataFrame(rows).to_parquet(data_path)
    prior_path = tmp_path / "arch_prior.json"
    prior_path.write_text(json.dumps({"median": prior}))

    tooth = FakeToothModel({(10, 10): boxes})
    disease = FakeDiseaseModel({(10, 10): ([imp_box], [0])})
    setup_models(monkeypatch, tooth, disease)

    out_csv = tmp_path / "disease_mmoral.csv"
    monkeypatch.setattr("sys.argv", ["eval_impaction.py", "--prior", str(prior_path),
                                     "--data", str(data_path), "--out", str(out_csv)])
    import eval_impaction
    eval_impaction.main()

    rows = out_csv.read_text().splitlines()
    # the reference says no teeth are impacted (truth is the empty set), but the
    # detector still composes a prediction from the nearest *coded* tooth rather than
    # dropping the box or crashing on the uncoded one
    assert any(row.startswith("img1.png,1,,41") for row in rows)


def test_main_can_score_a_different_diagnosis_class(tmp_path, monkeypatch, capsys):
    rows = [{"image_name": "img1.png", "image": b64_png((10, 10)),
             "answer": "There is caries on #16."}]
    data_path = tmp_path / "open_ended.parquet"
    pd.DataFrame(rows).to_parquet(data_path)
    prior_path = tmp_path / "arch_prior.json"
    prior_path.write_text(json.dumps({"median": {str(k): float(k) for k in range(1, 9)}}))

    box16 = [0, 0, 10, 10]
    tooth = FakeToothModel({(10, 10): [box16]})
    disease = FakeDiseaseModel({(10, 10): ([box16], [1])})   # class 1 == "Caries"
    setup_models(monkeypatch, tooth, disease)

    out_csv = tmp_path / "disease_mmoral.csv"
    monkeypatch.setattr("sys.argv", ["eval_impaction.py", "--prior", str(prior_path),
                                     "--data", str(data_path), "--out", str(out_csv),
                                     "--cls", "Caries"])
    import eval_impaction
    eval_impaction.main()

    out = capsys.readouterr().out
    assert "[Caries]" in out
