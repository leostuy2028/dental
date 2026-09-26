# Tests for detector/eval_router.py
# Measures what the router (detector/router.py) buys across base models of different
# strength: routed questions get the detector's answer, everything else keeps that
# model's own committed answer -- exact, since routing never changes the other answers.
# No API calls, no real weights: fake YOLO models and tiny local parquet/CSV files.

import base64
import io
import json

import pandas as pd
import ultralytics
from PIL import Image

from eval_router import detector_readings, emit_paper_table


# ---------- detector_readings ----------

def test_detector_readings_uses_the_cache_without_touching_the_model(tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"img1.png": {"count": 5, "wisdom": ["18"]}}))

    def boom(*a, **kw):
        raise AssertionError("YOLO should not be constructed when the cache hits")

    monkeypatch.setattr(ultralytics, "YOLO", boom)

    class Args:
        pass
    args = Args()
    args.cache, args.refresh = str(cache), False

    out = detector_readings(args, images={"img1.png": "unused"})
    assert out == {"img1.png": {"count": 5, "wisdom": ["18"]}}


def test_detector_readings_builds_and_writes_the_cache_when_missing(tmp_path, monkeypatch):
    class FakeTensor:
        def __init__(self, v):
            self.v = v
        def tolist(self):
            return self.v

    class FakeResult:
        def __init__(self, xyxy):
            self.boxes = type("B", (), {"xyxy": FakeTensor(xyxy)})()

    class FakeModel:
        def predict(self, source, imgsz, conf, iou, agnostic_nms, verbose):
            return [FakeResult([[0, 0, 10, 10], [20, 0, 30, 10]])]

    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: FakeModel())

    def b64_png():
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps({"median": {str(k): float(k) for k in range(1, 9)}}))
    cache = tmp_path / "nested" / "cache.json"

    class Args:
        pass
    args = Args()
    args.cache, args.refresh = str(cache), False
    args.prior, args.weights = str(prior_path), "fake.pt"
    args.imgsz, args.conf, args.iou = 1536, 0.20, 0.45

    out = detector_readings(args, images={"img1.png": b64_png()})
    assert out["img1.png"]["count"] == 2
    assert json.loads(cache.read_text()) == out


# ---------- emit_paper_table ----------

def test_emit_paper_table_writes_markdown_and_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = pd.DataFrame([{
        "run": "gemini-2.5-flash__coax-direct-k0__shuffled__n491",
        "base": 60.0, "routed": 70.0, "lift": 10.0,
        "model_on_routed": 50.0, "det_on_routed": 80.0,
    }])
    emit_paper_table(res, n_routed=5, n_total=10)

    md = (tmp_path / "paper_analysis" / "_generated" / "router_lift_table.md").read_text(encoding="utf-8")
    assert "| Gemini-2.5-flash | 60.0% | 70.0% | **+10.0** |" in md
    assert "*(run not found)*" in md   # the other PAPER_ROWS entries have no matching run

    vals = json.loads((tmp_path / "paper_analysis" / "_generated" / "router_lift.values.json").read_text())
    assert vals["n_routed"] == 5
    assert vals["n_closed"] == 10
    assert vals["rows"]["Gemini-2.5-flash"]["lift"] == 10.0


# ---------- main() end to end ----------

class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeResult:
    def __init__(self, xyxy):
        self.boxes = type("B", (), {"xyxy": FakeTensor(xyxy)})()


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


class FakeModel:
    def predict(self, source, imgsz, conf, iou, agnostic_nms, verbose):
        return [FakeResult(full_mouth()[:28])]   # 28 boxes, 3 of which end up coded *8


def b64_png():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(200, 200, 200)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_project(tmp_path, monkeypatch):
    """A tiny project layout under tmp_path: closed/open parquets, an arch prior, and
    one committed model CSV to combine with."""
    monkeypatch.chdir(tmp_path)
    closed_rows = [
        {"index": 0, "file_name": "img1.png", "question": "How many teeth are visualized?",
         "option1": "26", "option2": "27", "option3": "28", "option4": "29", "answer": "C"},
        {"index": 1, "file_name": "img1.png", "question": "How many wisdom teeth are detected?",
         "option1": "0", "option2": "1", "option3": "2", "option4": "3", "answer": "D"},
        {"index": 2, "file_name": "img1.png", "question": "Are the wisdom teeth impacted?",
         "option1": "Yes", "option2": "No", "option3": "Unclear", "option4": "Some", "answer": "B"},
    ]
    pd.DataFrame(closed_rows).to_parquet("closed.parquet")
    pd.DataFrame([{"image_name": "img1.png", "image": b64_png()}]).to_parquet("open.parquet")
    (tmp_path / "detector").mkdir(exist_ok=True)
    (tmp_path / "detector" / "arch_prior.json").write_text(
        json.dumps({"median": {str(k): float(k) for k in range(1, 9)}}))
    (tmp_path / "results" / "closed_ended" / "modelA").mkdir(parents=True)
    pd.DataFrame([{"index": 0, "correct": "True"}, {"index": 1, "correct": "False"},
                 {"index": 2, "correct": "True"}]).to_csv(
        "results/closed_ended/modelA/gemini-2.5-flash__coax-direct-k0__shuffled__n491.csv",
        index=False)
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: FakeModel())


def base_argv():
    return ["eval_router.py", "--weights", "fake.pt", "--prior", "detector/arch_prior.json",
            "--closed", "closed.parquet", "--open", "open.parquet",
            "--cache", "results/detector/router_readings.json",
            "--out", "results/detector/router_lift.csv",
            "--curated", "curated/mmoral_curated_closed.csv"]


def test_main_scores_the_router_lift_and_writes_the_paper_table(tmp_path, monkeypatch, capsys):
    make_project(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", base_argv())

    import eval_router
    eval_router.main()

    out = capsys.readouterr().out
    assert "scoring all 3 items, including any" in out
    assert "router selects 2 of 3 closed questions (66.7%)" in out
    assert "detector answers 2 of them; correct on 100.0%" in out
    assert "Every routed question is an API call not made: 2/3" in out

    rows = (tmp_path / "results" / "detector" / "router_lift.csv").read_text().splitlines()
    assert rows[0] == "run,n,base,routed,lift,n_routed,model_on_routed,det_on_routed,api_calls_saved"
    assert rows[1].startswith("gemini-2.5-flash__coax-direct-k0__shuffled__n491,3,")

    cache = json.loads((tmp_path / "results" / "detector" / "router_readings.json").read_text())
    assert cache["img1.png"]["count"] == 28
    assert cache["img1.png"]["wisdom"] == ["18", "28", "48"]

    md = (tmp_path / "paper_analysis" / "_generated" / "router_lift_table.md").read_text(encoding="utf-8")
    assert "Gemini-2.5-flash" in md


def test_main_curated_set_drops_items_outside_the_reading_score(tmp_path, monkeypatch, capsys):
    make_project(tmp_path, monkeypatch)
    (tmp_path / "curated").mkdir()
    # keep only indices 0 and 1; index 2 is dropped as "not in the reading score"
    pd.DataFrame([{"index": 0, "in_reading_score": "true"},
                 {"index": 1, "in_reading_score": "true"},
                 {"index": 2, "in_reading_score": "false"}]).to_csv(
        "curated/mmoral_curated_closed.csv", index=False)
    monkeypatch.setattr("sys.argv", base_argv())

    import eval_router
    eval_router.main()

    out = capsys.readouterr().out
    assert "scoring the CURATED set: 2 items (1 dropped by" in out


def test_main_all_491_flag_skips_rewriting_the_paper_fragment(tmp_path, monkeypatch, capsys):
    make_project(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", base_argv() + ["--all-491"])

    import eval_router
    eval_router.main()

    out = capsys.readouterr().out
    assert "paper fragment NOT rewritten; it belongs to the curated run" in out
    assert not (tmp_path / "paper_analysis").exists()


def test_main_skips_a_question_with_no_detector_reading_and_ignores_unusable_csvs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # img2.png is routed but never appears in open.parquet, so det.get(img2.png) is None
    closed_rows = [
        {"index": 0, "file_name": "img2.png", "question": "How many teeth are visualized?",
         "option1": "26", "option2": "27", "option3": "28", "option4": "29", "answer": "C"},
    ]
    pd.DataFrame(closed_rows).to_parquet("closed.parquet")
    pd.DataFrame([{"image_name": "img1.png", "image": b64_png()}]).to_parquet("open.parquet")
    (tmp_path / "detector").mkdir(exist_ok=True)
    (tmp_path / "detector" / "arch_prior.json").write_text(
        json.dumps({"median": {str(k): float(k) for k in range(1, 9)}}))

    results_dir = tmp_path / "results" / "closed_ended" / "modelA"
    results_dir.mkdir(parents=True)
    # a usable run
    pd.DataFrame([{"index": 0, "correct": "True"}]).to_csv(
        results_dir / "gemini-2.5-flash__coax-direct-k0__shuffled__n491.csv", index=False)
    # skipped: filename has no "n491"
    pd.DataFrame([{"index": 0, "correct": "True"}]).to_csv(results_dir / "old_run.csv", index=False)
    # skipped: filename is marked superseded even though it has "n491"
    pd.DataFrame([{"index": 0, "correct": "True"}]).to_csv(
        results_dir / "superseded_run__n491.csv", index=False)
    # skipped: matches the glob and "n491" but lacks the required columns
    pd.DataFrame([{"foo": 1}]).to_csv(results_dir / "malformed__n491.csv", index=False)

    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: FakeModel())
    monkeypatch.setattr("sys.argv", base_argv())

    import eval_router
    eval_router.main()

    out = capsys.readouterr().out
    assert "detector answers 0 of them" in out
    rows = (tmp_path / "results" / "detector" / "router_lift.csv").read_text().splitlines()
    # only the one well-formed, non-superseded, n491 run should appear
    assert len(rows) == 2
    assert rows[1].startswith("gemini-2.5-flash__coax-direct-k0__shuffled__n491,")


def test_main_reports_questions_handed_back_when_options_are_not_numeric(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # a wisdom question the router selects, but its options are prose, not numbers, so
    # answer_mcq() can't express a pick and it must be handed back to the VLM
    closed_rows = [
        {"index": 0, "file_name": "img1.png", "question": "How many wisdom teeth are detected?",
         "option1": "None", "option2": "One", "option3": "Two", "option4": "Three", "answer": "A"},
    ]
    pd.DataFrame(closed_rows).to_parquet("closed.parquet")
    pd.DataFrame([{"image_name": "img1.png", "image": b64_png()}]).to_parquet("open.parquet")
    (tmp_path / "detector").mkdir(exist_ok=True)
    (tmp_path / "detector" / "arch_prior.json").write_text(
        json.dumps({"median": {str(k): float(k) for k in range(1, 9)}}))
    (tmp_path / "results" / "closed_ended" / "modelA").mkdir(parents=True)
    pd.DataFrame([{"index": 0, "correct": "True"}]).to_csv(
        "results/closed_ended/modelA/gemini-2.5-flash__coax-direct-k0__shuffled__n491.csv",
        index=False)
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **kw: FakeModel())
    monkeypatch.setattr("sys.argv", base_argv())

    import eval_router
    eval_router.main()

    out = capsys.readouterr().out
    assert "detector answers 0 of them" in out
    assert "(1 handed back to the VLM: no detector reading, or non-numeric options)" in out
