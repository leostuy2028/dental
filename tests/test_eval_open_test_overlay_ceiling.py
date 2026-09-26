# Tests for eval_open/test_overlay_ceiling.py
# (an experiment SCRIPT, not a pytest file -- see pyproject.toml's testpaths).
# 4-image ceiling test: plain image vs a hand-placed FDI-numbered overlay image,
# gemini-3.5-flash, graded by GPT-4o. Every model/judge call is faked here.
import base64
import io

import pandas as pd
from PIL import Image

import eval_open.run_batched as rb
import eval_open.test_overlay_ceiling as OC


def _tiny_jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(buf, format="JPEG")
    return buf.getvalue()


def _write_overlay_file(ovdir, image_name):
    ovdir.mkdir(parents=True, exist_ok=True)
    path = ovdir / image_name.replace(".jpg", "_overlay.jpg")
    path.write_bytes(_tiny_jpeg_bytes())
    return path


def _patch_constants(tmp_path, monkeypatch, images, data_path):
    ovdir = tmp_path / "overlay_ceiling_5img"
    for im in images:
        _write_overlay_file(ovdir, im)
    primer_path = tmp_path / "opg_primer.txt"
    primer_path.write_text("tiny primer", encoding="utf-8")
    out_path = tmp_path / "results" / "open" / "overlay_needle.csv"

    monkeypatch.setattr(OC, "DATA", str(data_path))
    monkeypatch.setattr(OC, "PRIMER", str(primer_path))
    monkeypatch.setattr(OC, "OVDIR", str(ovdir))
    monkeypatch.setattr(OC, "OUT", str(out_path))
    monkeypatch.setattr(OC, "IMAGES", images)
    return out_path


def _fake_answer_image(b64, questions, primer, system, provider, model, detection_text=None):
    tag = "plain" if detection_text is None else "ov"
    return [f"{tag}_ans_{i}" for i in range(len(questions))], f"{tag}_raw"


def _fake_grade(prompt, judge=None):
    if "plain_ans" in prompt:
        return 0.3, "raw"
    if "ov_ans" in prompt:
        return 0.9, "raw"
    raise AssertionError(f"unexpected grading prompt: {prompt[:80]}")


def test_main_writes_csv_with_expected_columns_and_prints_all_subsets(tmp_path, monkeypatch, capsys):
    rows = [
        {"index": 0, "image_name": "aaa111.jpg", "category": "Teeth Findings",
         "question": "Which tooth has a filling?", "answer": "#14", "image": "B64_PLAIN_A"},
        {"index": 1, "image_name": "aaa111.jpg", "category": "General",
         "question": "What is the overall condition?", "answer": "fine", "image": "B64_PLAIN_A"},
        {"index": 2, "image_name": "bbb222.jpg", "category": "Teeth Findings",
         "question": "How many teeth are visible?", "answer": "30", "image": "B64_PLAIN_B"},
    ]
    data_path = tmp_path / "open_ended.parquet"
    pd.DataFrame(rows).to_parquet(data_path)
    images = ["aaa111.jpg", "bbb222.jpg"]
    out_path = _patch_constants(tmp_path, monkeypatch, images, data_path)

    monkeypatch.setattr(rb, "answer_image", _fake_answer_image)
    monkeypatch.setattr(OC, "grade", _fake_grade)

    OC.main()

    d = pd.read_csv(out_path)
    assert list(d.columns) == ["img", "teeth", "isloc", "plain", "ov"]
    assert len(d) == 3
    assert d["img"].tolist() == ["aaa111", "aaa111", "bbb222"]   # img[:6], no extension left
    assert d["teeth"].tolist() == [True, False, True]
    assert d["isloc"].tolist() == [True, False, False]
    assert (d["plain"] == 0.3).all() and (d["ov"] == 0.9).all()

    out = capsys.readouterr().out
    assert "gemini-3.5-flash, plain vs accurate overlay (4 images):" in out
    assert "ALL              n= 3   30.0% ->  90.0%  (+60.0)" in out
    assert "TEETH            n= 2   30.0% ->  90.0%  (+60.0)" in out
    assert "LOCALIZE         n= 1   30.0% ->  90.0%  (+60.0)" in out
    assert "non-teeth        n= 1   30.0% ->  90.0%  (+60.0)" in out
    assert "aaa111: 30% -> 90%  (n=2)" in out
    assert "bbb222: 30% -> 90%  (n=1)" in out
    assert "wrote " in out and "overlay_needle.csv" in out
    assert "(nondeterministic; exact %s vary run to run)" in out


def test_main_skips_printing_empty_subset_instead_of_crashing(tmp_path, monkeypatch, capsys):
    # Only one row, and it IS a teeth question, so d[~d.teeth] ("non-teeth") is an
    # empty DataFrame. s() guards with `if len(sub):`, so it should just skip the
    # print rather than crash on an empty-slice mean.
    rows = [
        {"index": 0, "image_name": "ccc333.jpg", "category": "Teeth Findings",
         "question": "Which tooth has a filling?", "answer": "#14", "image": "B64_PLAIN_C"},
    ]
    data_path = tmp_path / "open_ended.parquet"
    pd.DataFrame(rows).to_parquet(data_path)
    images = ["ccc333.jpg"]
    out_path = _patch_constants(tmp_path, monkeypatch, images, data_path)

    monkeypatch.setattr(rb, "answer_image", _fake_answer_image)
    monkeypatch.setattr(OC, "grade", _fake_grade)

    OC.main()

    d = pd.read_csv(out_path)
    assert len(d) == 1
    assert d["teeth"].tolist() == [True]

    out = capsys.readouterr().out
    assert "ALL              n= 1   30.0% ->  90.0%  (+60.0)" in out
    assert "TEETH            n= 1   30.0% ->  90.0%  (+60.0)" in out
    assert "LOCALIZE         n= 1   30.0% ->  90.0%  (+60.0)" in out
    assert "non-teeth" not in out   # empty subset -> s() silently skips the print
