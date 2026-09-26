# Tests for dataio/export_quality_survey.py
# Renders the question-quality survey as one HTML file per batch, embedding images as
# base64. It has a safety check meant to catch a stray "nan" (the None->NaN CSV round
# trip trap from RESEARCH_PLAN §3.7) before writing the page.

import base64
import io

import pandas as pd
import pytest
from PIL import Image

import dataio.export_quality_survey as m


def test_img_uri_downscales_a_wide_image_to_max_w():
    img = Image.new("RGB", (2000, 100), (10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    uri = m.img_uri(b64)

    assert uri.startswith("data:image/jpeg;base64,")
    decoded = base64.b64decode(uri.split(",", 1)[1])
    out_img = Image.open(io.BytesIO(decoded))
    assert out_img.width == m.MAX_W
    assert out_img.height == 80   # 100 * (1600/2000), same aspect ratio


def _make_repo(tmp_path):
    (tmp_path / "results" / "dentist_audit").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    return tmp_path


def _tiny_jpeg_b64():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), (1, 2, 3)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_main_renders_one_html_file_per_survey_with_closed_and_open_items(tmp_path, monkeypatch, capsys):
    repo = _make_repo(tmp_path)
    b64 = _tiny_jpeg_b64()

    man = pd.DataFrame({
        "image": ["000001", "000002"], "index": [1, 10], "task_type": ["closed", "open"],
        "question": ["What is A?", "Describe B"],
        "A": ["opt A", ""], "B": ["opt B", ""], "C": ["opt C", ""], "D": ["opt D", ""],
        "keyed_answer": ["A", ""], "keyed_text": ["opt A", ""],
        "reference": ["", "ref text"], "risk_rank": [0, 1], "survey": [1, 1],
        "item_id": ["q0001", "q0002"],
    })
    man.to_csv(repo / "results" / "dentist_audit" / "quality_manifest.csv", index=False)
    pd.DataFrame({"file_name": ["000001.jpg"], "image": [b64]}).to_parquet(
        repo / "data" / "closed_ended.parquet", index=False)
    pd.DataFrame({"image_name": ["000002.jpg"], "image": [b64]}).to_parquet(
        repo / "data" / "open_ended.parquet", index=False)

    monkeypatch.setattr(m, "REPO", str(repo))
    monkeypatch.setattr("sys.argv", ["prog"])

    m.main()

    html = (repo / "survey" / "quality_1.html").read_text(encoding="utf-8")
    assert "q0001" in html and "q0002" in html
    assert "opt A" in html
    assert "ref text" in html
    assert "Recorded answer:</b> A) opt A" in html

    printed = capsys.readouterr().out
    assert "survey 1: 2 images, 2 questions" in printed


def test_main_filters_to_requested_survey_batches(tmp_path, monkeypatch, capsys):
    repo = _make_repo(tmp_path)
    b64 = _tiny_jpeg_b64()

    man = pd.DataFrame({
        "image": ["000001", "000002"], "index": [1, 2], "task_type": ["closed", "closed"],
        "question": ["Q1", "Q2"], "A": ["a1", "a2"], "B": ["b1", "b2"], "C": ["c1", "c2"], "D": ["d1", "d2"],
        "keyed_answer": ["A", "B"], "keyed_text": ["a1", "b2"], "reference": ["", ""],
        "risk_rank": [0, 0], "survey": [1, 2], "item_id": ["q0001", "q0002"],
    })
    man.to_csv(repo / "results" / "dentist_audit" / "quality_manifest.csv", index=False)
    pd.DataFrame({"file_name": ["000001.jpg", "000002.jpg"], "image": [b64, b64]}).to_parquet(
        repo / "data" / "closed_ended.parquet", index=False)
    pd.DataFrame({"image_name": [], "image": []}).to_parquet(repo / "data" / "open_ended.parquet", index=False)

    monkeypatch.setattr(m, "REPO", str(repo))
    monkeypatch.setattr("sys.argv", ["prog", "--surveys", "2"])

    m.main()

    survey_dir = repo / "survey"
    assert [p.name for p in survey_dir.iterdir()] == ["quality_2.html"]


def test_main_refuses_to_write_when_a_bare_nan_appears_in_a_reference(tmp_path, monkeypatch):
    # this is the §3.7 guard: a bare "nan" between two tags (e.g. an open reference lost
    # in a CSV round-trip) must stop the write rather than ship a broken page.
    repo = _make_repo(tmp_path)
    b64 = _tiny_jpeg_b64()

    man = pd.DataFrame({
        "image": ["000001"], "index": [1], "task_type": ["open"], "question": ["Q1"],
        "A": [""], "B": [""], "C": [""], "D": [""], "keyed_answer": [""], "keyed_text": [""],
        "reference": ["nan"], "risk_rank": [0], "survey": [1], "item_id": ["q0001"],
    })
    man.to_csv(repo / "results" / "dentist_audit" / "quality_manifest.csv", index=False)
    pd.DataFrame({"file_name": [], "image": []}).to_parquet(repo / "data" / "closed_ended.parquet", index=False)
    pd.DataFrame({"image_name": ["000001.jpg"], "image": [b64]}).to_parquet(
        repo / "data" / "open_ended.parquet", index=False)

    monkeypatch.setattr(m, "REPO", str(repo))
    monkeypatch.setattr("sys.argv", ["prog"])

    with pytest.raises(SystemExit, match="'nan' cells in the rendered page"):
        m.main()


def test_main_does_not_catch_a_nan_multiple_choice_option(tmp_path, monkeypatch):
    # CURRENT BEHAVIOR (looks like a bug): the guard regex is
    # r">\s*nan\s*<|>\s*[A-D]\)\s*nan" but the source file has a stray literal backspace
    # character (0x08) right after "nan" in the second half of that pattern, so the
    # second alternative can never match real HTML. A "nan" landing in a multiple-choice
    # option (rendered as "A) nan") slips through uncaught, even though the same "nan"
    # appearing bare between tags (the open-reference case) is still caught.
    repo = _make_repo(tmp_path)
    b64 = _tiny_jpeg_b64()

    man = pd.DataFrame({
        "image": ["000001"], "index": [1], "task_type": ["closed"], "question": ["Q1"],
        "A": ["nan"], "B": ["b1"], "C": ["c1"], "D": ["d1"],
        "keyed_answer": ["A"], "keyed_text": ["nan"], "reference": [""],
        "risk_rank": [0], "survey": [1], "item_id": ["q0001"],
    })
    man.to_csv(repo / "results" / "dentist_audit" / "quality_manifest.csv", index=False)
    pd.DataFrame({"file_name": ["000001.jpg"], "image": [b64]}).to_parquet(
        repo / "data" / "closed_ended.parquet", index=False)
    pd.DataFrame({"image_name": [], "image": []}).to_parquet(repo / "data" / "open_ended.parquet", index=False)

    monkeypatch.setattr(m, "REPO", str(repo))
    monkeypatch.setattr("sys.argv", ["prog"])

    m.main()   # does NOT raise, even though the page literally contains "A) nan"

    html = (repo / "survey" / "quality_1.html").read_text(encoding="utf-8")
    assert "A) nan" in html
