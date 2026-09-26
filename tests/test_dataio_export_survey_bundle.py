# Tests for dataio/export_survey_bundle.py
# Builds the static dentist-survey bundle: one jpg per item plus a survey_data.json.
# The closed answer key is embedded but never shown to the viewer in the blind phase.

import base64
import json

import pandas as pd

import dataio.export_survey_bundle as m


def test_decode_jpeg_strips_a_data_uri_prefix_if_present():
    raw = base64.b64encode(b"hello").decode()
    assert m.decode_jpeg("data:image/jpeg;base64," + raw) == b"hello"
    assert m.decode_jpeg(raw) == b"hello"


def _write_inputs(tmp_path, img_b64):
    man_p = tmp_path / "manifest.csv"
    closed_p = tmp_path / "closed.parquet"
    open_p = tmp_path / "open.parquet"

    man = pd.DataFrame({
        "item_id": ["q1", "q2"], "survey_order": [2, 1],
        "task_type": ["closed", "open"], "index": [10, 20],
    })
    man.to_csv(man_p, index=False)

    closed = pd.DataFrame({
        "index": [10], "question": ["What?"], "image": [img_b64],
        "option1": ["A opt"], "option2": ["B opt"], "option3": ["C opt"], "option4": ["D opt"],
        "answer": ["B"],
    })
    closed.to_parquet(closed_p, index=False)

    op = pd.DataFrame({"index": [20], "question": ["Describe."], "image": [img_b64], "answer": ["some text ref"]})
    op.to_parquet(open_p, index=False)
    return man_p, closed_p, open_p


def test_main_writes_images_and_survey_data_json(tmp_path, monkeypatch, capsys):
    img_b64 = base64.b64encode(b"fakejpegbytes").decode()
    man_p, closed_p, open_p = _write_inputs(tmp_path, img_b64)
    out_dir = tmp_path / "survey"

    monkeypatch.setattr(m, "MANIFEST", str(man_p))
    monkeypatch.setattr(m, "CLOSED", str(closed_p))
    monkeypatch.setattr(m, "OPEN", str(open_p))
    monkeypatch.setattr(m, "OUT_DIR", str(out_dir))

    m.main()

    assert (out_dir / "images" / "q1.jpg").read_bytes() == b"fakejpegbytes"
    assert (out_dir / "images" / "q2.jpg").read_bytes() == b"fakejpegbytes"

    data = json.loads((out_dir / "survey_data.json").read_text(encoding="utf-8"))
    assert data["n_closed"] == 1 and data["n_open"] == 1
    items = {it["item_id"]: it for it in data["items"]}
    assert items["q1"]["options"] == {"A": "A opt", "B": "B opt", "C": "C opt", "D": "D opt"}
    assert items["q1"]["_key"] == "B"
    assert items["q2"]["reference"] == "some text ref"
    # items are sorted by survey_order, so q2 (order 1) comes before q1 (order 2)
    assert [it["item_id"] for it in data["items"]] == ["q2", "q1"]

    printed = capsys.readouterr().out
    assert "2 items" in printed and "2 images" in printed


def test_main_removes_stale_jpgs_from_a_previous_run(tmp_path, monkeypatch):
    img_b64 = base64.b64encode(b"fakejpegbytes").decode()
    man_p, closed_p, open_p = _write_inputs(tmp_path, img_b64)
    out_dir = tmp_path / "survey"

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True)
    stale = img_dir / "stale.jpg"
    stale.write_text("leftover from an earlier selection")

    monkeypatch.setattr(m, "MANIFEST", str(man_p))
    monkeypatch.setattr(m, "CLOSED", str(closed_p))
    monkeypatch.setattr(m, "OPEN", str(open_p))
    monkeypatch.setattr(m, "OUT_DIR", str(out_dir))

    m.main()

    assert not stale.exists()
    assert sorted(p.name for p in img_dir.iterdir()) == ["q1.jpg", "q2.jpg"]
