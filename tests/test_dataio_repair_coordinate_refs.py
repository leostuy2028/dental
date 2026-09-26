# Tests for dataio/repair_coordinate_refs.py
# Turns a JSON list of bounding boxes (the released "reference answer" for some open
# questions) into a plain sentence, dropping only the pixel coordinates.

import json

import pandas as pd

import dataio.repair_coordinate_refs as m


def test_findings_collects_tooth_and_label_pairs():
    node = [
        {"box_2d": [1, 2, 3, 4], "tooth_id": "36", "label": "Crown"},
        {"box_2d": [5, 6, 7, 8], "tooth_id": "38", "label": "Filling"},
    ]
    assert m.findings(node) == [("36", "Crown"), ("38", "Filling")]


def test_findings_reads_nested_shape_where_the_key_is_the_finding():
    node = {"Crown": {"box_2d": [1, 2, 3, 4]}, "tooth_id": "11"}
    assert m.findings(node) == [("11", "Crown")]


def test_findings_reads_true_boolean_labels_but_skips_false_ones():
    assert m.findings({"is_impacted": "true", "tooth_id": "48"}) == [("48", "impacted")]
    assert m.findings({"is_impacted": "false", "tooth_id": "48"}) == []


def test_findings_falls_back_to_key_value_text():
    assert m.findings({"tooth_id": "21", "custom_key": "somevalue"}) == [("21", "custom_key somevalue")]


def test_to_sentence_groups_by_tooth_and_appends_loose_findings():
    pairs = [("36", "crown"), ("36", "filling"), (None, "loose finding")]
    assert m.to_sentence(pairs) == "Tooth #36: crown, filling. Also noted: loose finding."


def test_to_sentence_reports_bare_tooth_numbers_with_no_finding():
    assert m.to_sentence([], teeth_seen=["11", "12"]) == "Tooth #11, #12."


def test_to_sentence_with_only_loose_findings_uses_findings_prefix():
    pairs = [(None, "loose1"), (None, "loose2")]
    assert m.to_sentence(pairs) == "Findings: loose1, loose2."


def test_teeth_in_collects_every_tooth_id_including_nested():
    node = [{"box_2d": [1, 2, 3, 4], "tooth_id": "36"},
            {"tooth_id": "38", "nested": {"tooth_id": "40"}}]
    assert m.teeth_in(node) == ["36", "38", "40"]


def test_repair_returns_a_sentence_for_valid_json():
    ref = json.dumps([{"box_2d": [1, 2, 3, 4], "tooth_id": "36", "label": "Crown"}])
    assert m.repair(ref) == "Tooth #36: crown."


def test_repair_returns_none_for_unparsable_json():
    assert m.repair("not json{") is None


def test_repair_returns_none_when_json_parses_but_has_no_findings():
    assert m.repair(json.dumps({"box_2d": [1, 2, 3, 4]})) is None


def test_main_writes_repaired_rows_and_reports_failures(tmp_path, monkeypatch, capsys):
    data_path = tmp_path / "open.parquet"
    manifest_path = tmp_path / "manifest.csv"
    out_path = tmp_path / "sub" / "out.csv"   # nested dir does not exist yet

    op = pd.DataFrame({
        "index": [1, 2, 3],
        "question": ["Q1", "Q2", "Q3"],
        "answer": [
            json.dumps([{"box_2d": [1, 2, 3, 4], "tooth_id": "36", "label": "Crown"}]),
            "not json{",
            json.dumps({"box_2d": [1, 2, 3, 4]}),
        ],
    })
    op.to_parquet(data_path, index=False)
    man = pd.DataFrame({"index": [1, 2, 3], "disposition": ["REPAIR", "REPAIR", "REPAIR"]})
    man.to_csv(manifest_path, index=False)

    monkeypatch.setattr(m, "DATA", str(data_path))
    monkeypatch.setattr(m, "MANIFEST", str(manifest_path))
    monkeypatch.setattr(m, "OUT", str(out_path))

    m.main()

    out = pd.read_csv(out_path)
    assert out["index"].tolist() == [1]
    assert out["repaired_reference"].tolist() == ["Tooth #36: crown."]

    printed = capsys.readouterr().out
    assert "REPAIR items: 3   repaired: 1   no recoverable content: 2" in printed
    assert "indices with nothing to recover: [2, 3]" in printed
    assert f"wrote {m.OUT}" in printed
