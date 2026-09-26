# Tests for paper_analysis/overlapping_options.py
# This finds closed-ended items where two of the four options overlap (identical,
# one contains the other, or near-duplicate token sets) so more than one option
# could be defended as correct.

import json

import pandas as pd

import overlapping_options as oo


# ---------- norm() ----------

def test_norm_lowercases_and_strips_punctuation():
    assert oo.norm("Tooth #15, impacted!") == "tooth 15 impacted"


def test_norm_handles_non_string_input():
    assert oo.norm(42) == "42"


# ---------- find() ----------

def make_test_frame():
    return pd.DataFrame({
        "index": [1, 2, 3, 4, 5],
        "question": ["q1", "q2", "q3", "q4", "q5"],
        # row1: option1 and option2 are exactly identical -> "identical"
        # row2: option1's text is a substring of option2's -> "contains"
        # row3: 9-word options sharing 8 of 9 tokens -> jaccard 8/10 = 0.8 -> "near-duplicate"
        # row4: four totally different options -> no overlap
        # row5: option1 is blank -> skipped entirely
        "option1": [
            "Impaction",
            "Tooth 15",
            "mesial distal buccal lingual occlusal apical cervical incisal caries",
            "Caries",
            "",
        ],
        "option2": [
            "Impaction",
            "Tooth 15 is impacted",
            "mesial distal buccal lingual occlusal apical cervical incisal cavity",
            "Impaction",
            "Impaction",
        ],
        "option3": ["Caries", "Caries", "Caries", "Bone loss", "Caries"],
        "option4": ["Bone loss", "Bone loss", "Bone loss", "Missing tooth", "Bone loss"],
        "answer": ["A", "C", "A", "A", "A"],
    })


def test_find_detects_identical_options_and_flags_the_key():
    d = oo.find(make_test_frame())
    row1 = d[d["index"] == 1].iloc[0]
    assert row1.kind == "identical"
    assert row1.opt_i == "A" and row1.opt_j == "B"
    assert row1.key_in_pair == True  # answer is "A", one of the identical pair


def test_find_detects_containment():
    d = oo.find(make_test_frame())
    row2 = d[d["index"] == 2].iloc[0]
    assert row2.kind == "contains"
    assert row2.key_in_pair == False  # answer is "C", not A or B


def test_find_detects_near_duplicates_at_the_jaccard_threshold():
    d = oo.find(make_test_frame())
    row3 = d[d["index"] == 3].iloc[0]
    assert row3.kind == "near-duplicate"


def test_find_skips_rows_with_no_overlap_and_rows_with_a_blank_option():
    d = oo.find(make_test_frame())
    # row4 (no overlap) and row5 (blank option) contribute nothing
    assert set(d["index"]) == {1, 2, 3}
    assert len(d) == 3


# ---------- main() ----------

def test_main_writes_csv_table_and_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    make_test_frame().to_parquet(data_dir / "closed_ended.parquet")

    oo.main()

    out = capsys.readouterr().out
    assert "3 of 5 closed items have two overlapping options (60.0%)" in out
    assert "the answer key is one of the overlapping pair in 2/3" in out

    csv_path = tmp_path / "results" / "closed_ended" / "overlapping_options.csv"
    assert csv_path.exists()
    written = pd.read_csv(csv_path)
    assert len(written) == 3

    table = (tmp_path / "paper_analysis" / "_generated" / "overlapping_options_table.md").read_text(
        encoding="utf-8")
    assert "| **total** | **3** | of which **2** have the key inside the pair |" in table

    vals = json.loads((tmp_path / "paper_analysis" / "_generated" / "overlapping_options.values.json")
                       .read_text(encoding="utf-8"))
    assert vals["n_items"] == 3
    assert vals["n_closed"] == 5
    assert vals["key_in_pair"] == 2
    assert vals["by_kind"] == {"identical": 1, "contains": 1, "near-duplicate": 1}


def test_main_table_skips_kinds_with_zero_rows(tmp_path, monkeypatch):
    # only the "identical" pair is present here, so the "contains" and
    # "near-duplicate" table rows must both be skipped (the `if not len(s): continue`
    # branch in main()).
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = make_test_frame().iloc[[0]]  # just row1 (the identical-options row)
    frame.to_parquet(data_dir / "closed_ended.parquet")

    oo.main()
    table = (tmp_path / "paper_analysis" / "_generated" / "overlapping_options_table.md").read_text(
        encoding="utf-8")
    assert "contains" not in table
    assert "near-duplicate" not in table
    assert "| identical | 1 |" in table
