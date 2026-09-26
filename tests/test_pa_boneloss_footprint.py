# Tests for paper_analysis/boneloss_footprint.py
# This script counts, in the closed and open datasets, how many items even mention
# bone loss and how many are KEYED as "no bone loss" (the at-risk footprint for the
# dentist-confirmed key weakness). It writes a values.json and prints a one-line
# summary per half.

import json

import pandas as pd

import boneloss_footprint as m


def _make_data(tmp_path, open_rows, closed_rows):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(open_rows).to_parquet(data_dir / "open_ended.parquet")
    pd.DataFrame(closed_rows).to_parquet(data_dir / "closed_ended.parquet")


def test_main_counts_bone_mentions_and_keyed_no_loss(tmp_path, monkeypatch, capsys):
    open_rows = [
        {"question": "Describe bone loss in this X-ray",
         "answer": "no apparent bone loss noted overall"},
        {"question": "What teeth are visible?",
         "answer": "There is alveolar bone resorption present."},
        {"question": "Describe overall dental status",
         "answer": "Teeth appear healthy with no significant bone loss detected."},
        {"question": "How many teeth are present?",
         "answer": "32 teeth total, all present."},
        {"question": "Any periodontal disease evident?",
         "answer": "Mild periodontal disease with bone loss around molars."},
    ]
    closed_rows = [
        {"question": "Which best describes the bone architecture in the mandible?",
         "option1": "No bone loss", "option2": "Mild bone loss",
         "option3": "Moderate bone loss", "option4": "Severe bone loss", "answer": "A"},
        {"question": "What is the diagnosis for tooth #14?",
         "option1": "Caries", "option2": "Periodontal abscess",
         "option3": "Impaction", "option4": "Normal", "answer": "D"},
        {"question": "Count the visible teeth",
         "option1": "28", "option2": "30", "option3": "32", "option4": "24", "answer": "C"},
        {"question": "Evaluate alveolar bone resorption severity",
         "option1": "None", "option2": "Mild", "option3": "Moderate", "option4": "Severe", "answer": "A"},
        {"question": "Which tooth shows no bone loss on X-ray?",
         "option1": "21", "option2": "14", "option3": "30", "option4": "8", "answer": "B"},
    ]
    _make_data(tmp_path, open_rows, closed_rows)

    monkeypatch.setattr(m, "REPO", str(tmp_path))
    out_dir = tmp_path / "_generated"
    monkeypatch.setattr(m, "OUT_DIR", str(out_dir))

    m.main()

    out = capsys.readouterr().out
    assert "OPEN  (n=5): bone-mentioned 4, keyed 'no bone loss' 2 (40.0%)" in out
    assert "CLOSED(n=5): bone-mentioned 4, keyed 'no bone loss' 1 (20.0%)" in out

    vals = json.loads((out_dir / "boneloss_footprint.values.json").read_text(encoding="utf-8"))
    assert vals["open"] == {
        "n": 5, "bone_mentioned": 4, "keyed_no_bone_loss": 2, "keyed_no_bone_loss_pct": 40.0,
    }
    assert vals["closed"] == {
        "n": 5, "bone_mentioned": 4, "keyed_no_bone_loss": 1, "keyed_no_bone_loss_pct": 20.0,
    }
    assert vals["_generator"] == "paper_analysis/boneloss_footprint.py"


def test_closed_noloss_short_circuits_to_zero_when_no_bone_items(tmp_path, monkeypatch, capsys):
    # when nothing in the closed half mentions bone, "bone.sum() and ...str.contains(...)"
    # short-circuits on the falsy 0 instead of calling .str.contains on an empty frame.
    open_rows = [
        {"question": "How many teeth are present?", "answer": "32 teeth, all healthy."},
        {"question": "What is the diagnosis?", "answer": "Mild caries on the upper left first molar."},
    ]
    closed_rows = [
        {"question": "How many teeth total?", "option1": "28", "option2": "30",
         "option3": "32", "option4": "24", "answer": "C"},
    ]
    _make_data(tmp_path, open_rows, closed_rows)
    monkeypatch.setattr(m, "REPO", str(tmp_path))
    out_dir = tmp_path / "_generated"
    monkeypatch.setattr(m, "OUT_DIR", str(out_dir))

    m.main()

    out = capsys.readouterr().out
    assert "OPEN  (n=2): bone-mentioned 0, keyed 'no bone loss' 0 (0.0%)" in out
    assert "CLOSED(n=1): bone-mentioned 0, keyed 'no bone loss' 0 (0.0%)" in out
    vals = json.loads((out_dir / "boneloss_footprint.values.json").read_text(encoding="utf-8"))
    assert vals["closed"]["keyed_no_bone_loss"] == 0
