# Tests for paper_analysis/shuffle_drop.py
# GPT-4o 2x2: original/revised prompt x original/shuffled key. Reads 4 committed run
# CSVs + 2 committed hand-label CSVs (for the "true", hand-read accuracy of the verbose
# original prompt) and writes a table + json into paper_analysis/_generated (redirected
# to tmp_path here).
import json
import os

import pandas as pd
import pytest

import shuffle_drop as sd


# ---------- wilson ----------

def test_wilson_returns_a_range_around_the_point_estimate():
    lo, hi = sd.wilson(50, 100)
    assert lo < 50.0 < hi


def test_wilson_is_narrower_with_more_data():
    lo_small, hi_small = sd.wilson(5, 10)
    lo_big, hi_big = sd.wilson(500, 1000)
    assert (hi_big - lo_big) < (hi_small - lo_small)


# ---------- true_accuracy ----------

def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_true_accuracy_combines_high_conf_reads_and_hand_labels(tmp_path):
    run_csv = tmp_path / "run.csv"
    labels_csv = tmp_path / "labels.csv"
    _write_csv(run_csv, [
        {"index": 0, "raw_response": "The answer is B", "answer": "B"},   # high_conf reads B -> correct
        {"index": 1, "raw_response": "Hmm not sure at all", "answer": "C"},  # ambiguous, needs label
        {"index": 2, "raw_response": "answer is D", "answer": "A"},        # high_conf reads D -> wrong
        {"index": 3, "raw_response": "no clear answer text", "answer": "B"},  # ambiguous, blank label
    ])
    _write_csv(labels_csv, [
        {"index": 1, "true_letter": "C"},
        {"index": 3, "true_letter": ""},
    ])

    correct, n = sd.true_accuracy(str(run_csv), str(labels_csv))

    assert n == 4
    # idx0 (B==B) and idx1 (label C==C) are correct; idx2 (D!=A) and idx3 (blank label) are not
    assert correct == 2


def test_true_accuracy_raises_if_an_ambiguous_reply_has_no_hand_label(tmp_path):
    run_csv = tmp_path / "run.csv"
    labels_csv = tmp_path / "labels.csv"
    _write_csv(run_csv, [{"index": 0, "raw_response": "no clear answer here", "answer": "A"}])
    _write_csv(labels_csv, [{"index": 99, "true_letter": "A"}])  # doesn't cover index 0

    with pytest.raises(SystemExit):
        sd.true_accuracy(str(run_csv), str(labels_csv))


# ---------- main ----------

def test_main_writes_table_and_json(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    os.makedirs(repo / "results", exist_ok=True)
    os.makedirs(repo / "paper_analysis", exist_ok=True)

    original_orig = "results/original_orig.csv"
    original_shuf = "results/original_shuf.csv"
    revised_orig = "results/revised_orig.csv"
    revised_shuf = "results/revised_shuf.csv"
    label_orig = "paper_analysis/hand_labels_orig.csv"
    label_shuf = "paper_analysis/hand_labels_shuf.csv"

    # same content as the true_accuracy unit test above: true=2/4 -> 50.0%
    _write_csv(repo / original_orig, [
        {"index": 0, "raw_response": "The answer is B", "answer": "B", "correct": 1},
        {"index": 1, "raw_response": "Hmm not sure at all", "answer": "C", "correct": 0},
        {"index": 2, "raw_response": "answer is D", "answer": "A", "correct": 0},
        {"index": 3, "raw_response": "no clear answer text", "answer": "B", "correct": 1},
    ])
    _write_csv(repo / label_orig, [
        {"index": 1, "true_letter": "C"},
        {"index": 3, "true_letter": ""},
    ])
    # true=2/4 -> 50.0% again, but via a different mix (idx0 and idx3 correct)
    _write_csv(repo / original_shuf, [
        {"index": 0, "raw_response": "Answer: A", "answer": "A", "correct": 1},
        {"index": 1, "raw_response": "unclear reply text", "answer": "A", "correct": 1},
        {"index": 2, "raw_response": "answer is C", "answer": "B", "correct": 0},
        {"index": 3, "raw_response": "answer is C", "answer": "C", "correct": 0},
    ])
    _write_csv(repo / label_shuf, [
        {"index": 1, "true_letter": "B"},  # mismatches answer "A" -> not correct
    ])
    _write_csv(repo / revised_orig, [
        {"index": 0, "answer": "B", "correct": 1},
        {"index": 1, "answer": "C", "correct": 1},
        {"index": 2, "answer": "A", "correct": 1},
        {"index": 3, "answer": "B", "correct": 0},
    ])
    _write_csv(repo / revised_shuf, [
        {"index": 0, "answer": "A", "correct": 1},
        {"index": 1, "answer": "A", "correct": 1},
        {"index": 2, "answer": "B", "correct": 1},
        {"index": 3, "answer": "C", "correct": 1},
    ])

    out_dir = tmp_path / "generated"
    monkeypatch.setattr(sd, "REPO", str(repo))
    monkeypatch.setattr(sd, "OUT_DIR", str(out_dir))
    monkeypatch.setattr(sd, "CSV", {
        ("original", "orig"): original_orig,
        ("original", "shuf"): original_shuf,
        ("revised", "orig"): revised_orig,
        ("revised", "shuf"): revised_shuf,
    })
    monkeypatch.setattr(sd, "TRUE", {
        "orig": (original_orig, label_orig),
        "shuf": (original_shuf, label_shuf),
    })

    sd.main()

    vals = json.loads((out_dir / "shuffle_drop.values.json").read_text(encoding="utf-8"))
    assert vals["n"] == 4
    assert vals["acc"] == {"original_orig": 50.0, "original_shuf": 50.0,
                           "revised_orig": 75.0, "revised_shuf": 100.0}
    assert vals["true_acc"] == {"orig": 50.0, "shuf": 50.0}
    # floor = the largest single-letter share of the answer key; the loop overwrites
    # floor[key] with whichever CSV.items() entry for that key it processes LAST
    # (revised_orig / revised_shuf here) -- CURRENT BEHAVIOR, not the "original key" CSV.
    assert vals["floor_original_key"] == 50.0   # revised_orig answers B,C,A,B -> B is 2/4
    assert vals["floor_shuffled_key"] == 50.0   # revised_shuf answers A,A,B,C -> A is 2/4
    assert vals["pipeline_gap_original_key"] == 25.0   # 75.0 - 50.0
    assert vals["pipeline_gap_shuffled_key"] == 50.0   # 100.0 - 50.0
    assert vals["true_prompt_gap_original_key"] == 25.0   # 75.0 - true 50.0
    assert vals["true_prompt_gap_shuffled_key"] == 50.0   # 100.0 - true 50.0
    assert vals["parser_cost_original_key"] == 0.0     # true 50.0 - parser 50.0
    assert vals["parser_cost_shuffled_key"] == 0.0
    assert vals["drop_original_prompt_parser"] == 0.0  # 50.0 - 50.0
    assert vals["drop_revised_prompt"] == -25.0        # 75.0 - 100.0
    assert vals["_generator"] == "paper_analysis/shuffle_drop.py"

    table = (out_dir / "shuffle_drop_table.md").read_text(encoding="utf-8")
    assert "50.0%" in table and "75.0%" in table and "100.0%" in table

    out = capsys.readouterr().out
    assert "pipeline gap (revised" in out
    assert "real prompt gap (revised" in out
    assert "parser cost (true" in out
    assert "floor: original key 50.0% -> shuffled key 50.0%" in out
