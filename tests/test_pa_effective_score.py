# Tests for paper_analysis/effective_score.py
# For each model, this compares its accuracy on the original vs. the shuffled
# (position-balanced) answer key, computes a "position-robust" accuracy (correct in
# BOTH orderings), and a letter-bias index (how far its predicted letters drift from
# a uniform A/B/C/D spread), then writes a Markdown table + JSON of the numbers.

import json

import pandas as pd

import effective_score as es


def make_csv(path, predicted, correct):
    d = pd.DataFrame({"index": list(range(len(predicted))), "predicted": predicted, "correct": correct})
    d.to_csv(path, index=False)


# ---------- wilson() ----------

def test_wilson_matches_hand_check():
    lo, hi = es.wilson(5, 20)
    assert (lo, hi) == (11.2, 46.9)


# ---------- letter_bias() ----------

def test_letter_bias_uniform_is_zero():
    df = pd.DataFrame({"predicted": ["A", "B", "C", "D"]})
    index, chi2 = es.letter_bias(df)
    assert index == 0.0
    assert chi2 == 0.0


def test_letter_bias_skewed_distribution():
    df = pd.DataFrame({"predicted": ["A", "A", "A", "A", "B", "C", "D", "A"]})
    index, chi2 = es.letter_bias(df)
    assert index == 50.0
    assert chi2 == 6.0


# ---------- main() end to end ----------

def test_main_writes_table_and_json(tmp_path, monkeypatch, capsys):
    ce_dir = tmp_path / "results" / "closed_ended"
    pb_dir = ce_dir / "position_bias"
    pb_dir.mkdir(parents=True)

    orig_path = ce_dir / "orig.csv"
    shuf_path = pb_dir / "shuf.csv"
    make_csv(orig_path, ["A", "B", "A", "C", "D", "A", "B", "A"],
             [True, False, True, True, False, True, False, True])
    make_csv(shuf_path, ["A", "A", "A", "A", "B", "C", "D", "A"],
             [True, True, False, True, False, True, True, False])

    monkeypatch.setattr(es, "REPO", str(tmp_path))
    monkeypatch.setattr(es, "OUT_DIR", str(tmp_path / "_generated"))
    monkeypatch.setattr(es, "MODELS", [
        ("TestModel", "results/closed_ended/orig.csv", "results/closed_ended/position_bias/shuf.csv"),
    ])

    es.main()

    printed = capsys.readouterr().out
    assert "| TestModel | 62.5% | 62.5% | 0.0 | 37.5% [13.7–69.4] | 50% |" in printed
    assert "chi-square vs uniform (shuffled key): {'TestModel': 6.0}" in printed

    table = (tmp_path / "_generated" / "effective_score_table.md").read_text(encoding="utf-8")
    assert "| TestModel | 62.5% | 62.5% | 0.0 | 37.5% [13.7–69.4] | 50% |" in table

    vals = json.loads((tmp_path / "_generated" / "effective_score.values.json").read_text(encoding="utf-8"))
    row = vals["TestModel"]
    assert row["n"] == 8
    assert row["acc_original"] == 62.5
    assert row["acc_shuffled"] == 62.5
    assert row["drop"] == 0.0
    assert row["position_robust_acc"] == 37.5
    assert row["position_robust_ci"] == [13.7, 69.4]
    assert row["letter_bias_index"] == 50
    assert row["chi2_vs_uniform"] == 6.0
    assert row["letter_bias_significant"] is False
    assert vals["_generator"] == "paper_analysis/effective_score.py"
    assert vals["_source_csv"] == [
        "results/closed_ended/orig.csv",
        "results/closed_ended/position_bias/shuf.csv",
    ]


def test_main_flags_significant_bias_with_asterisk(tmp_path, monkeypatch, capsys):
    # a model that almost always picks A on the shuffled key should trip chi2 > 7.82
    ce_dir = tmp_path / "results" / "closed_ended"
    pb_dir = ce_dir / "position_bias"
    pb_dir.mkdir(parents=True)
    orig_path = ce_dir / "orig.csv"
    shuf_path = pb_dir / "shuf.csv"
    make_csv(orig_path, ["A"] * 8, [True] * 8)
    make_csv(shuf_path, ["A", "A", "A", "A", "A", "A", "A", "B"], [True] * 7 + [False])

    monkeypatch.setattr(es, "REPO", str(tmp_path))
    monkeypatch.setattr(es, "OUT_DIR", str(tmp_path / "_generated"))
    monkeypatch.setattr(es, "MODELS", [
        ("BiasedModel", "results/closed_ended/orig.csv", "results/closed_ended/position_bias/shuf.csv"),
    ])

    es.main()
    printed = capsys.readouterr().out
    assert "BiasedModel" in printed
    assert "%\\*" in printed  # significance star present somewhere in the table row
