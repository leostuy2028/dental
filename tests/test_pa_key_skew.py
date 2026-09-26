# Tests for paper_analysis/key_skew.py
# This script reports how often each letter (A/B/C/D) is the correct answer in the
# closed-ended answer key, for the normal key and the position-balanced ("shuffled")
# key, then writes two Markdown tables + a JSON file of the numbers.

import json

import pandas as pd

import key_skew


def make_parquet(path, answers):
    d = pd.DataFrame({"index": list(range(len(answers))), "answer": answers})
    d.to_parquet(path)


# ---------- dist() ----------

def test_dist_counts_and_shares(tmp_path):
    p = tmp_path / "closed.parquet"
    make_parquet(p, ["A", "A", "A", "B", "B", "C", "D", "D"])
    result = key_skew.dist(str(p))
    assert result["n"] == 8
    assert result["count"] == {"A": 3, "B": 2, "C": 1, "D": 2}
    assert result["share"] == {"A": 37.5, "B": 25.0, "C": 12.5, "D": 25.0}
    assert result["always_best_letter"] == "A"
    assert result["always_best_pct"] == 37.5


def test_dist_tie_picks_first_letter_in_ABCD_order(tmp_path):
    # CURRENT BEHAVIOR (looks like a bug, or at least easy to miss): when several
    # letters are tied for the most common answer, max(cnt, key=cnt.get) silently
    # returns the first one seen (A, since counts are built in "ABCD" order), not
    # necessarily the "real" best letter or an explicit tie flag.
    p = tmp_path / "closed.parquet"
    make_parquet(p, ["A", "B", "C", "D", "A", "B", "C", "D"])
    result = key_skew.dist(str(p))
    assert result["always_best_letter"] == "A"
    assert result["always_best_pct"] == 25.0


# ---------- main() end to end ----------

def test_main_writes_tables_and_json(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    make_parquet(data_dir / "closed_ended.parquet", ["A", "A", "A", "B", "B", "C", "D", "D"])
    make_parquet(data_dir / "closed_ended_shuffled.parquet", ["A", "B", "C", "D", "A", "B", "C", "D"])

    out_dir = tmp_path / "_generated"
    monkeypatch.setattr(key_skew, "REPO", str(tmp_path))
    monkeypatch.setattr(key_skew, "OUT_DIR", str(out_dir))

    key_skew.main()

    printed = capsys.readouterr().out
    assert "complete (491): always-A = 37.5%  (shuffled always-A = 25.0%)" in printed
    assert "wrote" in printed

    table = (out_dir / "key_skew_table.md").read_text(encoding="utf-8")
    assert "| A | 3 | 37.5% |" in table
    assert "| C | 1 | 12.5% |" in table

    balance = (out_dir / "key_balance_table.md").read_text(encoding="utf-8")
    assert "| A | 3 (37.5%) | 2 (25.0%) |" in balance
    assert "| **Best always-one-letter score** | **37.5%** | **25.0%** |" in balance

    vals = json.loads((out_dir / "key_skew.values.json").read_text(encoding="utf-8"))
    assert vals["complete"]["count"] == {"A": 3, "B": 2, "C": 1, "D": 2}
    assert vals["shuffled"]["count"] == {"A": 2, "B": 2, "C": 2, "D": 2}
    assert vals["_generator"] == "paper_analysis/key_skew.py"
    assert vals["_note"] == "dataset-derived (answer key), not model outputs"
    assert vals["_source"] == ["data/closed_ended.parquet", "data/closed_ended_shuffled.parquet"]
