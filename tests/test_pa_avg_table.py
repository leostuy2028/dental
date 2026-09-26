# Tests for paper_analysis/avg_table.py
# This script has no functions - it is plain top-level code. On import it reads
# several results CSVs (relative to the current working directory), computes a
# few accuracy averages, and prints two markdown tables comparing our numbers
# against the paper's published leaderboard rows.

import sys


def _write_csv(path, header, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(str(v) for v in values) + "\n", encoding="utf-8")


def test_prints_both_tables_with_numbers_computed_from_the_fixtures(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    _write_csv(
        tmp_path / "results/closed_ended/knowledge_context"
        / "gemini-3.5-flash__coax-direct-ctx-opgprimer__whole__n491.csv",
        "correct", [1, 1, 1, 0],
    )
    _write_csv(
        tmp_path / "results/closed_ended/knowledge_context"
        / "gemini-3.5-flash__coax-direct-ctx-opgprimer__shuffled__n491.csv",
        "correct", [1, 1, 0, 0],
    )
    _write_csv(tmp_path / "results/open/batched_gemini35_plain578_scores.csv", "score", [1, 1, 1, 0, 0])
    _write_csv(
        tmp_path / "results/closed_ended/gpt-4o-2024-11-20__faithful-direct-k0__whole__n491.csv",
        "correct", [1, 0, 0, 0],
    )
    _write_csv(
        tmp_path / "results/closed_ended/gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv",
        "correct", [1, 1, 0, 0],
    )
    _write_csv(tmp_path / "results/open/batched_gpt4o_matched_scores.csv", "score", [1, 1, 0, 0])
    _write_csv(tmp_path / "results/open/coordarms_gpt4o_cpc_all_scores.csv", "score", [1, 0, 0, 0])

    # avg_table has no main(); importing it runs the whole script and prints.
    sys.modules.pop("avg_table", None)
    import avg_table  # noqa: F401

    out = capsys.readouterr().out

    # first table: our model's row (closed=75.0 from 3/4, open=60.0 from 3/5, avg=67.5)
    assert "| **gemini-3.5-flash (coax + primer)** | 75.0 | 60.0 | **67.5** |" in out
    # paper rows are hard-coded, not computed from our fixtures
    assert "| OralGPT (paper's best-avg model) | 39.60 | 52.77 | 46.19 |" in out
    assert "| GPT-4o (paper) | 45.40 | 37.50 | 41.45 |" in out
    assert "| Claude-3.7-Sonnet (paper) | 41.40 | 40.67 | 41.03 |" in out
    # footnote uses the debiased/"shuffled" closed CSV (closed_bal = 50.0 from 2/4)
    assert "On our debiased balanced key the same config scores 50.0% -> Avg 55.0" in out
    assert "still above OralGPT's 46.2." in out
    assert "§5.4" in out and "§6.1" in out  # section refs survive as literal text

    # second table: GPT-4o decomposition
    assert "| GPT-4o as the paper measured it | 45.4 | 37.5 | 41.5 | -4.7 |" in out
    assert "| GPT-4o, our reproduction of that pipeline | 25.0 | — | — | — |" in out
    assert (
        "**Same model, prompted properly** (no model change) | **50.0** | **50.0** | **50.0** | **+3.8** |"
        in out
    )
    assert "| OralGPT (the purpose-built dental model) | 39.6 | 52.8 | 46.2 | — |" in out
    assert (
        "*(for reference)* a newer generation, same treatment | 75.0 | 60.0 | 67.5 | +21.3 |"
        in out
    )
    assert "+25.0 points on the closed half" in out
    assert "50.0% vs 25.0% for the coordinate-eliciting arm, a paired +25.0" in out
