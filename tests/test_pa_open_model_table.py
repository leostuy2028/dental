# Tests for paper_analysis/open_model_table.py
# This prints one row per model-arm: overall / prose-ref / coord-ref accuracy, read
# straight from committed score CSVs. Gotcha: at IMPORT time it reads
# "eval_open/predictions.parquet" relative to the current directory, so these tests
# chdir into a tmp_path with a tiny fake file there and re-import the module fresh.

import importlib
import sys

import pandas as pd


def load_fresh_module(tmp_path, monkeypatch):
    """chdir into tmp_path (which must already hold eval_open/predictions.parquet)
    and (re)import open_model_table so its module-level RT read picks up our tiny
    fake file instead of the real one."""
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("open_model_table", None)
    module = importlib.import_module("open_model_table")
    return module


def make_predictions(tmp_path):
    eval_dir = tmp_path / "eval_open"
    eval_dir.mkdir()
    pd.DataFrame({
        "index": [1, 2, 3, 4],
        "ref_type": ["prose_ref", "prose_ref", "coord_ref", "coord_ref"],
    }).to_parquet(eval_dir / "predictions.parquet")


def write_scores(tmp_path, name, scores):
    out_dir = tmp_path / "results" / "open"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": [1, 2, 3, 4], "score": scores}).to_csv(out_dir / name, index=False)


def test_import_reads_tiny_predictions_file(tmp_path, monkeypatch):
    make_predictions(tmp_path)
    module = load_fresh_module(tmp_path, monkeypatch)
    try:
        assert list(module.RT.columns) == ["index", "ref_type"]
        assert len(module.RT) == 4
    finally:
        sys.modules.pop("open_model_table", None)


def test_stats_computes_overall_prose_and_coord_means(tmp_path, monkeypatch):
    make_predictions(tmp_path)
    write_scores(tmp_path, "one.csv", [1, 1, 0, 0])
    module = load_fresh_module(tmp_path, monkeypatch)
    try:
        n, overall, prose, coord = module.stats("results/open/one.csv")
        assert n == 4
        assert overall == 50.0
        assert prose == 100.0  # items 1,2 (prose_ref) both scored 1
        assert coord == 0.0    # items 3,4 (coord_ref) both scored 0
    finally:
        sys.modules.pop("open_model_table", None)


def test_main_prints_a_row_per_model_and_pending_for_a_missing_csv(tmp_path, monkeypatch, capsys):
    make_predictions(tmp_path)
    # 4 of the 5 ROWS files present; "coordarms_gpt4o_cpc_all_scores.csv" is left
    # missing on purpose, to exercise the `[PENDING]` fallback branch.
    write_scores(tmp_path, "batched_gpt4o_matched_scores.csv", [1, 1, 0, 0])
    write_scores(tmp_path, "batched_gpt5mini_scores.csv", [1, 0, 1, 0])
    write_scores(tmp_path, "batched_gpt5mini_ex_scores.csv", [0.5, 0.5, 0.5, 0.5])
    write_scores(tmp_path, "batched_gemini35_plain578_scores.csv", [1, 1, 1, 0])

    module = load_fresh_module(tmp_path, monkeypatch)
    try:
        module.main()
        out = capsys.readouterr().out

        assert "| GPT-4o (prompt-matched) | primer, no coordinates | 50.0% | 100.0% | 0.0% |" in out
        assert "| GPT-4o (coordinate-elicited) | primer + coordinate elicitation | `[PENDING]` | | |" in out
        assert "| gpt-5-mini | primer, no coordinates | 50.0% | 50.0% | 50.0% |" in out
        assert "| gpt-5-mini + 12 visual exemplars | primer + exemplars, no coordinates | 50.0% | 50.0% | 50.0% |" in out
        assert "| gemini-3.5-flash | primer, no coordinates (thinking 4000) | 75.0% | 100.0% | 50.0% |" in out
        assert "n=4 items, 477 prose-ref + 101 coord-ref" in out
    finally:
        sys.modules.pop("open_model_table", None)
