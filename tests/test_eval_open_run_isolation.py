# Tests for eval_open/run_isolation.py
# E-open-1: grades each item's prose/coords prediction variants under both
# rubrics with each judge, into a resumable CSV.

import pandas as pd
import pytest

import eval_open.run_isolation as run_isolation


# ---------- load_done ----------

def test_load_done_missing_file_returns_empty(tmp_path):
    done, records = run_isolation.load_done(tmp_path / "nope.csv")
    assert done == set()
    assert records == []


def test_load_done_reads_key_tuples_from_existing_csv(tmp_path):
    path = tmp_path / "out.csv"
    pd.DataFrame([
        {"index": 1, "ref_type": "coord_ref", "category": "c", "judge": "gemini",
         "model": "default", "rubric": "original", "variant": "prose", "score": 0.5, "raw": "r"},
        {"index": 2, "ref_type": "prose_ref", "category": "c", "judge": "claude",
         "model": "default", "rubric": "rephrased", "variant": "coords", "score": 0.1, "raw": "r2"},
    ]).to_csv(path, index=False)

    done, records = run_isolation.load_done(path)

    assert done == {(1, "gemini", "original", "prose"), (2, "claude", "rephrased", "coords")}
    assert len(records) == 2
    assert records[0]["score"] == 0.5


# ---------- sample_items ----------

def _preds(n_coord=3, n_prose=3):
    rows = []
    for i in range(n_coord):
        rows.append({"index": i, "ref_type": "coord_ref", "question": f"cq{i}", "answer": f"ca{i}"})
    for i in range(n_prose):
        rows.append({"index": 100 + i, "ref_type": "prose_ref", "question": f"pq{i}", "answer": f"pa{i}"})
    return pd.DataFrame(rows)


def test_sample_items_with_no_cap_keeps_the_whole_pool():
    preds = _preds(2, 2)
    result = run_isolation.sample_items(preds, coord_n=None, prose_n=None, seed=0)
    assert sorted(result["index"].tolist()) == [0, 1, 100, 101]


def test_sample_items_cap_matches_a_real_pandas_sample_call():
    preds = _preds(5, 5)
    result = run_isolation.sample_items(preds, coord_n=2, prose_n=1, seed=0)

    coord_expected = preds[preds.ref_type == "coord_ref"].sample(2, random_state=0)
    prose_expected = preds[preds.ref_type == "prose_ref"].sample(1, random_state=0)
    expected = pd.concat([coord_expected, prose_expected]).sort_values("index")
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))


def test_sample_items_n_larger_than_pool_selects_the_whole_pool_without_error():
    preds = _preds(3, 2)
    result = run_isolation.sample_items(preds, coord_n=100, prose_n=100, seed=0)
    assert sorted(result["index"].tolist()) == [0, 1, 2, 100, 101]


def test_sample_items_result_is_sorted_by_index():
    preds = _preds(3, 3)
    result = run_isolation.sample_items(preds, coord_n=None, prose_n=None, seed=0)
    assert result["index"].tolist() == sorted(result["index"].tolist())


# ---------- main ----------

def _write_predictions(tmp_path):
    df = pd.DataFrame([
        {"index": 1, "ref_type": "coord_ref", "category": "count", "question": "Q1",
         "answer": "GT1", "pred_prose": "prose-answer-1", "pred_coords": "coord-answer-1"},
        {"index": 2, "ref_type": "prose_ref", "category": "missing", "question": "Q2",
         "answer": "GT2", "pred_prose": "prose-answer-2", "pred_coords": "coord-answer-2"},
    ])
    path = tmp_path / "predictions.parquet"
    df.to_parquet(path)
    return path


def test_main_grades_every_cell_and_writes_final_csv(tmp_path, monkeypatch, capsys):
    pred_path = _write_predictions(tmp_path)
    monkeypatch.setattr(run_isolation, "PRED", str(pred_path))
    out_path = tmp_path / "nested" / "isolation.csv"  # parent dir does not exist yet

    grade_calls = []

    def fake_grade(prompt, judge=None, model=None, delay=None):
        grade_calls.append((judge, model, delay))
        return 0.75, "RAW"

    monkeypatch.setattr(run_isolation, "grade", fake_grade)
    monkeypatch.setattr(
        "sys.argv",
        ["run_isolation.py", "--judges", "gemini", "--coord-sample", "5", "--prose-sample", "5",
         "--out", str(out_path)],
    )

    run_isolation.main()

    # main() creates the output directory itself (unlike regrade.py's hardcoded path)
    assert out_path.exists()
    # 2 items x 1 judge x 2 rubrics x 2 variants = 8 cells
    assert len(grade_calls) == 8
    assert all(j == "gemini" and m is None and d == 0.0 for j, m, d in grade_calls)

    final = pd.read_csv(out_path)
    assert len(final) == 8
    assert set(final["variant"]) == {"prose", "coords"}
    assert set(final["rubric"]) == {"original", "rephrased"}
    assert (final["model"] == "default").all()

    printed = capsys.readouterr().out
    assert "2 items x 1 judges x 2 rubrics x 2 variants = 8 cells (0 already done)" in printed
    assert "done -> " in printed and "(8 rows)" in printed


def test_main_skips_already_done_cells(tmp_path, monkeypatch):
    pred_path = _write_predictions(tmp_path)
    monkeypatch.setattr(run_isolation, "PRED", str(pred_path))
    out_path = tmp_path / "isolation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # pre-mark one of the 8 cells as already done
    pd.DataFrame([
        {"index": 1, "ref_type": "coord_ref", "category": "count", "judge": "gemini",
         "model": "default", "rubric": "original", "variant": "prose", "score": 0.42, "raw": "old"},
    ]).to_csv(out_path, index=False)

    grade_calls = []
    monkeypatch.setattr(run_isolation, "grade",
                        lambda prompt, judge=None, model=None, delay=None: (grade_calls.append(1), (0.9, "R"))[1])
    monkeypatch.setattr(
        "sys.argv",
        ["run_isolation.py", "--judges", "gemini", "--out", str(out_path)],
    )

    run_isolation.main()

    assert len(grade_calls) == 7  # 8 total cells minus the 1 already done
    final = pd.read_csv(out_path)
    assert len(final) == 8
    # the pre-existing row keeps its original score, untouched
    kept = final[(final["index"] == 1) & (final["variant"] == "prose") & (final["rubric"] == "original")]
    assert kept["score"].iloc[0] == 0.42


def test_main_exits_when_gpt4o_requested_without_openai_key(tmp_path, monkeypatch):
    pred_path = _write_predictions(tmp_path)
    monkeypatch.setattr(run_isolation, "PRED", str(pred_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        ["run_isolation.py", "--judges", "gpt-4o,gemini", "--out", str(tmp_path / "o.csv")],
    )

    with pytest.raises(SystemExit):
        run_isolation.main()


def test_main_writes_periodic_checkpoint_every_20_records(tmp_path, monkeypatch, capsys):
    # 6 items x 2 judges x 2 rubrics x 2 variants = 48 cells, so the
    # "every 20 records" checkpoint write triggers partway through the loop
    rows = []
    for i in range(3):
        rows.append({"index": i, "ref_type": "coord_ref", "category": "count", "question": f"cq{i}",
                    "answer": f"ca{i}", "pred_prose": f"pp{i}", "pred_coords": f"pc{i}"})
    for i in range(3):
        rows.append({"index": 100 + i, "ref_type": "prose_ref", "category": "missing", "question": f"pq{i}",
                    "answer": f"pa{i}", "pred_prose": f"pp{i}", "pred_coords": f"pc{i}"})
    pred_path = tmp_path / "predictions.parquet"
    pd.DataFrame(rows).to_parquet(pred_path)
    monkeypatch.setattr(run_isolation, "PRED", str(pred_path))
    out_path = tmp_path / "isolation.csv"

    monkeypatch.setattr(run_isolation, "grade",
                        lambda prompt, judge=None, model=None, delay=None: (0.5, "R"))
    monkeypatch.setattr(
        "sys.argv",
        ["run_isolation.py", "--judges", "gemini,claude", "--out", str(out_path)],
    )

    run_isolation.main()

    final = pd.read_csv(out_path)
    assert len(final) == 48
    printed = capsys.readouterr().out
    assert "[20/48]" in printed  # the periodic checkpoint print fired
