# Tests for paper_analysis/position_stability.py
# This script pairs a model's answers on the plain 491 vs the shuffled-key 491 and
# buckets every item into content-stable (same tooth picked), letter-stable (same
# letter, now a different tooth), or neither. It writes a table + json into
# paper_analysis/_generated (we redirect that to tmp_path so nothing real is touched).
import json
import os

import pandas as pd
import pytest

import position_stability as ps


# ---------- _is_failed ----------

def test_is_failed_detects_api_failure_sentinel():
    assert ps._is_failed("max retries exceeded") == True


def test_is_failed_detects_empty_and_nan_text():
    assert ps._is_failed("") == True
    assert ps._is_failed("nan") == True
    assert ps._is_failed(float("nan")) == True  # not a string at all


def test_is_failed_false_for_a_real_reply():
    assert ps._is_failed("The answer is B") == False


# ---------- classify ----------

def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_classify_buckets_content_letter_neither_and_excludes_failed(tmp_path):
    clean_csv = tmp_path / "clean.csv"
    shuf_csv = tmp_path / "shuf.csv"
    # idx0: same TOOTH text picked both times (option1 clean == option2 shuffled) -> content
    # idx1: same LETTER both times, but the letter now points at a different tooth -> letter
    # idx2: different letter AND different tooth -> neither
    # idx3: shuffled reply is empty (failed) -> excluded from scoring entirely
    # idx4: predicted letter is missing/unparseable in one run -> also lands in "neither"
    # without ever looking up option text (isinstance/`in L2OPT` guard).
    _write_csv(clean_csv, [
        {"index": 0, "predicted": "A", "raw_response": "The answer is A", "correct": 1},
        {"index": 1, "predicted": "C", "raw_response": "Answer: C", "correct": 1},
        {"index": 2, "predicted": "A", "raw_response": "Some reasoning A", "correct": 0},
        {"index": 3, "predicted": "B", "raw_response": "Some reasoning B", "correct": 1},
        {"index": 4, "predicted": "", "raw_response": "I refuse to answer", "correct": 0},
    ])
    _write_csv(shuf_csv, [
        {"index": 0, "predicted": "B", "raw_response": "Reasoning B", "correct": 0},
        {"index": 1, "predicted": "C", "raw_response": "Answer: C", "correct": 1},
        {"index": 2, "predicted": "D", "raw_response": "Reasoning D", "correct": 0},
        {"index": 3, "predicted": "B", "raw_response": "", "correct": 1},
        {"index": 4, "predicted": "A", "raw_response": "Answer: A", "correct": 0},
    ])
    clean_opts = pd.DataFrame({
        "index": [0, 1, 2, 3, 4],
        "option1": ["tooth1", "n/a", "toothA", "n/a", "n/a"],
        "option2": ["n/a", "n/a", "n/a", "n/a", "n/a"],
        "option3": ["n/a", "toothX", "n/a", "n/a", "n/a"],
        "option4": ["n/a", "n/a", "n/a", "n/a", "n/a"],
    }).set_index("index")
    shuf_opts = pd.DataFrame({
        "index": [0, 1, 2, 3, 4],
        "option1": ["n/a", "n/a", "n/a", "n/a", "n/a"],
        "option2": ["tooth1", "n/a", "n/a", "n/a", "n/a"],
        "option3": ["n/a", "toothY", "n/a", "n/a", "n/a"],
        "option4": ["n/a", "n/a", "toothZ", "n/a", "n/a"],
    }).set_index("index")

    result = ps.classify(str(clean_csv), str(shuf_csv), clean_opts, shuf_opts)

    assert result["n"] == 4           # idx3 excluded (empty shuffled reply)
    assert result["excluded_failed"] == 1
    assert result["count"] == {"content": 1, "letter": 1, "neither": 2}
    assert result["pct"] == {"content": 25.0, "letter": 25.0, "neither": 50.0}
    # clean correct at idx 0,1,2,4 = 1,1,0,0 -> 2/4;  shuffled correct at idx 0,1,2,4 = 0,1,0,0 -> 1/4
    assert result["acc_clean"] == 50.0
    assert result["acc_shuffled"] == 25.0
    assert result["acc_drop"] == 25.0


def test_classify_prints_a_message_when_items_are_excluded_but_crashes_if_all_are(tmp_path, capsys):
    # CURRENT BEHAVIOR (looks like a bug): if EVERY item in the pair is excluded
    # (failed/empty reply), classify() still tries to build pct = 100*count/n with n==0
    # and raises ZeroDivisionError instead of returning a "no valid items" result.
    clean_csv = tmp_path / "clean.csv"
    shuf_csv = tmp_path / "shuf.csv"
    _write_csv(clean_csv, [{"index": 0, "predicted": "A", "raw_response": "A", "correct": 1}])
    _write_csv(shuf_csv, [{"index": 0, "predicted": "A", "raw_response": "max retries exceeded", "correct": 0}])
    opts = pd.DataFrame({"index": [0], "option1": ["x"], "option2": ["x"],
                          "option3": ["x"], "option4": ["x"]}).set_index("index")
    with pytest.raises(ZeroDivisionError):
        ps.classify(str(clean_csv), str(shuf_csv), opts, opts)
    out = capsys.readouterr().out
    assert "excluded 1 item(s)" in out


# ---------- main ----------

def _make_data_parquets(repo_dir):
    os.makedirs(os.path.join(repo_dir, "data"), exist_ok=True)
    opts = pd.DataFrame({
        "index": [0, 1, 2],
        "option1": ["a1", "b1", "c1"], "option2": ["a2", "b2", "c2"],
        "option3": ["a3", "b3", "c3"], "option4": ["a4", "b4", "c4"],
    })
    opts.to_parquet(os.path.join(repo_dir, "data", "closed_ended.parquet"))
    shuf = pd.DataFrame({
        "index": [0, 1, 2],
        "option1": ["a1", "b1", "c1"], "option2": ["a2", "b2", "c2"],
        "option3": ["a3", "b3", "c3"], "option4": ["a4", "b4", "c4"],
    })
    shuf.to_parquet(os.path.join(repo_dir, "data", "closed_ended_shuffled.parquet"))


def test_main_writes_table_and_json_and_prints_summary(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    os.makedirs(repo, exist_ok=True)
    _make_data_parquets(str(repo))

    clean_rel = "results/model_a_whole.csv"
    shuf_rel = "results/model_a_shuffled.csv"
    os.makedirs(repo / "results", exist_ok=True)
    _write_csv(repo / clean_rel, [
        {"index": 0, "predicted": "A", "raw_response": "answer is A", "correct": 1},
        {"index": 1, "predicted": "B", "raw_response": "answer is B", "correct": 1},
        {"index": 2, "predicted": "C", "raw_response": "answer is C", "correct": 0},
    ])
    _write_csv(repo / shuf_rel, [
        {"index": 0, "predicted": "A", "raw_response": "answer is A", "correct": 1},
        {"index": 1, "predicted": "B", "raw_response": "answer is B", "correct": 1},
        {"index": 2, "predicted": "D", "raw_response": "answer is D", "correct": 0},
    ])

    out_dir = tmp_path / "generated"
    monkeypatch.setattr(ps, "REPO", str(repo))
    monkeypatch.setattr(ps, "OUT_DIR", str(out_dir))
    monkeypatch.setattr(ps, "MODELS", {
        "Model-A": (clean_rel, shuf_rel),
        "Model-B (missing)": (clean_rel, "results/does_not_exist.csv"),
    })
    monkeypatch.setattr("sys.argv", ["position_stability.py"])

    ps.main()

    out = capsys.readouterr().out
    assert "skip Model-B (missing)" in out
    assert "Model-A" in out
    assert os.path.exists(out_dir / "position_stability_table.md")
    assert os.path.exists(out_dir / "position_stability.values.json")

    table = (out_dir / "position_stability_table.md").read_text(encoding="utf-8")
    assert "Model-A" in table
    assert "Model-B" not in table  # skipped model never gets a row

    vals = json.loads((out_dir / "position_stability.values.json").read_text(encoding="utf-8"))
    assert vals["_generator"] == "paper_analysis/position_stability.py"
    assert "Model-A" in vals
    assert vals["Model-A"]["n"] == 3


def test_main_with_compare_prompts_flag_uses_the_prompt_compare_dict(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    os.makedirs(repo, exist_ok=True)
    _make_data_parquets(str(repo))
    clean_rel = "results/prompt_a.csv"
    shuf_rel = "results/prompt_a_shuf.csv"
    os.makedirs(repo / "results", exist_ok=True)
    _write_csv(repo / clean_rel, [{"index": 0, "predicted": "A", "raw_response": "answer is A", "correct": 1}])
    _write_csv(repo / shuf_rel, [{"index": 0, "predicted": "A", "raw_response": "answer is A", "correct": 1}])

    out_dir = tmp_path / "generated2"
    monkeypatch.setattr(ps, "REPO", str(repo))
    monkeypatch.setattr(ps, "OUT_DIR", str(out_dir))
    monkeypatch.setattr(ps, "PROMPT_COMPARE", {"GPT-4o (revised)": (clean_rel, shuf_rel)})
    monkeypatch.setattr("sys.argv", ["position_stability.py", "--compare-prompts"])

    ps.main()

    assert os.path.exists(out_dir / "prompt_compare_table.md")
    assert os.path.exists(out_dir / "prompt_compare.values.json")
