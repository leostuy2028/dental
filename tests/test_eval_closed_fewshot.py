# Tests for eval_closed_fewshot.py
# The few-shot closed-ended MCQ harness for Claude: same-category examples are
# sampled from a held-out "pool" slice of the dataset, with an assertion that no
# example ever shares an index with a test question (contamination guard).

import os

import pandas as pd
import pytest

import eval_closed_fewshot as fewshot
import dataio.eval_data as eval_data
from clients.errors import APICallFailed


def make_pool():
    return pd.DataFrame({
        "index": [0, 1, 2, 3],
        "category": ["Teeth", "Teeth", "Patho", "Patho"],
        "question": ["q0", "q1", "q2", "q3"],
    })


def fake_build_prompt(row, examples=None, cot=False, mode="house"):
    return "system", [{"role": "user", "content": row["question"], "n_ex": len(examples or [])}]


# ---------- get_examples ----------

def test_get_examples_returns_nothing_when_k_is_zero():
    assert fewshot.get_examples(make_pool(), {"index": 99, "category": "Teeth"}, k=0) == []


def test_get_examples_samples_from_the_same_category():
    examples = fewshot.get_examples(make_pool(), {"index": 99, "category": "Teeth"}, k=1, seed=42)
    assert len(examples) == 1
    assert examples[0]["category"] == "Teeth"


def test_get_examples_caps_at_the_available_count():
    examples = fewshot.get_examples(make_pool(), {"index": 99, "category": "Patho"}, k=10, seed=1)
    assert len(examples) == 2  # only 2 Patho rows exist in the pool


def test_get_examples_returns_nothing_for_an_unseen_category():
    assert fewshot.get_examples(make_pool(), {"index": 99, "category": "Jaw"}, k=2, seed=1) == []


def test_get_examples_refuses_to_sample_the_test_row_itself():
    with pytest.raises(AssertionError, match="DATA CONTAMINATION"):
        fewshot.get_examples(make_pool(), {"index": 0, "category": "Teeth"}, k=1, seed=1)


# ---------- prove_no_contamination ----------

def test_prove_no_contamination_passes_on_disjoint_sets(capsys):
    pool = pd.DataFrame({"index": [0, 1, 2]})
    test = pd.DataFrame({"index": [3, 4, 5]})
    fewshot.prove_no_contamination(pool, test)
    out = capsys.readouterr().out
    assert "Result        : CLEAN" in out


def test_prove_no_contamination_raises_on_overlap():
    pool = pd.DataFrame({"index": [0, 1, 2]})
    test = pd.DataFrame({"index": [0, 4]})
    with pytest.raises(AssertionError, match="CONTAMINATION DETECTED"):
        fewshot.prove_no_contamination(pool, test)


# ---------- print_summary ----------

def test_print_summary_reports_the_real_accuracy_and_letter_counts(capsys):
    df = pd.DataFrame({"category": ["Teeth", "Teeth", "Patho"], "correct": [True, False, True],
                       "predicted": ["A", "B", None], "refused": [False, True, False]})
    fewshot.print_summary(df, "claude-haiku", 3)
    out = capsys.readouterr().out
    expected_acc = df["correct"].mean() * 100
    assert f"{expected_acc:.2f}%" in out
    assert "{'A': 1, 'B': 1, 'C': 0, 'D': 0}" in out
    assert "Refusals :" in out


def test_print_summary_without_a_refused_column_skips_the_refusal_line(capsys):
    df = pd.DataFrame({"category": ["Teeth"], "correct": [True], "predicted": ["A"]})
    fewshot.print_summary(df, "claude-haiku", 0)
    assert "Refusals" not in capsys.readouterr().out


# ---------- run(): end to end with fakes ----------

def make_dataset(n, categories=None):
    categories = categories or ["Teeth"] * n
    return pd.DataFrame({
        "index": list(range(n)),
        "file_name": [f"img{i}.jpg" for i in range(n)],
        "category": categories,
        "question": [f"Q{i}?" for i in range(n)],
        "answer": ["A" if i % 2 == 0 else "B" for i in range(n)],
    })


def test_run_zero_shot_scores_the_whole_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(3)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(fewshot, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(fewshot, "call", lambda system, messages, model=None, cot=False: ("A", "raw"))
    monkeypatch.setattr(fewshot, "looks_like_refusal", lambda raw: False)

    fewshot.run(model="claude-haiku", k=0, results_path="results/out.csv")

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [0, 1, 2]
    assert list(out_df["n_examples"]) == [0, 0, 0]  # k=0 -> no examples requested
    assert os.path.exists("results/out.csv.meta.json")


def test_run_k_shot_reserves_the_first_pool_size_rows_as_the_exemplar_pool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # more rows than POOL_SIZE so the pool/test split actually has both sides
    n = fewshot.POOL_SIZE + 2
    df = make_dataset(n)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(fewshot, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(fewshot, "call", lambda system, messages, model=None, cot=False: ("A", "raw"))
    monkeypatch.setattr(fewshot, "looks_like_refusal", lambda raw: False)

    fewshot.run(model="claude-haiku", k=1, results_path="results/out.csv")

    out_df = pd.read_csv("results/out.csv")
    # only the 2 rows after the pool are ever tested
    assert list(out_df["index"]) == [fewshot.POOL_SIZE, fewshot.POOL_SIZE + 1]
    assert (out_df["n_examples"] == 1).all()


def test_run_resumes_and_skips_already_done_indices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(3)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(fewshot, "build_prompt", fake_build_prompt)
    calls = []
    monkeypatch.setattr(fewshot, "call", lambda system, messages, model=None, cot=False:
                        (calls.append(messages) or "A", "raw"))
    monkeypatch.setattr(fewshot, "looks_like_refusal", lambda raw: False)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{"index": 0, "file_name": "img0.jpg", "category": "Teeth", "question": "Q0?",
                  "answer": "A", "predicted": "A", "raw_response": "raw", "correct": True,
                  "refused": False, "n_examples": 0}]).to_csv("results/out.csv", index=False)

    fewshot.run(model="claude-haiku", k=0, results_path="results/out.csv")

    assert len(calls) == 2  # only indices 1 and 2
    out_df = pd.read_csv("results/out.csv")
    assert sorted(out_df["index"]) == [0, 1, 2]


def test_run_skips_an_item_whose_call_raises_api_call_failed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(2)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(fewshot, "build_prompt", fake_build_prompt)

    def fake_call(system, messages, model=None, cot=False):
        if "Q0?" in str(messages):
            raise APICallFailed("boom")
        return "A", "raw"

    monkeypatch.setattr(fewshot, "call", fake_call)
    monkeypatch.setattr(fewshot, "looks_like_refusal", lambda raw: False)

    fewshot.run(model="claude-haiku", k=0, results_path="results/out.csv")

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [1]
    assert "[SKIP index 0]" in capsys.readouterr().out


def test_run_respects_start_and_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(5)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(fewshot, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(fewshot, "call", lambda system, messages, model=None, cot=False: ("A", "raw"))
    monkeypatch.setattr(fewshot, "looks_like_refusal", lambda raw: False)

    fewshot.run(model="claude-haiku", k=0, results_path="results/out.csv", start=1, limit=2)

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [1, 2]


def test_run_writes_the_meta_sidecar_with_accuracy_and_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(2)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(fewshot, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(fewshot, "call", lambda system, messages, model=None, cot=False: ("A", "raw"))
    monkeypatch.setattr(fewshot, "looks_like_refusal", lambda raw: False)

    fewshot.run(model="claude-haiku", k=0, results_path="results/out.csv",
               meta={"experiment": "E1"})

    import json
    meta = json.load(open("results/out.csv.meta.json", encoding="utf-8"))
    assert meta["experiment"] == "E1"
    assert meta["model"] == "claude-haiku"
    assert meta["n"] == 2


def test_run_writes_a_partial_csv_after_every_ten_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(11)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(fewshot, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(fewshot, "looks_like_refusal", lambda raw: False)

    def fake_call(system, messages, model=None, cot=False):
        if "Q10?" in str(messages):
            raise RuntimeError("simulated crash on the 11th item")
        return "A", "raw"

    monkeypatch.setattr(fewshot, "call", fake_call)

    with pytest.raises(RuntimeError, match="simulated crash"):
        fewshot.run(model="claude-haiku", k=0, results_path="results/out.csv")

    # the crash happened after the 10-results checkpoint write, so the partial
    # file on disk already has the first 10 rows even though run() never returned
    out_df = pd.read_csv("results/out.csv")
    assert len(out_df) == 10


def test_run_writing_the_progress_file_assumes_the_results_dir_is_named_results(tmp_path, monkeypatch):
    # CURRENT BEHAVIOR (looks like a bug): the progress line is always written to the
    # hardcoded path "results/progress_claude.txt", regardless of where --out points.
    # os.makedirs() correctly creates the DIRNAME of results_path, but if that dirname
    # is not literally "results", the progress-file write crashes.
    monkeypatch.chdir(tmp_path)
    df = make_dataset(1)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(fewshot, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(fewshot, "call", lambda system, messages, model=None, cot=False: ("A", "raw"))
    monkeypatch.setattr(fewshot, "looks_like_refusal", lambda raw: False)

    with pytest.raises(FileNotFoundError):
        fewshot.run(model="claude-haiku", k=0, results_path="otherdir/out.csv")
