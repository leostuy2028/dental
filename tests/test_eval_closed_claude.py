# Tests for eval_closed_claude.py
# Closed-ended MCQ harness for Claude with support for a context primer, visual
# exemplars, and extended thinking (--thinking-budget / --effort). Writes a
# pristine CSV + a meta sidecar via utils.results_io.

import json
import os

import pandas as pd
import pytest

import eval_closed_claude as ec
import dataio.eval_data as eval_data
from clients import claude_client
from clients.errors import APICallFailed


def make_dataset(n):
    return pd.DataFrame({
        "index": list(range(n)),
        "file_name": [f"img{i}.jpg" for i in range(n)],
        "category": ["Teeth"] * n,
        "question": [f"Q{i}?" for i in range(n)],
        "answer": ["A" if i % 2 == 0 else "B" for i in range(n)],
    })


def fake_build_prompt(row, cot=False, mode="coax", context=None, visual_exemplars=None):
    return "system", [{"role": "user", "content": row["question"]}]


# ---------- print_summary ----------

def test_print_summary_reports_real_accuracy_and_letter_counts(capsys):
    df = pd.DataFrame({"category": ["Teeth", "Teeth", "Patho"], "correct": [True, False, True],
                       "predicted": ["A", "B", None], "refused": [False, True, False]})
    ec.print_summary(df, "claude-opus", "effort=high")
    out = capsys.readouterr().out
    expected_acc = df["correct"].mean() * 100
    assert f"{expected_acc:.2f}%" in out
    assert "effort=high" in out
    assert "Refusals :" in out


def test_print_summary_without_refused_column_skips_the_refusal_line(capsys):
    df = pd.DataFrame({"category": ["Teeth"], "correct": [True], "predicted": ["A"]})
    ec.print_summary(df, "claude-opus", "think=1000")
    assert "Refusals" not in capsys.readouterr().out


# ---------- run(): happy path ----------

def test_run_scores_every_row_and_writes_csv_and_meta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(3)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(ec, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(claude_client, "call",
                        lambda system, messages, model=None, cot=False, thinking_budget=None,
                        effort=None: ("A", "raw"))
    monkeypatch.setattr(claude_client, "looks_like_refusal", lambda raw: False)

    ec.run(model="claude-opus-4-8", results_path="results/out.csv", data_path="data/closed_ended.parquet")

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [0, 1, 2]
    assert list(out_df["correct"]) == [True, False, True]
    meta = json.load(open("results/out.csv.meta.json", encoding="utf-8"))
    assert meta["model"] == "claude-opus-4-8"
    assert meta["n"] == 3


def test_run_passes_thinking_budget_and_effort_through_to_the_client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(1)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(ec, "build_prompt", fake_build_prompt)
    seen = {}

    def fake_call(system, messages, model=None, cot=False, thinking_budget=None, effort=None):
        seen["thinking_budget"] = thinking_budget
        seen["effort"] = effort
        return "A", "raw"

    monkeypatch.setattr(claude_client, "call", fake_call)
    monkeypatch.setattr(claude_client, "looks_like_refusal", lambda raw: False)

    ec.run(model="claude-opus-4-8", results_path="results/out.csv", data_path="d.parquet",
          thinking_budget=2048, effort="high")

    assert seen == {"thinking_budget": 2048, "effort": "high"}


def test_run_respects_start_and_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(5)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(ec, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(claude_client, "call",
                        lambda system, messages, model=None, cot=False, thinking_budget=None,
                        effort=None: ("A", "raw"))
    monkeypatch.setattr(claude_client, "looks_like_refusal", lambda raw: False)

    ec.run(model="claude-opus-4-8", results_path="results/out.csv", data_path="d.parquet",
          start=1, limit=2)

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [1, 2]


# ---------- run(): resuming ----------

def test_run_resumes_and_skips_already_done_indices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(3)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(ec, "build_prompt", fake_build_prompt)
    calls = []
    monkeypatch.setattr(claude_client, "call",
                        lambda system, messages, model=None, cot=False, thinking_budget=None,
                        effort=None: (calls.append(messages) or "A", "raw"))
    monkeypatch.setattr(claude_client, "looks_like_refusal", lambda raw: False)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{"index": 0, "file_name": "img0.jpg", "category": "Teeth", "question": "Q0?",
                  "answer": "A", "predicted": "A", "raw_response": "raw", "correct": True,
                  "refused": False}]).to_csv("results/out.csv", index=False)

    ec.run(model="claude-opus-4-8", results_path="results/out.csv", data_path="d.parquet")

    assert len(calls) == 2


# ---------- run(): API failures are skipped ----------

def test_run_skips_an_item_whose_call_raises_api_call_failed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(2)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(ec, "build_prompt", fake_build_prompt)

    def fake_call(system, messages, model=None, cot=False, thinking_budget=None, effort=None):
        if "Q0?" in str(messages):
            raise APICallFailed("boom")
        return "A", "raw"

    monkeypatch.setattr(claude_client, "call", fake_call)
    monkeypatch.setattr(claude_client, "looks_like_refusal", lambda raw: False)

    ec.run(model="claude-opus-4-8", results_path="results/out.csv", data_path="d.parquet")

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [1]
    assert "[SKIP index 0]" in capsys.readouterr().out


def test_run_prints_a_message_and_returns_early_when_everything_is_skipped(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(1)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(ec, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(claude_client, "call",
                        lambda *a, **k: (_ for _ in ()).throw(APICallFailed("down")))
    monkeypatch.setattr(claude_client, "looks_like_refusal", lambda raw: False)

    ec.run(model="claude-opus-4-8", results_path="results/out.csv", data_path="d.parquet")

    assert not os.path.exists("results/out.csv")
    assert "No results written" in capsys.readouterr().out


# ---------- run(): writes a partial CSV every 10 results ----------

def test_run_writes_a_partial_csv_after_every_ten_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(11)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(ec, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(claude_client, "looks_like_refusal", lambda raw: False)

    def fake_call(system, messages, model=None, cot=False, thinking_budget=None, effort=None):
        if "Q10?" in str(messages):
            raise RuntimeError("simulated crash on the 11th item")
        return "A", "raw"

    monkeypatch.setattr(claude_client, "call", fake_call)

    with pytest.raises(RuntimeError, match="simulated crash"):
        ec.run(model="claude-opus-4-8", results_path="results/out.csv", data_path="d.parquet")

    out_df = pd.read_csv("results/out.csv")
    assert len(out_df) == 10
