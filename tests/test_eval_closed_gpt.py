# Tests for eval_closed_gpt.py
# Closed-ended MCQ harness for GPT. Two prompt modes: "faithful" (uses the
# benchmark's own VLMEvalKit-style extractor with a seeded random fallback) and
# "coax" (strict letter extraction, honest refusal flag). Faithful mode
# hard-enforces the benchmark-faithful generation config (temp=0, max_tokens=8192,
# img_detail=high) and refuses a reasoning model outright.

import json
import os

import pandas as pd
import pytest

import eval_closed_gpt as eg
import dataio.eval_data as eval_data
from clients.errors import APICallFailed


def make_row(index=5, **overrides):
    row = {"index": index, "option1": "Caries", "option2": "Impaction",
          "option3": "None", "option4": "Bone loss"}
    row.update(overrides)
    return row


def make_dataset(n):
    return pd.DataFrame({
        "index": list(range(n)),
        "file_name": [f"img{i}.jpg" for i in range(n)],
        "category": ["Teeth"] * n,
        "question": [f"Q{i}?" for i in range(n)],
        "option1": ["1"] * n, "option2": ["2"] * n, "option3": ["3"] * n, "option4": ["4"] * n,
        "answer": ["A" if i % 2 == 0 else "B" for i in range(n)],
    })


def fake_build_prompt(row, examples=None, cot=False, mode="faithful", detail="high",
                      context=None, chart=None):
    return "system", [{"type": "text", "text": row["question"]}]


# ---------- extract ----------

def test_extract_faithful_mode_parses_a_parenthesized_letter():
    pred, used_fallback, refused = eg.extract("faithful", "The answer is (B).", make_row(), cot=False)
    assert (pred, used_fallback, refused) == ("B", False, False)


def test_extract_faithful_mode_flags_the_random_fallback():
    pred, used_fallback, refused = eg.extract("faithful", "no match here", make_row(index=5), cot=False)
    assert used_fallback is True
    assert pred in ("A", "B", "C", "D")


def test_extract_faithful_mode_is_reproducible_for_the_same_row_index():
    a = eg.extract("faithful", "no match here", make_row(index=5), cot=False)
    b = eg.extract("faithful", "no match here", make_row(index=5), cot=False)
    assert a == b


def test_extract_coax_mode_uses_strict_letter_extraction():
    pred, used_fallback, refused = eg.extract("coax", "The answer is B", make_row(), cot=False)
    assert (pred, used_fallback, refused) == ("B", False, False)


def test_extract_coax_mode_never_uses_the_random_fallback():
    pred, used_fallback, refused = eg.extract("coax", "gibberish, no letter", make_row(), cot=False)
    assert (pred, used_fallback) == (None, False)


def test_extract_coax_mode_flags_a_refusal():
    pred, used_fallback, refused = eg.extract("coax", "I cannot help with medical images", make_row(), cot=False)
    assert (pred, refused) == (None, True)


# ---------- get_examples ----------

def test_get_examples_returns_nothing_when_k_is_zero():
    pool = pd.DataFrame({"index": [0, 1], "category": ["Teeth", "Teeth"]})
    assert eg.get_examples(pool, {"index": 9, "category": "Teeth"}, k=0) == []


def test_get_examples_refuses_to_sample_the_test_row_itself():
    pool = pd.DataFrame({"index": [0, 1], "category": ["Teeth", "Teeth"]})
    with pytest.raises(AssertionError, match="DATA CONTAMINATION"):
        eg.get_examples(pool, {"index": 0, "category": "Teeth"}, k=1, seed=1)


def test_get_examples_returns_nothing_for_an_unseen_category():
    pool = pd.DataFrame({"index": [0], "category": ["Teeth"]})
    assert eg.get_examples(pool, {"index": 9, "category": "Jaw"}, k=1, seed=1) == []


# ---------- print_summary ----------

def test_print_summary_faithful_mode_shows_the_random_fallback_rate(capsys):
    df = pd.DataFrame({"category": ["Teeth", "Patho"], "correct": [True, False],
                       "predicted": ["A", None], "refused": [False, True],
                       "used_fallback": [False, True]})
    eg.print_summary(df, "gpt-4o", 2, "faithful")
    out = capsys.readouterr().out
    assert "Random-fallback (unparsed->guess): 50.0%" in out


def test_print_summary_coax_mode_shows_unparseable_rate_instead(capsys):
    df = pd.DataFrame({"category": ["Teeth", "Patho"], "correct": [True, False],
                       "predicted": ["A", None], "refused": [False, True],
                       "used_fallback": [False, False]})
    eg.print_summary(df, "gpt-4o", 2, "coax")
    out = capsys.readouterr().out
    assert "Unparseable (scored wrong): 50.0%" in out
    assert "Random-fallback" not in out


# ---------- run(): faithful mode hard-enforces benchmark config ----------

def test_run_faithful_mode_refuses_a_reasoning_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(1)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)

    with pytest.raises(SystemExit, match="requires a non-reasoning model"):
        eg.run(model="gpt-5.6-luna", k=0, results_path="results/out.csv",
              data_path="d.parquet", mode="faithful")


def test_run_faithful_mode_forces_image_detail_to_high(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(1)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eg.gpt_client, "call",
                        lambda system, content, model=None, cot=False, service_tier=None,
                        reasoning_effort="none": "The answer is (A).")

    eg.run(model="gpt-4o-2024-11-20", k=0, results_path="results/out.csv",
          data_path="d.parquet", mode="faithful", detail="low")

    assert "forcing image detail 'low' -> 'high'" in capsys.readouterr().out


def test_run_coax_mode_does_not_force_image_detail(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(1)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eg.gpt_client, "call",
                        lambda system, content, model=None, cot=False, service_tier=None,
                        reasoning_effort="none": "B")

    eg.run(model="gpt-4o-2024-11-20", k=0, results_path="results/out.csv",
          data_path="d.parquet", mode="coax", detail="low")

    assert "forcing image detail" not in capsys.readouterr().out


# ---------- run(): happy path ----------

def test_run_scores_every_row_and_writes_csv_and_meta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(3)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eg.gpt_client, "call",
                        lambda system, content, model=None, cot=False, service_tier=None,
                        reasoning_effort="none": "The answer is (A).")

    eg.run(model="gpt-4o-2024-11-20", k=0, results_path="results/out.csv",
          data_path="d.parquet", mode="faithful")

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [0, 1, 2]
    assert list(out_df["predicted"]) == ["A", "A", "A"]
    meta = json.load(open("results/out.csv.meta.json", encoding="utf-8"))
    assert meta["model"] == "gpt-4o-2024-11-20"
    assert meta["n"] == 3
    assert "refusal_pct" in meta
    assert "random_fallback_pct" in meta


def test_run_service_tier_and_reasoning_effort_are_forwarded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(1)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    seen = {}

    def fake_call(system, content, model=None, cot=False, service_tier=None, reasoning_effort="none"):
        seen["service_tier"] = service_tier
        seen["reasoning_effort"] = reasoning_effort
        return "B"

    monkeypatch.setattr(eg.gpt_client, "call", fake_call)

    eg.run(model="gpt-4o-2024-11-20", k=0, results_path="results/out.csv", data_path="d.parquet",
          mode="coax", service_tier="flex", reasoning_effort="low")

    assert seen == {"service_tier": "flex", "reasoning_effort": "low"}


def test_run_respects_start_and_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(5)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eg.gpt_client, "call",
                        lambda system, content, model=None, cot=False, service_tier=None,
                        reasoning_effort="none": "B")

    eg.run(model="gpt-4o-2024-11-20", k=0, results_path="results/out.csv", data_path="d.parquet",
          mode="coax", start=1, limit=2)

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [1, 2]


def test_run_k_shot_reserves_the_first_pool_size_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    n = eg.POOL_SIZE + 2
    df = make_dataset(n)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eg.gpt_client, "call",
                        lambda system, content, model=None, cot=False, service_tier=None,
                        reasoning_effort="none": "B")

    eg.run(model="gpt-4o-2024-11-20", k=1, results_path="results/out.csv", data_path="d.parquet",
          mode="coax")

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [eg.POOL_SIZE, eg.POOL_SIZE + 1]
    assert (out_df["n_examples"] == 1).all()


# ---------- run(): detector chart ----------

def test_run_builds_a_chart_from_the_detector_map(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(1)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    seen = {}

    def spy_build_prompt(row, examples=None, cot=False, mode="faithful", detail="high",
                         context=None, chart=None):
        seen["chart"] = chart
        return "system", [{"type": "text", "text": row["question"]}]

    monkeypatch.setattr(eg, "build_prompt", spy_build_prompt)
    monkeypatch.setattr(eg.gpt_client, "call",
                        lambda system, content, model=None, cot=False, service_tier=None,
                        reasoning_effort="none": "B")
    import detector.tooth_chart as tooth_chart
    monkeypatch.setattr(tooth_chart, "build_chart", lambda entry: "CHART TEXT")

    eg.run(model="gpt-4o-2024-11-20", k=0, results_path="results/out.csv", data_path="d.parquet",
          mode="coax", detector_map={"img0.jpg": {"n": 3}})

    assert seen["chart"] == "CHART TEXT"


def test_run_chart_is_none_when_the_detector_map_has_no_entry_for_the_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(1)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    seen = {}

    def spy_build_prompt(row, examples=None, cot=False, mode="faithful", detail="high",
                         context=None, chart=None):
        seen["chart"] = chart
        return "system", [{"type": "text", "text": row["question"]}]

    monkeypatch.setattr(eg, "build_prompt", spy_build_prompt)
    monkeypatch.setattr(eg.gpt_client, "call",
                        lambda system, content, model=None, cot=False, service_tier=None,
                        reasoning_effort="none": "B")

    eg.run(model="gpt-4o-2024-11-20", k=0, results_path="results/out.csv", data_path="d.parquet",
          mode="coax", detector_map={"other.jpg": {"n": 3}})

    assert seen["chart"] is None


# ---------- run(): resuming and API failures ----------

def test_run_resumes_and_skips_already_done_indices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(3)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    calls = []
    monkeypatch.setattr(eg.gpt_client, "call",
                        lambda system, content, model=None, cot=False, service_tier=None,
                        reasoning_effort="none": (calls.append(content) or "B"))

    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{"index": 0, "file_name": "img0.jpg", "category": "Teeth", "question": "Q0?",
                  "answer": "A", "predicted": "A", "raw_response": "raw", "correct": True,
                  "refused": False, "used_fallback": False, "n_examples": 0,
                  "prompt_mode": "coax"}]).to_csv("results/out.csv", index=False)

    eg.run(model="gpt-4o-2024-11-20", k=0, results_path="results/out.csv", data_path="d.parquet",
          mode="coax")

    assert len(calls) == 2


def test_run_skips_an_item_whose_call_raises_api_call_failed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(2)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)

    def fake_call(system, content, model=None, cot=False, service_tier=None, reasoning_effort="none"):
        if "Q0?" in str(content):
            raise APICallFailed("boom")
        return "B"

    monkeypatch.setattr(eg.gpt_client, "call", fake_call)

    eg.run(model="gpt-4o-2024-11-20", k=0, results_path="results/out.csv", data_path="d.parquet",
          mode="coax")

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [1]
    assert "[SKIP index 0]" in capsys.readouterr().out


def test_run_writes_a_partial_csv_after_every_ten_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_dataset(11)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)

    def fake_call(system, content, model=None, cot=False, service_tier=None, reasoning_effort="none"):
        if "Q10?" in str(content):
            raise RuntimeError("simulated crash on the 11th item")
        return "B"

    monkeypatch.setattr(eg.gpt_client, "call", fake_call)

    with pytest.raises(RuntimeError, match="simulated crash"):
        eg.run(model="gpt-4o-2024-11-20", k=0, results_path="results/out.csv", data_path="d.parquet",
              mode="coax")

    out_df = pd.read_csv("results/out.csv")
    assert len(out_df) == 10
