# Tests for eval_closed_gemini.py
# Closed-ended MCQ harness for Gemini. Supports 3 different few-shot pooling
# strategies (plain k-shot, image-disjoint "pool_images", or 0-shot), an E3
# no-image control, half-crops, a detector "chart" hint, and a context primer.

import json
import os

import pandas as pd
import pytest

import eval_closed_gemini as eg
import dataio.eval_data as eval_data
from clients.errors import APICallFailed


def make_dataset(n, image_ids=None):
    return pd.DataFrame({
        "index": list(range(n)),
        "image_id": image_ids or [f"img{i}" for i in range(n)],
        "file_name": [f"img{i}.jpg" for i in range(n)],
        "category": ["Teeth"] * n,
        "question": [f"Q{i}?" for i in range(n)],
        "answer": ["A" if i % 2 == 0 else "B" for i in range(n)],
    })


def fake_build_prompt(row, examples=None, cot=False, mode="house", context=None,
                      max_image_px=None, crops=False, chart=None, no_image=False,
                      visual_exemplars=None):
    return {"row_index": row["index"], "examples": examples, "context": context,
           "crops": crops, "chart": chart, "no_image": no_image}


def pin_gemini_model(monkeypatch):
    # run() does `gemini_client.MODEL = model` directly (not via monkeypatch), which
    # would otherwise leak across tests; pre-recording it with monkeypatch.setattr
    # makes monkeypatch restore the ORIGINAL value again on teardown.
    monkeypatch.setattr(eg.gemini_client, "MODEL", eg.gemini_client.MODEL)


# ---------- get_examples ----------

def make_pool():
    return pd.DataFrame({"index": [0, 1, 2, 3], "category": ["Teeth", "Teeth", "Patho", "Patho"]})


def test_get_examples_returns_nothing_when_k_is_zero():
    assert eg.get_examples(make_pool(), {"index": 99, "category": "Teeth"}, k=0) == []


def test_get_examples_samples_same_category_rows():
    examples = eg.get_examples(make_pool(), {"index": 99, "category": "Teeth"}, k=1, seed=1)
    assert len(examples) == 1
    assert examples[0]["category"] == "Teeth"


def test_get_examples_refuses_to_sample_the_test_row_itself():
    with pytest.raises(AssertionError, match="DATA CONTAMINATION"):
        eg.get_examples(make_pool(), {"index": 0, "category": "Teeth"}, k=1, seed=1)


def test_get_examples_returns_nothing_for_an_unseen_category():
    assert eg.get_examples(make_pool(), {"index": 99, "category": "Jaw"}, k=2, seed=1) == []


# ---------- print_summary ----------

def test_print_summary_reports_real_accuracy(capsys):
    df = pd.DataFrame({"category": ["Teeth", "Patho"], "correct": [True, False],
                       "predicted": ["A", None], "refused": [False, True]})
    eg.print_summary(df, "gemini-2.5-flash", 2)
    out = capsys.readouterr().out
    expected_acc = df["correct"].mean() * 100
    assert f"{expected_acc:.2f}%" in out
    assert "Refusals :" in out


def test_print_summary_without_refused_column_skips_the_refusal_line(capsys):
    df = pd.DataFrame({"category": ["Teeth"], "correct": [True], "predicted": ["A"]})
    eg.print_summary(df, "gemini-2.5-flash", 0)
    assert "Refusals" not in capsys.readouterr().out


# ---------- _chart_for ----------

def test_chart_for_returns_none_without_a_detector_map():
    assert eg._chart_for(None, {"file_name": "a.jpg"}) is None


def test_chart_for_returns_none_when_the_image_key_is_missing():
    assert eg._chart_for({"other.jpg": {"x": 1}}, {"file_name": "a.jpg"}) is None


def test_chart_for_returns_none_when_the_entry_is_falsy():
    assert eg._chart_for({"a.jpg": None}, {"file_name": "a.jpg"}) is None


def test_chart_for_builds_the_chart_for_a_matching_entry(monkeypatch):
    import detector.tooth_chart as tooth_chart
    monkeypatch.setattr(tooth_chart, "build_chart", lambda entry: f"CHART:{entry}")
    assert eg._chart_for({"a.jpg": {"n": 5}}, {"file_name": "a.jpg"}) == "CHART:{'n': 5}"


def test_chart_for_falls_back_to_image_id_when_file_name_is_absent(monkeypatch):
    import detector.tooth_chart as tooth_chart
    monkeypatch.setattr(tooth_chart, "build_chart", lambda entry: "CHART")
    assert eg._chart_for({"img1": {"n": 1}}, {"image_id": "img1"}) == "CHART"


# ---------- run(): the three pooling strategies ----------

def test_run_zero_shot_scores_the_whole_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pin_gemini_model(monkeypatch)
    df = make_dataset(3)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eg.gemini_client, "call", lambda parts, thinking_budget=None, cot=False: ("A", "raw"))
    monkeypatch.setattr(eg.gemini_client, "looks_like_refusal", lambda raw: False)

    eg.run(model="gemini-2.5-flash", k=0, results_path="results/out.csv")

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [0, 1, 2]
    assert (out_df["n_examples"] == 0).all()


def test_run_k_shot_reserves_the_first_pool_size_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pin_gemini_model(monkeypatch)
    n = eg.POOL_SIZE + 2
    df = make_dataset(n)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eg.gemini_client, "call", lambda parts, thinking_budget=None, cot=False: ("A", "raw"))
    monkeypatch.setattr(eg.gemini_client, "looks_like_refusal", lambda raw: False)

    eg.run(model="gemini-2.5-flash", k=1, results_path="results/out.csv")

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [eg.POOL_SIZE, eg.POOL_SIZE + 1]


def test_run_pool_images_keeps_test_images_disjoint_from_the_pool(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    pin_gemini_model(monkeypatch)
    df = make_dataset(6, image_ids=["imgA", "imgA", "imgB", "imgB", "imgC", "imgC"])
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eg.gemini_client, "call", lambda parts, thinking_budget=None, cot=False: ("A", "raw"))
    monkeypatch.setattr(eg.gemini_client, "looks_like_refusal", lambda raw: False)

    eg.run(model="gemini-2.5-flash", k=1, results_path="results/out.csv", pool_images=1)

    out_df = pd.read_csv("results/out.csv")
    # imgA (index 0,1) was reserved as the pool; only imgB/imgC questions are tested
    assert list(out_df["index"]) == [2, 3, 4, 5]
    assert "image-disjoint few-shot" in capsys.readouterr().out


# ---------- run(): resuming ----------

def test_run_resumes_and_skips_already_done_indices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pin_gemini_model(monkeypatch)
    df = make_dataset(3)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    calls = []
    monkeypatch.setattr(eg.gemini_client, "call",
                        lambda parts, thinking_budget=None, cot=False: (calls.append(parts) or "A", "raw"))
    monkeypatch.setattr(eg.gemini_client, "looks_like_refusal", lambda raw: False)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{"index": 0, "file_name": "img0.jpg", "category": "Teeth", "question": "Q0?",
                  "answer": "A", "predicted": "A", "raw_response": "raw", "correct": True,
                  "refused": False, "n_examples": 0}]).to_csv("results/out.csv", index=False)

    eg.run(model="gemini-2.5-flash", k=0, results_path="results/out.csv")

    assert len(calls) == 2


# ---------- run(): API failures are skipped ----------

def test_run_skips_an_item_whose_call_raises_api_call_failed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    pin_gemini_model(monkeypatch)
    df = make_dataset(2)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)

    def fake_call(parts, thinking_budget=None, cot=False):
        if parts["row_index"] == 0:
            raise APICallFailed("boom")
        return "A", "raw"

    monkeypatch.setattr(eg.gemini_client, "call", fake_call)
    monkeypatch.setattr(eg.gemini_client, "looks_like_refusal", lambda raw: False)

    eg.run(model="gemini-2.5-flash", k=0, results_path="results/out.csv")

    out_df = pd.read_csv("results/out.csv")
    assert list(out_df["index"]) == [1]
    assert "[SKIP index 0]" in capsys.readouterr().out


# ---------- run(): option pass-through ----------

def test_run_passes_no_image_and_context_and_crops_through_to_build_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pin_gemini_model(monkeypatch)
    df = make_dataset(1)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    seen = {}

    def spy_build_prompt(row, **kwargs):
        seen.update(kwargs)
        return {"row_index": row["index"]}

    monkeypatch.setattr(eg, "build_prompt", spy_build_prompt)
    monkeypatch.setattr(eg.gemini_client, "call", lambda parts, thinking_budget=None, cot=False: ("A", "raw"))
    monkeypatch.setattr(eg.gemini_client, "looks_like_refusal", lambda raw: False)

    eg.run(model="gemini-2.5-flash", k=0, results_path="results/out.csv",
          no_image=True, context="primer", crops=True)

    assert seen["no_image"] is True
    assert seen["context"] == "primer"
    assert seen["crops"] is True


def test_run_builds_a_chart_from_the_detector_map(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pin_gemini_model(monkeypatch)
    df = make_dataset(1)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    seen = {}

    def spy_build_prompt(row, **kwargs):
        seen["chart"] = kwargs["chart"]
        return {"row_index": row["index"]}

    monkeypatch.setattr(eg, "build_prompt", spy_build_prompt)
    monkeypatch.setattr(eg.gemini_client, "call", lambda parts, thinking_budget=None, cot=False: ("A", "raw"))
    monkeypatch.setattr(eg.gemini_client, "looks_like_refusal", lambda raw: False)
    import detector.tooth_chart as tooth_chart
    monkeypatch.setattr(tooth_chart, "build_chart", lambda entry: "CHART TEXT")

    eg.run(model="gemini-2.5-flash", k=0, results_path="results/out.csv",
          detector_map={"img0.jpg": {"n": 3}})

    assert seen["chart"] == "CHART TEXT"


# ---------- run(): meta + partial checkpoint ----------

def test_run_writes_the_meta_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pin_gemini_model(monkeypatch)
    df = make_dataset(2)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eg.gemini_client, "call", lambda parts, thinking_budget=None, cot=False: ("A", "raw"))
    monkeypatch.setattr(eg.gemini_client, "looks_like_refusal", lambda raw: False)

    eg.run(model="gemini-2.5-flash", k=0, results_path="results/out.csv", meta={"experiment": "E1"})

    meta = json.load(open("results/out.csv.meta.json", encoding="utf-8"))
    assert meta["experiment"] == "E1"
    assert meta["model"] == "gemini-2.5-flash"
    assert meta["n"] == 2


def test_run_writes_a_partial_csv_after_every_ten_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pin_gemini_model(monkeypatch)
    df = make_dataset(11)
    monkeypatch.setattr(eval_data, "read_closed", lambda path: df)
    monkeypatch.setattr(eg, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eg.gemini_client, "looks_like_refusal", lambda raw: False)

    def fake_call(parts, thinking_budget=None, cot=False):
        if parts["row_index"] == 10:
            raise RuntimeError("simulated crash on the 11th item")
        return "A", "raw"

    monkeypatch.setattr(eg.gemini_client, "call", fake_call)

    with pytest.raises(RuntimeError, match="simulated crash"):
        eg.run(model="gemini-2.5-flash", k=0, results_path="results/out.csv")

    out_df = pd.read_csv("results/out.csv")
    assert len(out_df) == 10
