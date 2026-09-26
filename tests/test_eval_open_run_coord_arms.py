# Tests for eval_open/run_coord_arms.py
# This runs the four-arm "coordinate elicitation" study: one item x one arm gets
# one real-model answer (answer_openai_custom), then one real-model grade (grade),
# under the benchmark's own unchanged rubric. It has resumable answers/scores CSVs
# and writes reproducibility sidecars via utils.results_io.write_results at the end.

import base64
import json
import os
from io import BytesIO

import pandas as pd
from PIL import Image

import eval_open.run_coord_arms as rc


def _tiny_jpeg_b64():
    img = Image.new("RGB", (20, 10))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------- looks_refused ----------

def test_looks_refused_true_on_a_refusal_phrase():
    assert rc.looks_refused("As an AI I cannot help with that.") is True


def test_looks_refused_case_insensitive():
    assert rc.looks_refused("AS AN AI I CANNOT ASSIST") is True


def test_looks_refused_false_on_a_normal_answer():
    assert rc.looks_refused("The wisdom tooth #38 is impacted.") is False


def test_looks_refused_true_on_empty_string():
    assert rc.looks_refused("") is True


def test_looks_refused_false_on_none():
    # CURRENT BEHAVIOR (looks like a bug): str(None) == "None", which is non-empty
    # and doesn't match the refusal pattern, so a None answer is NOT flagged as refused
    assert rc.looks_refused(None) is False


def test_looks_refused_only_checks_the_first_200_characters():
    # word-boundary spacing so the phrase would actually match if it were reached
    pad = "x " * 90  # 180 chars
    inside = pad + "i cannot assist"
    assert len(inside) < 200
    assert rc.looks_refused(inside) is True

    pad200 = "x " * 100  # exactly 200 chars
    outside = pad200 + "i cannot assist"
    # the phrase starts exactly at index 200, so a[:200] never contains it
    assert rc.looks_refused(outside) is False


# ---------- select_items ----------

def _write_preds(path, index, ref_type):
    pd.DataFrame({"index": index, "ref_type": ref_type}).to_parquet(path)


def test_select_items_coord_ref_only(tmp_path, monkeypatch):
    preds_path = tmp_path / "preds.parquet"
    _write_preds(preds_path, [10, 11, 12, 13, 14],
                ["coord_ref", "coord_ref", "prose_ref", "prose_ref", "prose_ref"])
    monkeypatch.setattr(rc, "PREDS", str(preds_path))

    assert rc.select_items("coord_ref", 2, 0) == [10, 11]


def test_select_items_prose_ref_only(tmp_path, monkeypatch):
    preds_path = tmp_path / "preds.parquet"
    _write_preds(preds_path, [10, 11, 12, 13, 14],
                ["coord_ref", "coord_ref", "prose_ref", "prose_ref", "prose_ref"])
    monkeypatch.setattr(rc, "PREDS", str(preds_path))

    assert rc.select_items("prose_ref", 2, 0) == [13, 14]


def test_select_items_all_ref_type(tmp_path, monkeypatch):
    preds_path = tmp_path / "preds.parquet"
    _write_preds(preds_path, [10, 11, 12, 13, 14],
                ["coord_ref", "coord_ref", "prose_ref", "prose_ref", "prose_ref"])
    monkeypatch.setattr(rc, "PREDS", str(preds_path))

    assert rc.select_items("all", 3, 0) == [10, 11, 12]


def test_select_items_n_larger_than_pool_returns_the_whole_pool(tmp_path, monkeypatch):
    preds_path = tmp_path / "preds.parquet"
    _write_preds(preds_path, [10, 11, 12, 13, 14],
                ["coord_ref", "coord_ref", "prose_ref", "prose_ref", "prose_ref"])
    monkeypatch.setattr(rc, "PREDS", str(preds_path))

    assert rc.select_items("all", 100, 0) == [10, 11, 12, 13, 14]


# ---------- phase1_answer ----------

def _setup_primer(tmp_path, monkeypatch, text="PRIMER TEXT"):
    primer_path = tmp_path / "primer.txt"
    primer_path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(rc, "PRIMER", str(primer_path))


def test_phase1_answer_writes_expected_columns_for_every_item_arm_pair(tmp_path, monkeypatch):
    _setup_primer(tmp_path, monkeypatch)
    img_b64 = _tiny_jpeg_b64()
    items = pd.DataFrame({"index": [1], "image": [img_b64], "question": ["Q1?"], "answer": ["GT1"]})

    def fake_answer(image_b64, system, user_text, model=None, max_tokens=None, detail=None):
        return "a real answer"

    monkeypatch.setattr(rc, "answer_openai_custom", fake_answer)
    ans_out = str(tmp_path / "answers.csv")

    out = rc.phase1_answer(items, rc.ARMS, "model-x", ans_out)

    assert list(out.columns) == ["index", "arm", "question", "gt", "answer", "refused"]
    assert sorted(out["arm"]) == sorted(rc.ARMS)
    assert (out["answer"] == "a real answer").all()
    assert (out["refused"] == 0).all()
    assert os.path.exists(ans_out)


def test_phase1_answer_marks_refused_when_the_reply_looks_like_a_refusal(tmp_path, monkeypatch):
    _setup_primer(tmp_path, monkeypatch)
    img_b64 = _tiny_jpeg_b64()
    items = pd.DataFrame({"index": [1], "image": [img_b64], "question": ["Q1?"], "answer": ["GT1"]})

    def fake_answer(image_b64, system, user_text, model=None, max_tokens=None, detail=None):
        return "As an AI I cannot help with that."

    monkeypatch.setattr(rc, "answer_openai_custom", fake_answer)
    out = rc.phase1_answer(items, ["plain"], "model-x", str(tmp_path / "answers.csv"))

    assert out["refused"].tolist() == [1]


def test_phase1_answer_resumes_and_skips_index_arm_pairs_already_done(tmp_path, monkeypatch):
    _setup_primer(tmp_path, monkeypatch)
    img_b64 = _tiny_jpeg_b64()
    items = pd.DataFrame({"index": [1, 2], "image": [img_b64, img_b64],
                          "question": ["Q1?", "Q2?"], "answer": ["GT1", "GT2"]})
    ans_out = str(tmp_path / "answers.csv")
    pd.DataFrame({"index": [1], "arm": ["plain"], "question": ["Q1?"], "gt": ["GT1"],
                 "answer": ["done-before"], "refused": [0]}).to_csv(ans_out, index=False)

    calls = []

    def fake_answer(image_b64, system, user_text, model=None, max_tokens=None, detail=None):
        calls.append(1)
        return "fresh answer"

    monkeypatch.setattr(rc, "answer_openai_custom", fake_answer)
    out = rc.phase1_answer(items, ["plain"], "model-x", ans_out)

    assert len(calls) == 1  # only index=2/plain was missing
    assert sorted(out["answer"]) == ["done-before", "fresh answer"]


# ---------- phase2_grade ----------

def test_phase2_grade_writes_a_score_per_answer(tmp_path, monkeypatch):
    answers = pd.DataFrame({"index": [1, 2], "arm": ["plain", "plain"],
                            "question": ["Q1?", "Q2?"], "gt": ["GT1", "GT2"],
                            "answer": ["A1", "A2"]})

    def fake_grade(prompt, judge=None, model=None):
        return 0.6, "raw text"

    monkeypatch.setattr(rc, "grade", fake_grade)
    score_out = str(tmp_path / "scores.csv")

    out = rc.phase2_grade(answers, "gpt-4o", None, score_out)

    assert list(out.columns) == ["index", "arm", "score", "judge_raw"]
    assert (out["score"] == 0.6).all()
    assert os.path.exists(score_out)


def test_phase2_grade_resumes_and_skips_index_arm_pairs_already_scored(tmp_path, monkeypatch):
    answers = pd.DataFrame({"index": [1, 2], "arm": ["plain", "plain"],
                            "question": ["Q1?", "Q2?"], "gt": ["GT1", "GT2"],
                            "answer": ["A1", "A2"]})
    score_out = str(tmp_path / "scores.csv")
    pd.DataFrame({"index": [1], "arm": ["plain"], "score": [1.0], "judge_raw": ["x"]}).to_csv(
        score_out, index=False)

    calls = []

    def fake_grade(prompt, judge=None, model=None):
        calls.append(1)
        return 0.2, "raw2"

    monkeypatch.setattr(rc, "grade", fake_grade)
    out = rc.phase2_grade(answers, "gpt-4o", None, score_out)

    assert len(calls) == 1
    assert sorted(out["score"].tolist()) == [0.2, 1.0]


# ---------- report ----------

def test_report_prints_a_summary_and_does_not_crash(capsys):
    answers = pd.DataFrame({"index": [1, 2, 1, 2], "arm": ["plain", "plain", "coax", "coax"],
                            "refused": [0, 1, 0, 0]})
    scores = pd.DataFrame({"index": [1, 2, 1, 2], "arm": ["plain", "plain", "coax", "coax"],
                          "score": [0.5, 0.5, 0.7, 0.9]})

    rc.report(answers, scores, ["plain", "coax"])

    out = capsys.readouterr().out
    assert "COORD-ARMS" in out
    assert "plain" in out and "coax" in out


def test_report_handles_zero_differences_without_calling_wilcoxon(capsys):
    # every score is identical across arms -> d.abs().sum() is 0 -> the code
    # takes the "else 1.0" branch instead of calling wilcoxon at all
    answers = pd.DataFrame({"index": [1, 2], "arm": ["plain", "plain"], "refused": [0, 0]})
    scores = pd.DataFrame({"index": [1, 2], "arm": ["plain", "plain"], "score": [0.5, 0.5]})

    rc.report(answers, scores, ["plain"])

    out = capsys.readouterr().out
    assert "COORD-ARMS" in out


# ---------- main() ----------

def test_main_writes_csvs_and_reproducibility_sidecars(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img_b64 = _tiny_jpeg_b64()

    data_path = tmp_path / "open_ended.parquet"
    pd.DataFrame({"index": [1, 2], "image": [img_b64, img_b64],
                 "question": ["Q1?", "Q2?"], "answer": ["GT1", "GT2"]}).to_parquet(data_path)
    preds_path = tmp_path / "predictions.parquet"
    pd.DataFrame({"index": [1, 2], "ref_type": ["coord_ref", "coord_ref"]}).to_parquet(preds_path)
    primer_path = tmp_path / "primer.txt"
    primer_path.write_text("PRIMER", encoding="utf-8")

    monkeypatch.setattr(rc, "DATA", str(data_path))
    monkeypatch.setattr(rc, "PREDS", str(preds_path))
    monkeypatch.setattr(rc, "PRIMER", str(primer_path))
    monkeypatch.setattr(rc, "answer_openai_custom",
                        lambda *a, **kw: "a real answer")
    monkeypatch.setattr(rc, "grade", lambda *a, **kw: (0.5, "raw"))

    monkeypatch.setattr(
        "sys.argv",
        ["run_coord_arms.py", "--n", "2", "--ref-type", "coord_ref", "--seed", "0",
         "--model", "model-x", "--judge", "gpt-4o", "--arms", "plain,coax", "--tag", "tagT"],
    )
    rc.main()

    answers = pd.read_csv("results/open/coordarms_tagT_answers.csv")
    scores = pd.read_csv("results/open/coordarms_tagT_scores.csv")
    assert len(answers) == 4  # 2 items x 2 arms
    assert len(scores) == 4
    assert (scores["score"] == 0.5).all()

    with open("results/open/coordarms_tagT_answers.csv.meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model"] == "model-x"
    assert meta["arms"] == ["plain", "coax"]
    assert "code_commit" in meta and "generated_utc" in meta
    assert os.path.exists("results/open/coordarms_tagT_scores.csv.meta.json")
