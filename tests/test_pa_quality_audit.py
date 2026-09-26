# Tests for paper_analysis/quality_audit.py
# Reads back dentist "quality audit" survey submissions (JSON files) against a manifest
# CSV and reports, over the reviewed questions, how often the recorded answer was
# confirmed / wrong / undeterminable. Writes a table + json into paper_analysis/_generated
# (we redirect that to tmp_path so nothing real is touched).
import json
import os

import pandas as pd

import quality_audit as qa


def _manifest_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _submission_json(path, dentist_name, responses, batch=1, token="tok1"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"dentist_name": dentist_name, "token": token, "batch": batch,
                   "responses": responses}, f)


# ---------- p() ----------

def test_p_joins_paths_under_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(qa, "REPO", str(tmp_path))
    assert qa.p("a", "b.csv") == os.path.join(str(tmp_path), "a", "b.csv")


# ---------- load() ----------

def test_load_reads_manifest_and_flattens_submission_responses(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "REPO", str(tmp_path))
    os.makedirs(tmp_path / "results" / "dentist_audit", exist_ok=True)
    _manifest_csv(tmp_path / "results/dentist_audit/quality_manifest.csv", [
        {"item_id": 1, "index": 0, "task_type": "closed", "question": "Q1", "image": "a", "survey": 1},
    ])
    _submission_json(tmp_path / "results/dentist_audit/quality_alice_submission_tok1.json",
                      "Alice", [{"item_id": 1, "verdict": "correct", "comment": " looks fine "}])

    man, ans, subs = qa.load()
    assert len(man) == 1
    assert len(subs) == 1
    assert list(ans["item_id"]) == [1]
    assert ans.loc[0, "verdict"] == "correct"
    assert ans.loc[0, "comment"] == "looks fine"   # stripped
    assert ans.loc[0, "rater"] == "Alice"


# ---------- main(): no submissions yet ----------

def test_main_reports_nothing_reviewed_yet_when_no_submissions_exist(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(qa, "REPO", str(tmp_path))
    monkeypatch.setattr(qa, "OUT_DIR", str(tmp_path / "generated"))
    os.makedirs(tmp_path / "results" / "dentist_audit", exist_ok=True)
    _manifest_csv(tmp_path / "results/dentist_audit/quality_manifest.csv", [
        {"item_id": 1, "index": 0, "task_type": "closed", "question": "Q1", "image": "img1", "survey": 1},
        {"item_id": 2, "index": 1, "task_type": "closed", "question": "Q2", "image": "img1", "survey": 1},
    ])

    qa.main()

    out = capsys.readouterr().out
    assert "No quality-survey submissions found yet." in out
    assert "manifest holds 2 questions over 1 images in 1 surveys" in out
    assert not os.path.exists(tmp_path / "generated")  # nothing written


# ---------- main(): full run ----------

def test_main_writes_table_and_json_with_submissions(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(qa, "REPO", str(tmp_path))
    out_dir = tmp_path / "generated"
    monkeypatch.setattr(qa, "OUT_DIR", str(out_dir))
    os.makedirs(tmp_path / "results" / "dentist_audit", exist_ok=True)
    _manifest_csv(tmp_path / "results/dentist_audit/quality_manifest.csv", [
        {"item_id": 1, "index": 0, "task_type": "closed", "question": "Q1", "image": "a", "survey": 1},
        {"item_id": 2, "index": 1, "task_type": "closed", "question": "Q2", "image": "a", "survey": 1},
        {"item_id": 3, "index": 2, "task_type": "open", "question": "Q3", "image": "b", "survey": 1},
        {"item_id": 4, "index": 3, "task_type": "open", "question": "Q4", "image": "b", "survey": 1},
    ])
    _submission_json(tmp_path / "results/dentist_audit/quality_alice_submission_tok1.json", "Alice", [
        {"item_id": 1, "verdict": "correct", "comment": ""},
        {"item_id": 2, "verdict": "incorrect", "comment": "wrong tooth"},
        {"item_id": 3, "verdict": "unsure", "comment": ""},
        {"item_id": 4, "verdict": "correct", "comment": ""},
    ])

    qa.main()

    table = (out_dir / "quality_audit_table.md").read_text(encoding="utf-8")
    assert "| recorded answer confirmed | 2 | 50.0% |" in table
    assert "| recorded answer wrong | 1 | 25.0% |" in table
    assert "| cannot be determined from the radiograph | 1 | 25.0% |" in table
    assert "| *reviewed so far* | 4 | of 4 |" in table

    vals = json.loads((out_dir / "quality_audit.values.json").read_text(encoding="utf-8"))
    assert vals["reviewed"] == 4
    assert vals["in_manifest"] == 4
    assert vals["submissions"] == 1
    assert vals["counts"] == {"correct": 2, "incorrect": 1, "unsure": 1}
    assert vals["by_task"] == {
        "closed": {"correct": 1, "incorrect": 1, "unsure": 0},
        "open": {"correct": 1, "incorrect": 0, "unsure": 1},
    }
    assert vals["comments"] == 1
    assert vals["_generator"] == "paper_analysis/quality_audit.py"

    out = capsys.readouterr().out
    assert "by task type:" in out
    assert "comments left on 1 of 4 reviewed questions" in out
    assert "flagged items with a comment (the actionable ones):" in out
    assert "incorrect" in out and "wrong tooth" in out
