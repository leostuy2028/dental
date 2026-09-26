# Tests for dataio/make_quality_survey.py
# Builds the question-quality survey manifest: closed KEEP/FLAG items plus open
# KEEP/FLAG/REPAIR items, minus anything already answered in round 1, batched into
# surveys of N images each. REPO is a module-level constant, so we monkeypatch it
# directly to point every relative path at tmp_path.

import json

import pandas as pd

import dataio.make_quality_survey as m


def _make_repo(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "results" / "corrected_benchmark").mkdir(parents=True)
    (tmp_path / "results" / "dentist_audit").mkdir(parents=True)
    return tmp_path


def test_already_answered_returns_empty_sets_when_no_round1_files_exist(tmp_path, monkeypatch):
    _make_repo(tmp_path)
    monkeypatch.setattr(m, "REPO", str(tmp_path))

    closed_idx, open_idx = m.already_answered()

    assert closed_idx == set() and open_idx == set()


def test_already_answered_reads_round1_submission_and_manifest(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    sub_p = repo / "results" / "dentist_audit" / "submission_sandeep.json"
    man_p = repo / "results" / "dentist_audit" / "survey_manifest.csv"
    sub_p.write_text(json.dumps({"answers": [{"item_id": "x1"}, {"item_id": "x2"}]}))
    pd.DataFrame({"item_id": ["x1", "x2"], "task_type": ["closed", "open"], "index": [1, 10]}).to_csv(man_p, index=False)
    monkeypatch.setattr(m, "REPO", str(repo))

    closed_idx, open_idx = m.already_answered()

    assert closed_idx == {1} and open_idx == {10}


def test_risk_by_image_averages_wrongness_across_position_bias_runs(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    pos_dir = repo / "results" / "closed_ended" / "position_bias"
    pos_dir.mkdir(parents=True)
    pd.DataFrame({"index": [1, 2], "correct": [1, 0]}).to_csv(pos_dir / "model1__shuffled__n491.csv", index=False)
    monkeypatch.setattr(m, "REPO", str(repo))

    closed = pd.DataFrame({
        "index": [1, 2], "file_name": ["img1.jpg", "img2.jpg"],
        "category": ["Cat1", "Cat2"], "question": ["Q1", "Q2"],
        "option1": ["a1", "a2"], "option2": ["b1", "b2"], "option3": ["c1", "c2"], "option4": ["d1", "d2"],
        "answer": ["A", "B"],
    })

    risk = m.risk_by_image(closed, {1, 2})

    assert risk == {"img1": 0.0, "img2": 1.0}   # img2's only run got it wrong -> higher risk


def _write_main_inputs(tmp_path):
    repo = _make_repo(tmp_path)
    closed = pd.DataFrame({
        "index": [1, 2], "file_name": ["img1.jpg", "img2.jpg"],
        "category": ["Cat1", "Cat2"], "question": ["Q1?", "Q2?"],
        "option1": ["a1", "a2"], "option2": ["b1", "b2"], "option3": ["c1", "c2"], "option4": ["d1", "d2"],
        "answer": ["A", "B"],
    })
    closed.to_parquet(repo / "data" / "closed_ended.parquet", index=False)
    op = pd.DataFrame({"index": [10, 11], "image_name": ["img3.jpg", "img4.jpg"],
                        "category": ["Patho", "Jaw"], "question": ["Q10?", "Q11?"],
                        "answer": ["ref10", "ref11"]})
    op.to_parquet(repo / "data" / "open_ended.parquet", index=False)
    pd.DataFrame({"index": [1, 2], "disposition": ["KEEP", "DROP"]}).to_csv(
        repo / "results" / "corrected_benchmark" / "manifest_closed.csv", index=False)
    pd.DataFrame({"index": [10, 11], "disposition": ["FLAG", "DROP"]}).to_csv(
        repo / "results" / "corrected_benchmark" / "manifest_open.csv", index=False)
    pd.DataFrame({"index": [10], "repaired_reference": ["fixed ref10"]}).to_csv(
        repo / "results" / "corrected_benchmark" / "repaired_references.csv", index=False)
    pd.DataFrame({"index": [1, 2], "answer": ["C", "B"]}).to_parquet(
        repo / "data" / "closed_ended_shuffled.parquet", index=False)
    return repo


def test_main_keeps_only_keep_flag_repair_items_and_uses_the_shuffled_key(tmp_path, monkeypatch, capsys):
    repo = _write_main_inputs(tmp_path)
    monkeypatch.setattr(m, "REPO", str(repo))
    monkeypatch.setattr("sys.argv", ["prog"])

    m.main()

    out = pd.read_csv(repo / m.OUT)
    # index 2 (closed, DROP) and index 11 (open, DROP) are excluded
    assert sorted(out["index"].tolist()) == [1, 10]

    closed_row = out[out["task_type"] == "closed"].iloc[0]
    assert closed_row["keyed_answer"] == "C"     # from the shuffled key, not the raw "A"
    assert closed_row["keyed_text"] == "c1"      # option3, matching the shuffled letter

    open_row = out[out["task_type"] == "open"].iloc[0]
    assert open_row["reference"] == "fixed ref10"   # repaired reference used, not the raw "ref10"

    printed = capsys.readouterr().out
    assert "to review: 2 questions over 2 images (1 multiple-choice, 1 free-text)" in printed
    assert "skipped as already answered in round 1: 0 closed, 0 open" in printed


def test_main_batches_images_per_survey_flag(tmp_path, monkeypatch, capsys):
    repo = _make_repo(tmp_path)
    closed = pd.DataFrame({
        "index": [1, 2, 3], "file_name": ["img1.jpg", "img2.jpg", "img3.jpg"],
        "category": ["Cat1", "Cat2", "Cat3"], "question": ["Q1?", "Q2?", "Q3?"],
        "option1": ["a1", "a2", "a3"], "option2": ["b1", "b2", "b3"],
        "option3": ["c1", "c2", "c3"], "option4": ["d1", "d2", "d3"], "answer": ["A", "B", "C"],
    })
    closed.to_parquet(repo / "data" / "closed_ended.parquet", index=False)
    op = pd.DataFrame({"index": [10], "image_name": ["img4.jpg"], "category": ["Patho"],
                        "question": ["Q10?"], "answer": ["ref10"]})
    op.to_parquet(repo / "data" / "open_ended.parquet", index=False)
    pd.DataFrame({"index": [1, 2, 3], "disposition": ["KEEP"] * 3}).to_csv(
        repo / "results" / "corrected_benchmark" / "manifest_closed.csv", index=False)
    pd.DataFrame({"index": [10], "disposition": ["KEEP"]}).to_csv(
        repo / "results" / "corrected_benchmark" / "manifest_open.csv", index=False)
    pd.DataFrame({"index": [], "repaired_reference": []}).to_csv(
        repo / "results" / "corrected_benchmark" / "repaired_references.csv", index=False)
    pd.DataFrame({"index": [1, 2, 3], "answer": ["A", "B", "C"]}).to_parquet(
        repo / "data" / "closed_ended_shuffled.parquet", index=False)

    monkeypatch.setattr(m, "REPO", str(repo))
    monkeypatch.setattr("sys.argv", ["prog", "--images-per-survey", "2"])

    m.main()

    out = pd.read_csv(repo / m.OUT)
    assert out.set_index("image")["survey"].to_dict() == {
        "img1": 1, "img2": 1, "img3": 2, "img4": 2,
    }
    printed = capsys.readouterr().out
    assert "2 images per survey -> 2 surveys" in printed
