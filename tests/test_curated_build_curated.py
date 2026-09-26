# Tests for curated/build_curated.py
# Builds the "curated" MMOral-Bench release: two small CSVs (disposition +
# corrected key for multiple choice, disposition + repaired reference for free
# text) plus a VERSION.json summary. No images, no copied questions — just the
# changes the paper makes to the released benchmark.

import json
import os

import pandas as pd

import curated.build_curated as build_curated


def write_fixture(base):
    """A tiny stand-in for data/*.parquet + results/corrected_benchmark/*.csv,
    covering every disposition the closed and open sections branch on."""
    os.makedirs(os.path.join(base, "data"), exist_ok=True)
    os.makedirs(os.path.join(base, "results", "corrected_benchmark"), exist_ok=True)

    # index 14 is KEEP but its question still mentions "bone loss": the flag_reason
    # column checks the question text independently of the disposition.
    cl = pd.DataFrame({
        "index": [10, 11, 12, 13, 14],
        "question": ["Is there caries?", "Any bone loss present?", "Tooth numbering question",
                    "Missing region", "Signs of bone loss here"],
        "answer": ["A", "B", "C", "D", "A"],
    })
    sh = pd.DataFrame({"index": [10, 11, 12, 13, 14], "answer": ["B", "A", "D", "C", "C"]})
    op = pd.DataFrame({"index": [1], "question": ["dummy"], "answer": ["x"]})
    cl.to_parquet(os.path.join(base, "data", "closed_ended.parquet"))
    sh.to_parquet(os.path.join(base, "data", "closed_ended_shuffled.parquet"))
    op.to_parquet(os.path.join(base, "data", "open_ended.parquet"))

    mc = pd.DataFrame({"index": [10, 11, 12, 13, 14],
                       "disposition": ["KEEP", "FLAG", "FLAG", "DROP", "KEEP"]})
    mc.to_csv(os.path.join(base, "results", "corrected_benchmark", "manifest_closed.csv"), index=False)

    mo = pd.DataFrame({"index": [20, 21, 22, 23, 24],
                       "concreteness": ["high", "low", "medium", "medium", "high"],
                       "disposition": ["KEEP", "REPAIR", "SPATIAL", "MALFORMED", "SEPARATE"]})
    mo.to_csv(os.path.join(base, "results", "corrected_benchmark", "manifest_open.csv"), index=False)

    rep = pd.DataFrame({"index": [20, 21, 22, 23, 24],
                        "repaired_reference": ["", "rewritten sentence", "", "", ""]})
    rep.to_csv(os.path.join(base, "results", "corrected_benchmark", "repaired_references.csv"), index=False)


def run_build(monkeypatch, tmp_path):
    base = str(tmp_path)
    write_fixture(base)
    monkeypatch.setattr(build_curated, "REPO", base)
    monkeypatch.setattr(build_curated, "HERE", base)
    monkeypatch.setattr(build_curated, "git_commit", lambda: "deadbee")
    build_curated.main()
    closed = pd.read_csv(os.path.join(base, "mmoral_curated_closed.csv"), keep_default_na=False)
    opened = pd.read_csv(os.path.join(base, "mmoral_curated_open.csv"), keep_default_na=False)
    version = json.load(open(os.path.join(base, "VERSION.json"), encoding="utf-8"))
    return closed, opened, version


# ---------- p() / git_commit() ----------

def test_p_joins_onto_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(build_curated, "REPO", str(tmp_path))
    assert build_curated.p("data", "x.parquet") == os.path.join(str(tmp_path), "data", "x.parquet")


def test_git_commit_falls_back_to_unknown_on_error(monkeypatch, tmp_path):
    def fake_check_output(cmd, cwd=None):
        raise FileNotFoundError("no git")
    monkeypatch.setattr(build_curated.subprocess, "check_output", fake_check_output)
    assert build_curated.git_commit() == "unknown"


# ---------- closed csv ----------

def test_closed_csv_keeps_one_row_per_manifest_index(monkeypatch, tmp_path):
    closed, _, _ = run_build(monkeypatch, tmp_path)
    assert list(closed["index"]) == [10, 11, 12, 13, 14]


def test_closed_reason_column_comes_from_the_reason_dict(monkeypatch, tmp_path):
    closed, _, _ = run_build(monkeypatch, tmp_path)
    row = closed[closed["index"] == 10].iloc[0]
    assert row["reason"] == build_curated.REASON["KEEP"]


def test_closed_original_and_balanced_answers_come_from_the_two_different_keys(monkeypatch, tmp_path):
    closed, _, _ = run_build(monkeypatch, tmp_path)
    row = closed[closed["index"] == 10].iloc[0]
    assert row["original_answer"] == "A"
    assert row["balanced_answer"] == "B"


def test_bone_loss_questions_get_the_bone_loss_flag_reason_regardless_of_disposition(monkeypatch, tmp_path):
    closed, _, _ = run_build(monkeypatch, tmp_path)
    # CURRENT BEHAVIOR (looks like a bug, or at least surprising): index 14 is
    # disposition KEEP, not FLAG, but its question text mentions "bone loss" so it
    # still gets the bone-loss flag_reason. The flag text check runs independently
    # of the disposition check.
    row14 = closed[closed["index"] == 14].iloc[0]
    assert row14["disposition"] == "KEEP"
    assert row14["flag_reason"].startswith("bone loss")

    row11 = closed[closed["index"] == 11].iloc[0]
    assert row11["disposition"] == "FLAG"
    assert row11["flag_reason"].startswith("bone loss")


def test_flag_without_bone_loss_gets_the_generic_tooth_code_reason(monkeypatch, tmp_path):
    closed, _, _ = run_build(monkeypatch, tmp_path)
    row = closed[closed["index"] == 12].iloc[0]
    assert row["disposition"] == "FLAG"
    assert "FDI and US Universal" in row["flag_reason"]


def test_keep_without_bone_loss_has_no_flag_reason(monkeypatch, tmp_path):
    closed, _, _ = run_build(monkeypatch, tmp_path)
    row = closed[closed["index"] == 10].iloc[0]
    assert row["flag_reason"] == ""


def test_drop_rows_are_excluded_from_the_reading_score(monkeypatch, tmp_path):
    closed, _, _ = run_build(monkeypatch, tmp_path)
    row = closed[closed["index"] == 13].iloc[0]
    assert row["disposition"] == "DROP"
    assert row["in_reading_score"] == False


def test_non_drop_rows_are_included_in_the_reading_score(monkeypatch, tmp_path):
    closed, _, _ = run_build(monkeypatch, tmp_path)
    assert closed[closed["index"] != 13]["in_reading_score"].eq(True).all()


# ---------- open csv ----------

def test_open_csv_keeps_one_row_per_manifest_index(monkeypatch, tmp_path):
    _, opened, _ = run_build(monkeypatch, tmp_path)
    assert list(opened["index"]) == [20, 21, 22, 23, 24]


def test_repair_row_gets_its_reference_rewritten_and_flagged_changed(monkeypatch, tmp_path):
    _, opened, _ = run_build(monkeypatch, tmp_path)
    row = opened[opened["index"] == 21].iloc[0]
    assert row["repaired_reference"] == "rewritten sentence"
    assert row["reference_changed"] == True


def test_rows_without_a_repair_have_an_empty_reference_and_unchanged_flag(monkeypatch, tmp_path):
    _, opened, _ = run_build(monkeypatch, tmp_path)
    row = opened[opened["index"] == 20].iloc[0]
    assert row["repaired_reference"] == ""
    assert row["reference_changed"] == False


def test_score_track_follows_the_disposition(monkeypatch, tmp_path):
    _, opened, _ = run_build(monkeypatch, tmp_path)
    tracks = dict(zip(opened["index"], opened["score_track"]))
    assert tracks[20] == "reading"    # KEEP
    assert tracks[21] == "reading"    # REPAIR
    assert tracks[22] == "overlap"    # SPATIAL
    assert tracks[23] == "excluded"   # MALFORMED
    assert tracks[24] == "writing"    # SEPARATE


def test_in_reading_score_excludes_separate_spatial_and_malformed(monkeypatch, tmp_path):
    _, opened, _ = run_build(monkeypatch, tmp_path)
    scored = dict(zip(opened["index"], opened["in_reading_score"]))
    assert scored[20] == True
    assert scored[21] == True
    assert scored[22] == False
    assert scored[23] == False
    assert scored[24] == False


# ---------- VERSION.json ----------

def test_version_json_records_the_counts_and_commit(monkeypatch, tmp_path):
    _, _, version = run_build(monkeypatch, tmp_path)
    assert version["version"] == build_curated.VERSION
    assert version["built_from_code_commit"] == "deadbee"
    assert version["closed"]["total"] == 5
    assert version["closed"]["in_reading_score"] == 4
    assert version["open"]["total"] == 5
    assert version["open"]["in_reading_score"] == 2
    assert version["open"]["references_rewritten"] == 1


def test_main_prints_a_summary(monkeypatch, tmp_path, capsys):
    run_build(monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert "MMOral-Bench (curated)" in out
    assert "wrote curated/mmoral_curated_closed.csv" in out
