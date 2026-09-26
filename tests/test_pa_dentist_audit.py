# Tests for paper_analysis/dentist_audit.py
# Scores one dentist's audit submission against the survey manifest: per model-
# agreement bucket (T1/T2/T3/C), counts agree/disagree/cant/none and a certified
# key-error rate (with a Wilson 95% CI), then a re-weighted whole-key error estimate
# and the open-ended endorsement counts. No API calls; pure CSV/JSON crunching.

import json

import pandas as pd
import pytest

import dentist_audit as da


def make_model_csv(path, keys, predicted):
    pd.DataFrame({"index": list(range(len(keys))), "answer": keys, "predicted": predicted}).to_csv(path)


def build_repo(base):
    """A tiny repo layout that lets full_bucket_sizes()/main() run end to end.

    6 items (index 0..5), 3 "models" (gpt4o/g25/g35), each with a shuffled and an
    original run. Designed by hand so the bucket outcome of dataio.make_dentist_survey's
    real `buckets()` function is known:
      item0: all 3 agree WITH the key (shuffled)          -> C
      item1: all 3 agree on a NON-key letter, in BOTH runs -> T1
      item2: 2 of 3 agree on a non-key letter (shuffled)   -> T2
      item3: all 3 disagree with each other (shuffled)     -> T3
      item4: 2 of 3 agree WITH the key but not all 3       -> "B" (not tallied)
      item5: 2 of 3 agree on a non-key letter (shuffled)   -> T2
    """
    (base / "paper_analysis").mkdir(parents=True, exist_ok=True)
    keys = ["A"] * 6
    closed = pd.DataFrame({
        "index": list(range(6)),
        "option1": [f"opt1{i}" for i in range(6)], "option2": [f"opt2{i}" for i in range(6)],
        "option3": [f"opt3{i}" for i in range(6)], "option4": [f"opt4{i}" for i in range(6)],
    })
    closed.to_parquet(base / "closed.parquet")

    shuf_preds = {"gpt4o": ["A", "B", "B", "B", "A", "C"],
                  "g25":   ["A", "B", "B", "C", "A", "C"],
                  "g35":   ["A", "B", "C", "D", "B", "D"]}
    orig_preds = {"gpt4o": ["A", "C", "A", "A", "A", "A"],
                  "g25":   ["A", "C", "A", "A", "A", "A"],
                  "g35":   ["A", "C", "A", "A", "A", "A"]}
    for m in ("gpt4o", "g25", "g35"):
        make_model_csv(base / f"shuf_{m}.csv", keys, shuf_preds[m])
        make_model_csv(base / f"orig_{m}.csv", keys, orig_preds[m])

    man = pd.DataFrame([
        {"item_id": "c1", "bucket": "T1", "answer_key": "A"},   # agree
        {"item_id": "c2", "bucket": "T1", "answer_key": "A"},   # disagree, ERROR
        {"item_id": "c3", "bucket": "T2", "answer_key": "A"},   # disagree, DEFENSIBLE
        {"item_id": "c4", "bucket": "T2", "answer_key": "A"},   # CANT
        {"item_id": "c5", "bucket": "T3", "answer_key": "A"},   # disagree, UNSURE
        {"item_id": "c6", "bucket": "T3", "answer_key": "A"},   # NONE
        {"item_id": "c7", "bucket": "T3", "answer_key": "A"},   # disagree, no verdict
        {"item_id": "c8", "bucket": "C", "answer_key": "A"},    # agree
        {"item_id": "c9", "bucket": "C", "answer_key": "A"},    # disagree, ERROR
    ])
    man.to_csv(base / "manifest.csv", index=False)

    sub = {
        "token": "dentist1",
        "answers": [
            {"item_id": "c1", "task_type": "closed", "choice": "A"},
            {"item_id": "c2", "task_type": "closed", "choice": "B"},
            {"item_id": "c3", "task_type": "closed", "choice": "C"},
            {"item_id": "c4", "task_type": "closed", "choice": "CANT"},
            {"item_id": "c5", "task_type": "closed", "choice": "D"},
            {"item_id": "c6", "task_type": "closed", "choice": "NONE"},
            {"item_id": "c7", "task_type": "closed", "choice": "B"},
            {"item_id": "c8", "task_type": "closed", "choice": "A"},
            {"item_id": "c9", "task_type": "closed", "choice": "D"},
            {"item_id": "o1", "task_type": "open", "choice": "Agree"},
            {"item_id": "o2", "task_type": "open", "choice": "Partial"},
        ],
        "adjudications": [
            {"item_id": "c2", "verdict": "ERROR"},
            {"item_id": "c3", "verdict": "DEFENSIBLE"},
            {"item_id": "c5", "verdict": "UNSURE"},
            {"item_id": "c9", "verdict": "ERROR"},
        ],
    }
    (base / "submission.json").write_text(json.dumps(sub), encoding="utf-8")
    return sub


def wire_module(monkeypatch, base):
    monkeypatch.setattr(da, "__file__", str(base / "paper_analysis" / "dentist_audit.py"))
    monkeypatch.setattr(da, "CLOSED", "closed.parquet")
    monkeypatch.setattr(da, "SHUF", {"gpt4o": "shuf_gpt4o.csv", "g25": "shuf_g25.csv", "g35": "shuf_g35.csv"})
    monkeypatch.setattr(da, "ORIG", {"gpt4o": "orig_gpt4o.csv", "g25": "orig_g25.csv", "g35": "orig_g35.csv"})


# ---------- wilson() ----------

def test_wilson_zero_n_returns_zero():
    assert da.wilson(0, 0) == (0.0, 0.0)


def test_wilson_matches_hand_check():
    lo, hi = da.wilson(1, 2)
    assert round(lo, 1) == 9.5
    assert round(hi, 1) == 90.5


# ---------- full_bucket_sizes() ----------

def test_full_bucket_sizes(tmp_path, monkeypatch):
    build_repo(tmp_path)
    wire_module(monkeypatch, tmp_path)
    sizes = da.full_bucket_sizes(str(tmp_path))
    assert sizes == {"T1": 1, "T2": 2, "T3": 1, "C": 1}


# ---------- main() end to end ----------

def test_main_reports_buckets_and_reweighted_estimate(tmp_path, monkeypatch, capsys):
    build_repo(tmp_path)
    wire_module(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["dentist_audit.py", str(tmp_path / "submission.json"), "manifest.csv"])

    da.main()

    printed = capsys.readouterr().out
    assert "Dentist audit — token 'dentist1'" in printed
    assert "*** TEST DATA" not in printed
    assert "[pipeline check] 5 definite disagreements, 4 adjudications, exact match: False" in printed
    assert "T1        2      1         1      1     0     0                50% [9-91]" in printed
    assert "T2        2      0         1      0     1     0                 0% [0-66]" in printed
    assert "T3        3      0         2      0     0     1                 0% [0-56]" in printed
    assert "C         2      1         1      1     0     0                50% [9-91]" in printed
    assert "[predicts errors?] suspect T1+T2: 1/4 (25%) vs control C: 1/2 (50%)" in printed
    assert ("[re-weighted] full-491 bucket sizes {'T1': 1, 'T2': 2, 'T3': 1, 'C': 1}; "
            "estimated key errors in those 5 items ≈ 1 (20%)") in printed
    assert "Open-ended reference endorsement (n=2): {'Agree': 1, 'Partial': 1}" in printed


def test_main_test_token_adds_warning_banner(tmp_path, monkeypatch, capsys):
    sub = build_repo(tmp_path)
    sub["token"] = "TESTtoken123"
    (tmp_path / "submission.json").write_text(json.dumps(sub), encoding="utf-8")
    wire_module(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["dentist_audit.py", str(tmp_path / "submission.json"), "manifest.csv"])

    da.main()

    printed = capsys.readouterr().out
    assert "*** TEST DATA — numbers meaningless ***" in printed
    assert "*** Reminder: TEST DATA — do not report any of the above. ***" in printed


def test_main_uses_default_manifest_path_when_not_given(tmp_path, monkeypatch, capsys):
    sub = build_repo(tmp_path)
    # put the manifest at the module's hard-coded default location instead
    default_manifest = tmp_path / "results" / "dentist_audit" / "survey_manifest.csv"
    default_manifest.parent.mkdir(parents=True)
    (tmp_path / "manifest.csv").rename(default_manifest)
    wire_module(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["dentist_audit.py", str(tmp_path / "submission.json")])

    da.main()

    printed = capsys.readouterr().out
    assert "Dentist audit" in printed
    assert "Open-ended reference endorsement" in printed


def test_main_exits_with_usage_message_if_no_submission_arg(monkeypatch):
    monkeypatch.setattr("sys.argv", ["dentist_audit.py"])
    with pytest.raises(SystemExit) as exc:
        da.main()
    assert "usage: python paper_analysis/dentist_audit.py" in str(exc.value)


def test_main_crashes_on_unknown_item_id_with_lettered_choice(tmp_path, monkeypatch):
    # CURRENT BEHAVIOR (looks like a bug): the main accumulation loop guards against
    # an answer whose item_id isn't in the manifest with `if iid not in man.index:
    # continue` (dentist_audit.py, inside the `for a in sub["answers"]` loop). The
    # `disagree_ids` cross-check set comprehension a few lines later does the same
    # "closed task + lettered choice != key" filter but WITHOUT that same guard, so
    # an unknown item_id with a lettered, non-key choice makes `man.loc[...]` raise
    # KeyError instead of being skipped like it is in the main loop.
    sub = build_repo(tmp_path)
    sub["answers"].append({"item_id": "zzz_unknown", "task_type": "closed", "choice": "A"})
    (tmp_path / "submission.json").write_text(json.dumps(sub), encoding="utf-8")
    wire_module(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["dentist_audit.py", str(tmp_path / "submission.json"), "manifest.csv"])

    with pytest.raises(KeyError):
        da.main()
