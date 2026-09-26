# Tests for paper_analysis/question_concreteness.py
# Classifies every open-ended question as Concrete / Broad / Vague, then reports each
# model's free-text score per class. `classify` is also used (imported) by other
# scripts, so it gets thorough coverage here. Writes into paper_analysis/_generated,
# which we redirect to tmp_path.
import json
import math
import os

import pandas as pd

import question_concreteness as qc


# ---------- classify: Concrete ----------

def test_classify_specific_tooth_number_is_concrete():
    assert qc.classify("What tooth is number 21?") == "Concrete"


def test_classify_how_many_is_concrete():
    assert qc.classify("How many teeth are missing?") == "Concrete"


def test_classify_which_tooth_or_teeth_is_concrete():
    assert qc.classify("Which teeth show decay?") == "Concrete"
    assert qc.classify("Which tooth is impacted?") == "Concrete"


def test_classify_detect_or_identify_is_concrete():
    assert qc.classify("Please detect the wisdom teeth.") == "Concrete"
    assert qc.classify("Accurately identify the mandibular structures.") == "Concrete"


def test_classify_which_mandibular_canal_or_areas_is_concrete():
    assert qc.classify("Which mandibular canal is visible?") == "Concrete"


def test_classify_historical_intervention_is_concrete():
    assert qc.classify("Is there a historical intervention observed?") == "Concrete"


def test_classify_list_the_teeth_is_concrete():
    assert qc.classify("List the teeth that are missing.") == "Concrete"


# ---------- classify: Vague ----------

def test_classify_caption_or_summarize_is_vague():
    assert qc.classify("Caption this image.") == "Vague"
    assert qc.classify("Summarize the findings.") == "Vague"
    assert qc.classify("Describe the findings shown.") == "Vague"


# ---------- classify: Broad ----------

def test_classify_general_condition_is_broad():
    assert qc.classify("What is the general condition of the teeth?") == "Broad"


def test_classify_recommend_is_broad():
    assert qc.classify("What treatment would you recommend?") == "Broad"


# ---------- classify: fallback ----------

def test_classify_falls_back_to_concrete_for_unmatched_text():
    # CURRENT BEHAVIOR: anything matching none of the rules is called Concrete
    # ("remaining are list/detect-type -> concrete"), even nonsense text.
    assert qc.classify("zzz qux flim flam") == "Concrete"


def test_classify_is_case_insensitive():
    assert qc.classify("HOW MANY teeth are visible?") == "Concrete"


def test_classify_checks_concrete_rules_before_broad_rules():
    # contains both a Concrete cue ("how many") and a Broad cue ("general condition")
    # -> Concrete wins because its checks run first.
    assert qc.classify("How many teeth show the general condition of decay?") == "Concrete"


def test_classify_handles_non_string_input():
    assert qc.classify(123) == "Concrete"  # str(123) matches no pattern -> fallback


# ---------- acc_ci ----------

def test_acc_ci_on_a_hand_countable_series():
    s = pd.Series([1, 1, 0, 0])
    mu, lo, hi, n = qc.acc_ci(s)
    assert n == 4
    assert mu == 50.0
    # cross-check against the formula by hand: se = std(ddof=1)/sqrt(n)*100
    se = s.std(ddof=1) / math.sqrt(4) * 100
    assert lo == round(50.0 - 1.96 * se, 1)
    assert hi == round(50.0 + 1.96 * se, 1)


def test_acc_ci_single_row_has_zero_spread():
    mu, lo, hi, n = qc.acc_ci(pd.Series([1]))
    assert (mu, lo, hi, n) == (100.0, 100.0, 100.0, 1)


# ---------- main ----------

def test_main_writes_table_and_json_and_prints_templating_fingerprints(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    os.makedirs(repo / "data", exist_ok=True)
    os.makedirs(repo / "results" / "open", exist_ok=True)

    open_ended = pd.DataFrame({
        "index": [0, 1, 2, 3, 4, 5],
        "question": [
            "What tooth number is #21?",       # Concrete
            "How many teeth are missing?",     # Concrete
            "What is the general condition of the teeth?",  # Broad
            "What structures are visible in the jaw?",       # Broad
            "Caption this radiograph.",         # Vague
            "Summarize the findings.",          # Vague
        ],
        "answer": [
            "teeth visualized with clear anatomical definition",
            "no apparent bone loss present",
            "normal answer text",
            "box_2d reference here",
            "normal answer text",
            "normal answer text",
        ],
        "image_name": ["img0", "img1", "img2", "img3", "img4", "img5"],
    })
    open_ended.to_parquet(repo / "data/open_ended.parquet")

    pd.DataFrame({"index": [0, 1, 2, 3, 4, 5], "score": [1, 1, 0, 1, 1, 1]}).to_csv(
        repo / "results/open/batched_gpt5mini_scores.csv", index=False)
    pd.DataFrame({"index": [0, 1, 2, 3, 4, 5], "score": [0, 1, 1, 0, 1, 1]}).to_csv(
        repo / "results/open/batched_gemini35_plain578_scores.csv", index=False)

    out_dir = tmp_path / "generated"
    monkeypatch.setattr(qc, "REPO", str(repo))
    monkeypatch.setattr(qc, "OUT_DIR", str(out_dir))

    qc.main()

    table = (out_dir / "question_concreteness_table.md").read_text(encoding="utf-8")
    assert "**Concrete**" in table
    assert "**Broad**" in table
    assert "**Vague**" in table
    assert "| All | 6 | 100% |" in table

    vals = json.loads((out_dir / "question_concreteness.values.json").read_text(encoding="utf-8"))
    assert vals["total"] == 6
    assert vals["buckets"]["Concrete"]["n"] == 2
    assert vals["buckets"]["Broad"]["n"] == 2
    assert vals["buckets"]["Vague"]["n"] == 2
    assert vals["all"]["n"] == 6
    t = vals["templating"]
    assert t["n_refs"] == 6
    assert t["clear_anatomical_definition"] == 1
    assert t["no_apparent_bone_loss"] == 1
    assert t["coordinate_box_refs"] == 1
    assert vals["_generator"] == "paper_analysis/question_concreteness.py"
    assert vals["_source_csv"] == ["results/open/batched_gpt5mini_scores.csv",
                                   "results/open/batched_gemini35_plain578_scores.csv"]

    out = capsys.readouterr().out
    assert "templating fingerprints in 6 refs" in out
    assert "wrote" in out
