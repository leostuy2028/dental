# Tests for paper_analysis/faithful_true_accuracy.py
# This script builds the paper's §5.2 table for GPT-4o on the full 491-question set:
# the benchmark parser's accuracy on the original ("faithful") prompt, the model's
# TRUE (hand-verified) accuracy on that same prompt, and the revised ("coax") prompt's
# parser accuracy -- plus a McNemar test between the two pipelines. Writes a markdown
# table + a JSON of the numbers. No API calls: everything comes from committed CSVs,
# the dataset parquet, and the committed hand-label CSV.

import json
import os

import pandas as pd
import pytest

import faithful_true_accuracy as fta


# ---------- wilson ----------

def test_wilson_matches_hand_checked_interval():
    lo, hi = fta.wilson(3, 6)
    assert (lo, hi) == (18.8, 81.2)


# ---------- mcnemar ----------

def test_mcnemar_counts_discordant_pairs_with_continuity_correction():
    a_right = [True, False, True, False]
    b_right = [True, True, False, False]
    # item1: both right (concordant); item2: a-wrong b-right -> b=1;
    # item3: a-right b-wrong -> c=1; item4: both wrong (concordant)
    b, c, chi2, p = fta.mcnemar(a_right, b_right)
    assert (b, c) == (1, 1)
    assert chi2 == 0.5
    assert round(p, 4) == 0.4795


def test_mcnemar_no_discordant_pairs_gives_chi2_zero_p_one():
    b, c, chi2, p = fta.mcnemar([True, True], [True, True])
    assert (b, c, chi2, p) == (0, 0, 0.0, 1.0)


# ---------- high_conf ----------

def test_high_conf_reads_answer_is_cue():
    assert fta.high_conf("The answer is A") == "A"


def test_high_conf_is_case_insensitive_for_the_cue():
    assert fta.high_conf("the ANSWER IS d") == "D"


def test_high_conf_reads_correct_answer_cue():
    assert fta.high_conf("The correct answer is D") == "D"


def test_high_conf_uses_last_cue_when_several_appear():
    assert fta.high_conf("Answer: B. Wait, re-checking. Answer: C") == "C"


def test_high_conf_reads_bold_letter_when_no_cue():
    assert fta.high_conf("**C.** is correct") == "C"


def test_high_conf_bold_pattern_is_case_sensitive():
    # CURRENT BEHAVIOR (looks like a bug): the "answer is X" cue regex is compiled
    # with re.I, but the bold-letter fallback regex has no re.I flag, so a lowercase
    # bolded letter (e.g. a model replying "**b.**") is silently never read, even
    # though high_conf's own docstring implies it reads "a bolded letter" generically.
    assert fta.high_conf("**b.** correct") is None


def test_high_conf_returns_none_for_a_refusal_with_no_letter():
    assert fta.high_conf("I cannot determine the correct answer") is None


def test_high_conf_handles_none_input():
    assert fta.high_conf(None) is None


# ---------- paper_pick ----------

def test_paper_pick_matches_a_bare_letter():
    opts = pd.DataFrame({
        "option1": ["Tooth #31"], "option2": ["Tooth #32"],
        "option3": ["Tooth #33"], "option4": ["None of the above"],
    }, index=[1])
    letter, fallback = fta.paper_pick(1, "A", opts)
    assert (letter, fallback) == ("A", False)


def test_paper_pick_falls_back_to_option_text_match():
    opts = pd.DataFrame({
        "option1": ["Tooth #31"], "option2": ["Tooth #32"],
        "option3": ["Tooth #33"], "option4": ["None of the above"],
    }, index=[1])
    letter, fallback = fta.paper_pick(1, "It is clearly Tooth #32 here", opts)
    assert (letter, fallback) == ("B", False)


def test_paper_pick_uses_seeded_random_fallback_when_nothing_matches():
    opts = pd.DataFrame({
        "option1": ["Tooth #31"], "option2": ["Tooth #32"],
        "option3": ["Tooth #33"], "option4": ["None of the above"],
    }, index=[1])
    letter, fallback = fta.paper_pick(1, "completely unrelated text", opts)
    assert fallback is True
    assert letter in ("A", "B", "C", "D")
    # seeded on i=1, so re-running is reproducible
    letter2, fallback2 = fta.paper_pick(1, "completely unrelated text", opts)
    assert (letter2, fallback2) == (letter, fallback)


# ---------- main() end to end ----------

def build_fixture(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    results_dir = tmp_path / "results/closed_ended"
    results_dir.mkdir(parents=True)
    pa_dir = tmp_path / "pa"
    pa_dir.mkdir()

    opts = pd.DataFrame({
        "index": [1, 2, 3, 4, 5, 6],
        "option1": ["Tooth #31"] * 6,
        "option2": ["Tooth #32"] * 6,
        "option3": ["Tooth #33"] * 6,
        "option4": ["None of the above"] * 6,
    })
    opts.to_parquet(data_dir / "closed_ended.parquet")

    fa = pd.DataFrame({
        "index": [1, 2, 3, 4, 5, 6],
        "raw_response": [
            "The answer is A",
            "I think it might be B, not sure",
            "**C.** is correct",
            "The correct answer is D",
            "I cannot determine the correct answer",
            "Answer: A",
        ],
        "answer": ["A", "B", "D", "D", "A", "A"],
    })
    fa.to_csv(results_dir / "gpt-4o-2024-11-20__faithful-direct-k0__whole__n491.csv", index=False)

    cx = pd.DataFrame({
        "index": [1, 2, 3, 4, 5],
        "raw_response": ["A", "B.", "  C)", "not a letter reply at all", "D"],
        "answer": ["A", "B", "C", "D", "D"],
    })
    cx.to_csv(results_dir / "gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv", index=False)

    labels = pd.DataFrame({"index": [2, 5], "true_letter": ["B", ""]})
    labels.to_csv(pa_dir / "faithful_hand_labels.csv", index=False)
    return pa_dir


def patch_module_paths(monkeypatch, tmp_path, pa_dir):
    monkeypatch.setattr(fta, "REPO", str(tmp_path))
    monkeypatch.setattr(fta, "HERE", str(pa_dir))
    monkeypatch.setattr(fta, "OUT_DIR", str(tmp_path / "out"))


def test_main_prints_table_and_summary_lines(tmp_path, monkeypatch, capsys):
    pa_dir = build_fixture(tmp_path)
    patch_module_paths(monkeypatch, tmp_path, pa_dir)

    fta.main()
    out = capsys.readouterr().out

    assert "| Original prompt, scored by the benchmark's parser | 50.0% | 18.8 to 81.2 |" in out
    assert ("| Original prompt, scored by the model's true answer (hand-verified) | "
            "66.7% | 30.0 to 90.3 |") in out
    assert "| Revised prompt, scored by the benchmark's parser | 60.0% | 23.1 to 88.2 |" in out

    assert "parser misreads: 3/6 (50.0%); parser cost 16.7 pts (true 66.7 vs parser 50.0)" in out
    assert "coax: 4/5 bare replies; benchmark parser 60.0% with 2 random fallbacks" in out
    assert "true prompt effect (coax - faithful, both scored honestly): -6.7 pts" in out
    assert "McNemar (benchmark pipeline vs coax pipeline): coax-right=2, faithful-right=1, chi2=0.0, p=1.00e+00" in out


def test_main_writes_table_and_json_files(tmp_path, monkeypatch):
    pa_dir = build_fixture(tmp_path)
    patch_module_paths(monkeypatch, tmp_path, pa_dir)

    fta.main()

    table_path = tmp_path / "out" / "faithful_true_accuracy_table.md"
    json_path = tmp_path / "out" / "faithful_true_accuracy.values.json"
    assert table_path.exists()
    assert json_path.exists()
    assert "True = high-confidence read + committed hand-labels" in table_path.read_text(encoding="utf-8")

    vals = json.loads(json_path.read_text(encoding="utf-8"))
    assert vals["n"] == 6
    assert vals["faithful_refusals"] == 1
    assert vals["parser_misreads"] == 3
    assert vals["coax_bare_replies"] == 4
    assert vals["coax_paper_parser_fallbacks"] == 2
    assert vals["mcnemar_combined"] == {
        "coax_right_faithful_wrong": 2, "faithful_right_coax_wrong": 1, "chi2": 0.0, "p": 1.0,
    }
    assert vals["_generator"] == "paper_analysis/faithful_true_accuracy.py"


def test_main_raises_systemexit_when_an_ambiguous_reply_has_no_hand_label(tmp_path, monkeypatch):
    pa_dir = build_fixture(tmp_path)
    # drop the label for index 5 (the refusal), so it becomes unlabeled-ambiguous
    labels = pd.DataFrame({"index": [2], "true_letter": ["B"]})
    labels.to_csv(pa_dir / "faithful_hand_labels.csv", index=False)
    patch_module_paths(monkeypatch, tmp_path, pa_dir)

    with pytest.raises(SystemExit, match="Missing hand-labels for ambiguous rows"):
        fta.main()
