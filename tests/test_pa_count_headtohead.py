# Tests for paper_analysis/count_headtohead.py
# Head-to-head tooth-COUNTING accuracy: compares the trained detector and a few VLMs
# against a reference tooth count parsed out of the benchmark's own ground-truth
# answers, and prints/writes a table of "how often each system commits to a number"
# and "how close its number is" (exact / within-1 / mean error).

import json

import pandas as pd

import count_headtohead as ch


# ---------- gt_count() ----------

def test_gt_count_extracts_number():
    assert ch.gt_count("Findings show 24 teeth are visualized in this OPG.") == 24
    assert ch.gt_count("There are 20 teeth present.") == 20
    assert ch.gt_count("18 teeth detected.") == 18


def test_gt_count_none_when_no_match():
    assert ch.gt_count("no count here") is None


# ---------- parse_count() ----------

def test_parse_count_tier1_explicit_visualized_phrase():
    assert ch.parse_count("The image shows 24 teeth are visualized clearly.", False) == 24


def test_parse_count_tier2_dentition_of_phrase():
    assert ch.parse_count("This OPG demonstrates 22 teeth.", False) == 22


def test_parse_count_tier3_bare_teeth_phrase():
    assert ch.parse_count("total of 19 teeth", False) == 19


def test_parse_count_tier4_bare_number_for_howmany_question():
    assert ch.parse_count("23", True) == 23
    assert ch.parse_count("there are 25", True) == 25


def test_parse_count_no_match_gives_none():
    assert ch.parse_count("no number stated", False) is None


def test_parse_count_bare_number_ignored_when_not_howmany():
    # tier 4 only fires when howmany=True
    assert ch.parse_count("23", False) is None


def test_parse_count_avoids_subcount_like_wisdom_teeth():
    # "3 wisdom teeth" should NOT be read as the total count, since "wisdom" is
    # deliberately excluded from the allowed descriptor list in NT.
    assert ch.parse_count("3 wisdom teeth are present along with the rest", False) is None


# ---------- score() ----------

def test_score_computes_exact_within1_and_mae():
    per_img = {"img1": 24, "img2": 21, "img3": 18}
    ref = {"img1": 24, "img2": 20, "img3": 18}
    covered, exact, within1, mae = ch.score(per_img, ref)
    assert covered == 3
    assert exact == 2
    assert within1 == 3
    assert round(mae, 4) == round(1 / 3, 4)


def test_score_no_covered_images_returns_zeros_and_nan():
    covered, exact, within1, mae = ch.score({}, {"img1": 24})
    assert (covered, exact, within1) == (0, 0, 0)
    assert mae != mae  # nan != nan


def test_score_ignores_images_not_in_reference():
    result = ch.score({"imgX": 10}, {"img1": 24})
    assert result == (0, 0, 0, float("nan")) or result[:3] == (0, 0, 0)


# ---------- main() end to end ----------

def build_main_fixture(base):
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "results" / "open").mkdir(parents=True, exist_ok=True)
    (base / "results" / "detector").mkdir(parents=True, exist_ok=True)

    op = pd.DataFrame({
        "index": [0, 1, 2, 3],
        "image_name": ["img1", "img1", "img2", "img3"],
        "question": ["How many teeth are visualized?", "Findings",
                     "How many teeth are visualized?", "How many teeth are visualized?"],
        "answer": ["24 teeth are visualized in total.", "24 teeth are visualized in the OPG.",
                   "20 teeth are present.", "18 teeth are detected."],
    })
    op.to_parquet(base / "data" / "open_ended.parquet")

    modelA = pd.DataFrame({"index": [0, 2, 3], "answer": [
        "I count 24 teeth are visualized total.",
        "I see 21 teeth are present.",
        "18 teeth are detected here."]})
    modelA.to_csv(base / "results" / "open" / "modelA.csv", index=False)

    modelB = pd.DataFrame({"index": [0], "answer": ["24 teeth are visualized clearly."]})
    modelB.to_csv(base / "results" / "open" / "modelB.csv", index=False)

    modelC = pd.DataFrame({"index": [0, 2, 3], "answer": ["I cannot tell.", "Unclear.", "No answer."]})
    modelC.to_csv(base / "results" / "open" / "modelC.csv", index=False)

    det = pd.DataFrame({"image": ["img1", "img2", "img3"], "detector_count": [24, 19, 18],
                         "ref_count": [24, 20, 18]})
    det.to_csv(base / "results" / "detector" / "mmoral_counts.csv", index=False)


def test_main_with_detector_present(tmp_path, monkeypatch, capsys):
    build_main_fixture(tmp_path)
    monkeypatch.setattr(ch, "OPEN", str(tmp_path / "data" / "open_ended.parquet"))
    monkeypatch.setattr(ch, "DETECTOR_CSV_CANDIDATES", [str(tmp_path / "results" / "detector" / "mmoral_counts.csv")])
    monkeypatch.setattr(ch, "MODELS", [
        ("Model A", str(tmp_path / "results" / "open" / "modelA.csv")),
        ("Model B", str(tmp_path / "results" / "open" / "modelB.csv")),
        ("Model C", str(tmp_path / "results" / "open" / "modelC.csv")),
    ])
    # main()'s out_dir is anchored to the module's OWN __file__, not a constant we
    # can monkeypatch directly -- so fake __file__ itself to redirect the writes.
    monkeypatch.setattr(ch, "__file__", str(tmp_path / "count_headtohead.py"))

    ch.main()

    printed = capsys.readouterr().out
    assert "Reference set: 3 MMOral images with a dentist-confirmed count." in printed
    assert "Model A        3/3   |     67%     100% |     67%     100%   0.33" in printed
    assert "Model B        1/3   |    100%     100% |     33%      33%   0.00" in printed
    assert "Model C        0/3   |    (never commits to a number)" in printed
    assert "Detector       3/3   |     67%     100% |     67%     100%   0.33" in printed
    assert "Same-subset check" in printed

    out_dir = tmp_path / "_generated"
    table = (out_dir / "count_headtohead_table.md").read_text(encoding="utf-8")
    assert "| Model A | 3 / 3 | 67% / 100% |" in table
    assert "| Model B | 1 / 3 | 33% / 33% |" in table
    assert "| Model C | 0 / 3 | — |" in table
    assert "| **Tooth detector** | **3 / 3** | **67% / 100%** |" in table

    vals = json.loads((out_dir / "count_headtohead.values.json").read_text(encoding="utf-8"))
    assert vals["n_reference_images"] == 3
    assert vals["rows"]["Model A"] == {"states_count": 3, "exact_pct": 66.7, "within1_pct": 100.0}
    assert vals["rows"]["Model B"] == {"states_count": 1, "exact_pct": 33.3, "within1_pct": 33.3}
    assert vals["rows"]["Model C"] == {"states_count": 0, "exact_pct": 0.0, "within1_pct": 0.0}
    assert vals["rows"]["Tooth detector"] == {"states_count": 3, "exact_pct": 66.7, "within1_pct": 100.0}


def test_main_skips_model_whose_csv_is_missing(tmp_path, monkeypatch, capsys):
    build_main_fixture(tmp_path)
    monkeypatch.setattr(ch, "OPEN", str(tmp_path / "data" / "open_ended.parquet"))
    monkeypatch.setattr(ch, "DETECTOR_CSV_CANDIDATES", [str(tmp_path / "results" / "detector" / "mmoral_counts.csv")])
    monkeypatch.setattr(ch, "MODELS", [
        ("Model A", str(tmp_path / "results" / "open" / "modelA.csv")),
        ("Ghost Model", str(tmp_path / "results" / "open" / "does_not_exist.csv")),
    ])
    monkeypatch.setattr(ch, "__file__", str(tmp_path / "count_headtohead.py"))

    ch.main()

    printed = capsys.readouterr().out
    assert "Model A" in printed
    assert "Ghost Model" not in printed


def test_main_without_detector_uses_hardcoded_placeholder_row(tmp_path, monkeypatch, capsys):
    # CURRENT BEHAVIOR (looks like a bug): when no detector CSV is found, `det` is
    # falsy, so the fairness check and the markdown "Tooth detector" row are
    # correctly skipped -- but `results.append(("Detector",) + (score(det, ref) if
    # det else (86, 44, 74, 0.87)))` still unconditionally appends a hardcoded
    # placeholder tuple (86, 44, 74, 0.87) taken from some other run, and the
    # terminal table prints it as if it were real, with nonsensical percentages
    # (e.g. "2200%") once N is small.
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "open").mkdir(parents=True, exist_ok=True)
    op = pd.DataFrame({
        "index": [0, 1],
        "image_name": ["img1", "img2"],
        "question": ["How many teeth are visualized?", "How many teeth are visualized?"],
        "answer": ["24 teeth are visualized in total.", "20 teeth are present."],
    })
    op.to_parquet(tmp_path / "data" / "open_ended.parquet")
    modelA = pd.DataFrame({"index": [0, 1], "answer": ["24 teeth are visualized.", "20 teeth are present here."]})
    modelA.to_csv(tmp_path / "results" / "open" / "modelA.csv", index=False)

    monkeypatch.setattr(ch, "OPEN", str(tmp_path / "data" / "open_ended.parquet"))
    monkeypatch.setattr(ch, "DETECTOR_CSV_CANDIDATES", [str(tmp_path / "results" / "detector" / "missing.csv")])
    monkeypatch.setattr(ch, "MODELS", [("Model A", str(tmp_path / "results" / "open" / "modelA.csv"))])
    monkeypatch.setattr(ch, "__file__", str(tmp_path / "count_headtohead.py"))

    ch.main()

    printed = capsys.readouterr().out
    assert "Model A        2/2   |    100%     100% |    100%     100%   0.00" in printed
    assert "Detector      86/2   |     51%      86% |   2200%    3700%   0.87" in printed
    # fairness check and markdown "Tooth detector" row are skipped when det is empty
    assert "Same-subset check" not in printed
    table = (tmp_path / "_generated" / "count_headtohead_table.md").read_text(encoding="utf-8")
    assert "Tooth detector" not in table
