# Tests for paper_analysis/ensemble_delta.py
# This is a readout for a "two models deliberate together" probe: it compares an
# ensemble's score against each single model's baseline and the per-item oracle
# (best of the two drafts), and classifies whether the ensemble kept its own draft,
# took the other model's, or wrote something new. Writes a table + JSON of numbers.

import json

import pandas as pd

import ensemble_delta as ed


def build_fixture(base, gem_scores, gpt_scores, ens_scores, gem_answers, gpt_answers, ens_finals):
    n = len(gem_scores)
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "results" / "open").mkdir(parents=True, exist_ok=True)
    idx = list(range(n))

    pd.DataFrame({"index": idx, "question": [f"q{i}" for i in idx],
                  "answer": [f"ref{i}" for i in idx]}).to_parquet(base / "data" / "open_ended.parquet")
    pd.DataFrame({"index": idx, "score": gem_scores}).to_csv(
        base / "results" / "open" / "batched_gemini35_plain578_scores.csv", index=False)
    pd.DataFrame({"index": idx, "score": gpt_scores}).to_csv(
        base / "results" / "open" / "batched_gpt5mini_scores.csv", index=False)
    pd.DataFrame({"index": idx, "score": ens_scores}).to_csv(
        base / "results" / "open" / "ensemble_gemjudge_concrete_scores.csv", index=False)
    pd.DataFrame({"index": idx, "answer": gem_answers}).to_csv(
        base / "results" / "open" / "batched_gemini35_plain578_answers.csv", index=False)
    pd.DataFrame({"index": idx, "answer": gpt_answers}).to_csv(
        base / "results" / "open" / "batched_gpt5mini_answers.csv", index=False)
    pd.DataFrame({"index": idx, "final": ens_finals}).to_csv(
        base / "results" / "open" / "ensemble_gemjudge_concrete_answers.csv", index=False)


# ---------- norm() ----------

def test_norm_strips_punctuation_and_case():
    assert ed.norm("Hello, World!!  ") == "hello world"


def test_norm_handles_none():
    assert ed.norm(None) == "none"


# ---------- ci95() ----------

def test_ci95_matches_hand_check():
    d = pd.Series([0.1, 0.2, 0.3, 0.1])
    mean, lo, hi = ed.ci95(d)
    assert (round(float(mean), 1), round(float(lo), 1), round(float(hi), 1)) == (17.5, 8.1, 26.9)


# ---------- main() end to end ----------

def test_main_writes_table_and_reports_behaviour_split(tmp_path, monkeypatch, capsys):
    build_fixture(
        tmp_path,
        gem_scores=[0.5, 0.6, 0.9, 0.2, 0.4],
        gpt_scores=[0.7, 0.5, 0.3, 0.8, 0.4],
        ens_scores=[0.7, 0.6, 0.9, 0.8, 0.5],
        gem_answers=["gem answer 0", "gem answer 1", "gem answer 2", "gem answer 3", "gem answer 4"],
        gpt_answers=["gpt answer 0", "gpt answer 1", "gpt answer 2", "gpt answer 3", "gpt answer 4"],
        ens_finals=["gem answer 0", "gpt answer 1", "totally new answer", "GEM ANSWER 3!", "brand new text"],
    )
    monkeypatch.setattr(ed, "REPO", str(tmp_path))
    monkeypatch.setattr(ed, "OUT_DIR", str(tmp_path / "_generated"))

    ed.main()

    printed = capsys.readouterr().out
    assert "| gemini-3.5-flash (baseline draft) | 52.0% |" in printed
    assert "| gpt-5-mini (second draft) | 54.0% |" in printed
    assert "| per-item oracle (better of the two) | 68.0% |" in printed
    assert "| **ensemble** (gemini re-reads with both drafts) | **70.0%** |" in printed
    assert "delta vs gemini baseline: +18.0 pts  [-3.8, +39.8]" in printed
    assert "oracle headroom was +16.0 pts; captured 112.5%" in printed
    assert "beat BOTH drafts: 1   worse than BOTH: 0" in printed
    assert "behaviour: kept own 2 | took other 1 | rewrote 2" in printed

    vals = json.loads((tmp_path / "_generated" / "ensemble_delta.values.json").read_text(encoding="utf-8"))
    assert vals["n"] == 5
    assert vals["gemini35_baseline"] == 52.0
    assert vals["gpt5mini"] == 54.0
    assert vals["per_item_oracle"] == 68.0
    assert vals["ensemble"] == 70.0
    assert vals["delta_vs_gemini"] == [18.0, -3.8, 39.8]
    assert vals["oracle_gap_pts"] == 16.0
    assert vals["captured_of_oracle_gap_pct"] == 112.5
    assert vals["items_beat_both_drafts"] == 1
    assert vals["items_worse_than_both_drafts"] == 0
    assert vals["behaviour"] == {"kept_own_draft": 2, "took_other_draft": 1, "rewrote": 2}
    assert "Not prompt-matched to the baseline" in vals["caveat"]
    assert vals["_generator"] == "paper_analysis/ensemble_delta.py"

    table = (tmp_path / "_generated" / "ensemble_delta_table.md").read_text(encoding="utf-8")
    assert "| System (concrete questions, n=5) | score |" in table


def test_main_handles_zero_oracle_gap(tmp_path, monkeypatch, capsys):
    # CURRENT BEHAVIOR: when the baseline already equals the oracle (gap == 0), the
    # "captured_of_oracle_gap_pct" is left as None instead of, say, 0 or 100 -- the
    # code explicitly guards the division with "if gap else None".
    build_fixture(
        tmp_path,
        gem_scores=[0.8, 0.8, 0.8],
        gpt_scores=[0.5, 0.5, 0.5],
        ens_scores=[0.8, 0.8, 0.8],
        gem_answers=["a0", "a1", "a2"],
        gpt_answers=["b0", "b1", "b2"],
        ens_finals=["a0", "a1", "a2"],
    )
    monkeypatch.setattr(ed, "REPO", str(tmp_path))
    monkeypatch.setattr(ed, "OUT_DIR", str(tmp_path / "_generated"))

    ed.main()

    printed = capsys.readouterr().out
    assert "oracle headroom was +0.0 pts; captured None%" in printed
    assert "behaviour: kept own 3 | took other 0 | rewrote 0" in printed

    vals = json.loads((tmp_path / "_generated" / "ensemble_delta.values.json").read_text(encoding="utf-8"))
    assert vals["oracle_gap_pts"] == 0.0
    assert vals["captured_of_oracle_gap_pct"] is None
