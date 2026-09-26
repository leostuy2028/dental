# Tests for dataio/build_whole491.py
# Merges committed raw model outputs into the two canonical GPT-4o whole-491 result
# files (faithful and coax), re-deriving predicted/correct with the real extractors
# (clients.parsing.extract_letter, utils.vlmeval_parse.faithful_predict). Rows whose
# index is in BLANK_OPTION take raw_response from the blanks38-none run; everything
# else comes from the original whole491 run. REPO is a module-level constant computed
# at import time, so we can monkeypatch it directly to point everything at tmp_path.

import json
import os

import pandas as pd

import dataio.build_whole491 as m


def _write_inputs(tmp_path):
    repo = str(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    opts = pd.DataFrame({
        "index": [2, 41],                     # 41 is one of the real BLANK_OPTION indices
        "file_name": ["f2.jpg", "f41.jpg"],
        "category": ["C2", "C41"],
        "question": ["Q2", "Q41"],
        "option1": ["optA2", "None"], "option2": ["optB2", "optB41"],
        "option3": ["optC2", "optC41"], "option4": ["optD2", "optD41"],
        "answer": ["B", "A"],
    })
    opts.to_parquet(data_dir / "closed_ended.parquet", index=False)

    (tmp_path / m.SUP).mkdir(parents=True)
    (tmp_path / m.OUT).mkdir(parents=True, exist_ok=True)
    for cfg in m.RUNS.values():
        whole = pd.DataFrame({"index": [2, 41], "raw_response": ["(B)", "WRONG-FROM-WHOLE"]})
        whole.to_csv(tmp_path / cfg["whole491_nan"], index=False)
        # only index 41 (a real BLANK_OPTION member) gets a blanks38-none row
        b38 = pd.DataFrame({"index": [41], "raw_response": ["The answer is A"]})
        b38.to_csv(tmp_path / cfg["blanks38_none"], index=False)
    return repo, opts


def test_build_prefers_blanks38_raw_response_for_blank_option_indices(tmp_path, monkeypatch):
    repo, opts = _write_inputs(tmp_path)
    monkeypatch.setattr(m, "REPO", repo)
    opts_idx = opts.set_index("index", drop=False)

    df = m.build("faithful", m.RUNS["faithful"], opts_idx)

    row2 = df[df["index"] == 2].iloc[0]
    row41 = df[df["index"] == 41].iloc[0]
    assert row2["source"] == "whole491"
    assert row2["raw_response"] == "(B)"
    assert row41["source"] == "blanks38-none"
    assert row41["raw_response"] == "The answer is A"   # NOT "WRONG-FROM-WHOLE"
    assert row2["predicted"] == "B" and row2["correct"] == True
    assert row41["predicted"] == "A" and row41["correct"] == True


def test_build_coax_mode_uses_the_bare_letter_extractor(tmp_path, monkeypatch):
    repo, opts = _write_inputs(tmp_path)
    monkeypatch.setattr(m, "REPO", repo)
    opts_idx = opts.set_index("index", drop=False)

    df = m.build("coax", m.RUNS["coax"], opts_idx)

    assert df.set_index("index")["predicted"].to_dict() == {2: "B", 41: "A"}
    assert df["used_fallback"].tolist() == [False, False]
    assert df["prompt_mode"].unique().tolist() == ["coax"]


def test_main_writes_both_result_files_with_meta_sidecars(tmp_path, monkeypatch, capsys):
    repo, _ = _write_inputs(tmp_path)
    monkeypatch.setattr(m, "REPO", repo)

    m.main()

    for cfg in m.RUNS.values():
        out_path = tmp_path / cfg["out"]
        assert out_path.exists()
        meta = json.loads((tmp_path / (cfg["out"] + ".meta.json")).read_text())
        assert meta["n"] == 2
        assert meta["model"] == "gpt-4o-2024-11-20"
        assert meta["dataset"] == "closed_ended (canonical, blanks = 'None')"

    printed = capsys.readouterr().out
    assert "faithful " in printed and "acc=100.0%" in printed
    assert "coax     " in printed
    # the accuracy on just the 38 'None' items is reported separately
    assert "on the 38 'None' items: 100.0%" in printed
