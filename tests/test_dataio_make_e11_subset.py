# Tests for dataio/make_e11_subset.py
# Builds the E11 knowledge-context selection: relevant-and-missed items (split into
# "canary" FDI/counting items and other misses) plus a small relevant-and-correct
# control, sampled deterministically with a seeded random.Random.

import pandas as pd

import dataio.make_e11_subset as m


def _write_inputs(tmp_path):
    base_p = tmp_path / "base.csv"
    shuf_p = tmp_path / "shuf.parquet"
    base = pd.DataFrame({
        "index": [1, 2, 3, 4, 5],
        "category": ["Teeth1", "Patho1", "Jaw1", "Other", "HisT1"],
        "correct": [False, False, True, False, True],
    })
    base.to_csv(base_p, index=False)
    sh = pd.DataFrame({
        "index": [1, 2, 3, 4, 5],
        "question": [
            "how many teeth are visible?",   # canary (matches the FDI/counting regex)
            "what pathology is seen?",       # relevant miss, not canary
            "no issue here",                 # relevant + correct
            "no issue here",                 # irrelevant category ("Other")
            "no issue here",                 # relevant + correct (2nd candidate)
        ],
    })
    sh.to_parquet(shuf_p, index=False)
    return base_p, shuf_p


def test_main_selects_misses_and_a_correct_control(tmp_path, monkeypatch, capsys):
    base_p, shuf_p = _write_inputs(tmp_path)
    out_sel = tmp_path / "sel.csv"
    out_parquet = tmp_path / "sel.parquet"

    monkeypatch.setattr(m, "BASE", str(base_p))
    monkeypatch.setattr(m, "SHUF", str(shuf_p))
    monkeypatch.setattr(m, "OUT_SEL", str(out_sel))
    monkeypatch.setattr(m, "OUT_PARQUET", str(out_parquet))
    monkeypatch.setattr(m, "N_MISS_CANARY", 1)
    monkeypatch.setattr(m, "N_MISS_OTHER", 1)
    monkeypatch.setattr(m, "N_CORRECT", 1)

    m.main()

    sel = pd.read_csv(out_sel)
    # index 4 ("Other") is never relevant, so it can never be selected
    assert 4 not in sel["index"].tolist()
    assert len(sel) == 3
    # the miss-canary and miss-other items are always both included (pool size == request)
    assert {1, 2} <= set(sel["index"])
    assert sel.set_index("index").loc[1, "is_canary"] == True
    assert sel.set_index("index").loc[2, "is_canary"] == False
    # the one correct item drawn is deterministic given the seed
    assert sel.set_index("index").loc[3, "heuristic_correct"] == True

    parquet_out = pd.read_parquet(out_parquet)
    assert sorted(parquet_out["index"].tolist()) == sorted(sel["index"].tolist())

    printed = capsys.readouterr().out
    assert "selected 3: misses 2 + correct 1 | canary 1" in printed


def test_main_creates_missing_output_directories(tmp_path, monkeypatch):
    base_p, shuf_p = _write_inputs(tmp_path)
    out_sel = tmp_path / "nested" / "sel.csv"   # nested dir does not exist yet
    out_parquet = tmp_path / "sel.parquet"

    monkeypatch.setattr(m, "BASE", str(base_p))
    monkeypatch.setattr(m, "SHUF", str(shuf_p))
    monkeypatch.setattr(m, "OUT_SEL", str(out_sel))
    monkeypatch.setattr(m, "OUT_PARQUET", str(out_parquet))
    monkeypatch.setattr(m, "N_MISS_CANARY", 1)
    monkeypatch.setattr(m, "N_MISS_OTHER", 1)
    monkeypatch.setattr(m, "N_CORRECT", 1)

    m.main()

    assert out_sel.exists()
    assert out_parquet.exists()
