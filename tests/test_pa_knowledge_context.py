# Tests for paper_analysis/knowledge_context.py
# This script compares a model's accuracy with vs without an in-context reading
# primer on paired items, and reports how often the primer RESCUES a wrong answer
# vs BREAKS a right one (the "signal" that tells real learning from answer-churn).

import pandas as pd

import knowledge_context as kc


# ---------- mcnemar ----------

def test_mcnemar_zero_flips_is_one():
    assert kc.mcnemar(0, 0) == 1.0


def test_mcnemar_known_values():
    # verified by actually running kc.mcnemar with these inputs
    assert kc.mcnemar(3, 1) == 0.625
    assert kc.mcnemar(5, 5) == 1.0


# ---------- wilson ----------

def test_wilson_zero_n_is_zero_zero():
    assert kc.wilson(0, 0) == (0.0, 0.0)


def test_wilson_known_values():
    lo, hi = kc.wilson(10, 10)
    assert round(lo, 4) == 72.246
    assert hi == 100.0


# ---------- main() end to end ----------

def _write_kc_fixture(tmp_path):
    # 6 paired items. Indices 1/4/6 are misses without context (~ac); of those,
    # 1 and 4 get rescued by the primer, 6 does not. Index 5 is right without
    # context and BREAKS with the primer (the one "broke" flip).
    sel = pd.DataFrame({
        "index": [1, 2, 3, 4, 5, 6],
        "category": ["Teeth", "Patho", "HisT", "Jaw", "Teeth", "Patho"],
        "is_canary": [True, False, False, True, False, False],
    })
    noctx = pd.DataFrame({
        "index": [1, 2, 3, 4, 5, 6],
        "correct": [False, False, True, False, True, True],
    })
    ctx = pd.DataFrame({
        "index": [1, 2, 3, 4, 5, 6],
        "correct": [True, False, True, True, True, False],
    })
    sel_path = tmp_path / "sel.csv"
    noctx_path = tmp_path / "noctx.csv"
    ctx_path = tmp_path / "ctx.csv"
    sel.to_csv(sel_path, index=False)
    noctx.to_csv(noctx_path, index=False)
    ctx.to_csv(ctx_path, index=False)
    return noctx_path, ctx_path, sel_path


def test_main_prints_the_full_e11_screen(tmp_path, monkeypatch, capsys):
    noctx_path, ctx_path, sel_path = _write_kc_fixture(tmp_path)
    # NOCTX/CTX/SEL are joined onto a repo path inside main() via os.path.join;
    # os.path.join discards the first part when the second is already absolute,
    # so pointing these constants at absolute tmp_path files works without
    # touching how main() computes its repo path.
    monkeypatch.setattr(kc, "NOCTX", str(noctx_path))
    monkeypatch.setattr(kc, "CTX", str(ctx_path))
    monkeypatch.setattr(kc, "SEL", str(sel_path))

    kc.main()
    out = capsys.readouterr().out

    assert "E11 knowledge-context screen — gemini-3.5-flash, 6 items (paired)" in out
    assert "overall acc: no-context 50.0%  ->  +primer 66.7%" in out
    assert "paired flips: rescued 2, broke 1, net +1, McNemar p=1.000" in out
    assert ">>> RESCUE rate (of 3 misses): 67% [21-94]" in out
    assert ">>> BREAK  rate (of 3 correct): 33% [6-79]" in out
    assert "SIGNAL = rescue - break = +33 pts" in out
    assert "Teeth  rescued 1/1 (100%)" in out
    assert "Patho  rescued 0/1 (0%)" in out
    assert "Jaw    rescued 1/1 (100%)" in out
    # HisT has zero misses (item 3 is correct without context), so it's skipped
    assert "HisT" not in out
    assert "canary (FDI/count Qs whose answer is in the primer): misses 2, rescued 2" in out
