# Tests for paper_analysis/resolution_check.py
# This script checks whether downscaling images to 768px changes the model's answers
# or accuracy, by comparing committed full-res and 768px result CSVs (accuracy, a
# rescued/broke signal, and per-item agreement), then estimates the token savings.

import base64
import io

import pandas as pd
from PIL import Image

import resolution_check as rc


# ---------- gem_tokens ----------

def test_gem_tokens_small_image_is_flat_258():
    assert rc.gem_tokens(300, 300) == 258
    assert rc.gem_tokens(384, 384) == 258


def test_gem_tokens_scales_for_larger_images():
    # ceil(1000/768)=2, ceil(500/768)=1 -> 2*1*258
    assert rc.gem_tokens(1000, 500) == 516


# ---------- signal ----------

def test_signal_counts_rescued_and_broke():
    noctx = pd.DataFrame({"correct": [False, True, False, True]}, index=[1, 2, 3, 4])
    ctx = pd.DataFrame({"correct": [True, True, False, False]}, index=[1, 2, 3, 4])
    # item1: F->T rescued, item2: T->T same, item3: F->F same, item4: T->F broke
    out = rc.signal(noctx, ctx, [1, 2, 3, 4])
    assert out["noctx_acc"] == 50.0
    assert out["ctx_acc"] == 50.0
    assert out["rescued"] == 1
    assert out["broke"] == 1
    assert out["net"] == 0


# ---------- load ----------

def test_load_reads_csv_and_sets_index(tmp_path, monkeypatch):
    sub = tmp_path / "sub"
    sub.mkdir()
    csv_path = sub / "fixture.csv"
    pd.DataFrame({"index": [1, 2], "correct": [True, False]}).to_csv(csv_path, index=False)
    monkeypatch.setitem(rc.F, "full_noctx", "sub/fixture.csv")
    df = rc.load(str(tmp_path), "full_noctx")
    assert list(df.index) == [1, 2]
    assert list(df["correct"]) == [True, False]


# ---------- main() end to end ----------

FIXED = pd.DataFrame({
    "index": [1, 2, 3, 4, 5, 6],
    "correct": [True, False, True, True, False, True],
    "predicted": ["A", "B", "A", "C", "B", "A"],
    "category": ["Teeth", "Patho", "HisT", "Jaw", "SumRec", "Teeth,HisT"],
})


def make_png_b64(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(10, 20, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_main_end_to_end_prints_all_sections(monkeypatch, capsys):
    # Every pd.read_csv call in main() (there are several hard-coded paths besides the
    # F dict) is intercepted here and given the same tiny 6-row fixture, regardless of
    # which file was asked for -- this avoids touching any real file in results/, while
    # still exercising every line main() can reach (the optional p1024/whole-491 blocks
    # only run because their paths happen to exist for real on disk; read_csv itself
    # never touches disk content here since it is replaced below).
    def fake_read_csv(path, *a, **k):
        return FIXED.copy()

    fake_parquet_df = pd.DataFrame({"image": [make_png_b64(50, 50), make_png_b64(900, 600)]})

    def fake_read_parquet(path, *a, **k):
        return fake_parquet_df.copy()

    monkeypatch.setattr(rc.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(rc.pd, "read_parquet", fake_read_parquet)

    rc.main()
    out = capsys.readouterr().out

    assert "768px vs full-res — §5.4 E11 reproduction (6 items)" in out

    assert "noctx : full-res  66.7%   768px  66.7%   (Δ +0.0)" in out
    assert "ctx   : full-res  66.7%   768px  66.7%   (Δ +0.0)" in out

    assert "full  : 66.7% -> 66.7%   rescued 0 / broke 0 (net +0)" in out
    assert "px768 : 66.7% -> 66.7%   rescued 0 / broke 0 (net +0)" in out

    assert "noctx : 100% (6/6)" in out
    assert "Teeth  100% (n=2)" in out
    assert "Patho  100% (n=1)" in out
    assert "HisT   100% (n=2)" in out
    assert "Jaw    100% (n=1)" in out
    assert "ctx   : 100% (6/6)" in out

    # optional blocks -- these only appear if the referenced (real, committed) files
    # happen to exist on this machine; they do in this repo, so both should be present
    if "E11 no-context agreement" in out:
        assert "1024px: 100% (6/6)   acc 66.7%" in out
    if "WHOLE-491 (unbiased)" in out:
        assert "accuracy: full-res 66.7%  ->  768px 66.7%" in out

    assert "Est. Gemini image tokens over the 2 items: full 774 -> 768px 516 (1.5x cheaper images)" in out


def test_main_skips_optional_blocks_when_files_missing(monkeypatch, capsys):
    def fake_read_csv(path, *a, **k):
        return FIXED.copy()

    fake_parquet_df = pd.DataFrame({"image": [make_png_b64(50, 50)]})

    def fake_read_parquet(path, *a, **k):
        return fake_parquet_df.copy()

    monkeypatch.setattr(rc.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(rc.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(rc.os.path, "exists", lambda p: False)

    rc.main()
    out = capsys.readouterr().out

    # main() still ran the mandatory sections
    assert "Accuracy:" in out
    assert "Signal (noctx -> +primer):" in out
    # but both optional (file-existence-gated) sections were skipped
    assert "E11 no-context agreement" not in out
    assert "WHOLE-491 (unbiased)" not in out
