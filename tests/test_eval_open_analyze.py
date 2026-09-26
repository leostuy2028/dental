# Tests for eval_open/analyze.py
# This file reads the E-open-1 isolation CSV (one score per index/judge/rubric/variant)
# and reports the "coordinate bonus" (score with coordinates minus score without),
# paired per item, with a Wilcoxon signed-rank test.

import pandas as pd

import eval_open.analyze as an


def write_isolation_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_main_prints_one_line_per_judge_rubric_ref_type_with_the_paired_bonus(
    tmp_path, monkeypatch, capsys
):
    csv_path = tmp_path / "isolation.csv"
    write_isolation_csv(csv_path, [
        dict(index=1, ref_type="coord_ref", judge="gpt-4o", rubric="original", variant="prose", score=0.5),
        dict(index=1, ref_type="coord_ref", judge="gpt-4o", rubric="original", variant="coords", score=0.9),
        dict(index=2, ref_type="coord_ref", judge="gpt-4o", rubric="original", variant="prose", score=0.4),
        dict(index=2, ref_type="coord_ref", judge="gpt-4o", rubric="original", variant="coords", score=0.8),
        dict(index=1, ref_type="coord_ref", judge="gpt-4o", rubric="rephrased", variant="prose", score=0.9),
        dict(index=1, ref_type="coord_ref", judge="gpt-4o", rubric="rephrased", variant="coords", score=0.9),
        dict(index=2, ref_type="coord_ref", judge="gpt-4o", rubric="rephrased", variant="prose", score=0.8),
        dict(index=2, ref_type="coord_ref", judge="gpt-4o", rubric="rephrased", variant="coords", score=0.8),
        dict(index=3, ref_type="prose_ref", judge="gpt-4o", rubric="original", variant="prose", score=0.6),
        dict(index=3, ref_type="prose_ref", judge="gpt-4o", rubric="original", variant="coords", score=0.6),
    ])
    monkeypatch.setattr(an, "IN", str(csv_path))

    an.main()
    out = capsys.readouterr().out

    assert "paired items: 5" in out
    # original coord_ref: prose mean 0.45, coords mean 0.85, bonus +0.400
    assert "gpt-4o   original   coord_ref     2  0.450  0.850  +0.400" in out
    # rephrased coord_ref: format-invariant, no bonus at all
    assert "gpt-4o   rephrased  coord_ref     2  0.850  0.850  +0.000" in out
    # a rubric/ref_type combo with no rows at all (rephrased x prose_ref) is skipped
    assert "rephrased  prose_ref" not in out
    assert "=== HEADLINE: coordinate bonus on coord_ref items (mean over judges) ===" in out
    assert "original  : mean bonus = +0.400  (n=2)" in out
    assert "rephrased : mean bonus = +0.000  (n=2)" in out
    # bias removed = orig - reph = 0.400, and 100% of the original bonus was removed
    assert "bias removed by rephrase: +0.400 (100% of the original bonus)" in out


def test_main_prints_a_blank_line_instead_of_dividing_by_a_zero_original_bonus(
    tmp_path, monkeypatch, capsys
):
    # when the ORIGINAL rubric's own coord_ref bonus is exactly zero, printing
    # "(1 - reph/orig)" would divide by zero; the code guards this with
    # `... if orig else ""`, so it should print nothing instead of crashing.
    csv_path = tmp_path / "isolation.csv"
    write_isolation_csv(csv_path, [
        dict(index=1, ref_type="coord_ref", judge="gpt-4o", rubric="original", variant="prose", score=0.5),
        dict(index=1, ref_type="coord_ref", judge="gpt-4o", rubric="original", variant="coords", score=0.5),
        dict(index=1, ref_type="coord_ref", judge="gpt-4o", rubric="rephrased", variant="prose", score=0.5),
        dict(index=1, ref_type="coord_ref", judge="gpt-4o", rubric="rephrased", variant="coords", score=0.5),
    ])
    monkeypatch.setattr(an, "IN", str(csv_path))

    an.main()
    out = capsys.readouterr().out

    assert "original  : mean bonus = +0.000  (n=1)" in out
    assert "bias removed by rephrase" not in out


def test_main_handles_more_than_one_judge(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "isolation.csv"
    write_isolation_csv(csv_path, [
        dict(index=1, ref_type="coord_ref", judge="gpt-4o", rubric="original", variant="prose", score=0.2),
        dict(index=1, ref_type="coord_ref", judge="gpt-4o", rubric="original", variant="coords", score=0.6),
        dict(index=1, ref_type="coord_ref", judge="gemini", rubric="original", variant="prose", score=0.3),
        dict(index=1, ref_type="coord_ref", judge="gemini", rubric="original", variant="coords", score=0.3),
        dict(index=1, ref_type="coord_ref", judge="gpt-4o", rubric="rephrased", variant="prose", score=0.6),
        dict(index=1, ref_type="coord_ref", judge="gpt-4o", rubric="rephrased", variant="coords", score=0.6),
        dict(index=1, ref_type="coord_ref", judge="gemini", rubric="rephrased", variant="prose", score=0.3),
        dict(index=1, ref_type="coord_ref", judge="gemini", rubric="rephrased", variant="coords", score=0.3),
    ])
    monkeypatch.setattr(an, "IN", str(csv_path))

    an.main()
    out = capsys.readouterr().out

    # both judges get their own printed row, in sorted order (gemini before gpt-4o)
    gemini_line = "gemini   original   coord_ref     1  0.300  0.300  +0.000"
    gpt_line = "gpt-4o   original   coord_ref     1  0.200  0.600  +0.400"
    assert gemini_line in out
    assert gpt_line in out
    assert out.index(gemini_line) < out.index(gpt_line)
