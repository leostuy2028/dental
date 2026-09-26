# Tests for dataio/rescore_predictions.py
# Re-derives the predicted/correct columns of a result CSV from raw_response, using the
# real letter extractor (clients.parsing.extract_letter). It should change nothing when
# the file is already correct, and only rewrite the two derived columns otherwise.

import pandas as pd

import dataio.rescore_predictions as m


def test_rescore_file_skips_a_csv_missing_required_columns(tmp_path, capsys):
    path = tmp_path / "missing.csv"
    pd.DataFrame({"foo": [1]}).to_csv(path, index=False)

    changed = m.rescore_file(str(path))

    assert changed == 0
    out = capsys.readouterr().out
    assert "[skip]" in out
    assert "missing required columns" in out


def test_rescore_file_leaves_an_already_correct_file_untouched(tmp_path, capsys):
    path = tmp_path / "nochange.csv"
    df = pd.DataFrame({"raw_response": ["B", "A"], "predicted": ["B", "A"], "answer": ["B", "C"]})
    df.to_csv(path, index=False)
    before = path.read_text()

    changed = m.rescore_file(str(path))

    assert changed == 0
    assert "[ok, no change]" in capsys.readouterr().out
    assert path.read_text() == before


def test_rescore_file_fixes_a_misparsed_verbose_reply(tmp_path, capsys):
    path = tmp_path / "change.csv"
    df = pd.DataFrame({
        "raw_response": ["The correct answer is **D** (which is #38).", "B"],
        "predicted": ["A", "B"],   # first one was mis-parsed as A originally
        "answer": ["D", "B"],
    })
    df.to_csv(path, index=False)

    changed = m.rescore_file(str(path))

    assert changed == 1
    out = capsys.readouterr().out
    assert "[rescored]" in out
    assert "1 rows changed, acc 50.0 -> 100.0" in out

    after = pd.read_csv(path)
    assert after["predicted"].tolist() == ["D", "B"]
    assert after["correct"].tolist() == [True, True]
    # raw_response and answer columns are untouched
    assert after["raw_response"].tolist() == df["raw_response"].tolist()
    assert after["answer"].tolist() == ["D", "B"]


def test_rescore_file_uses_cot_parsing_when_filename_has_cot_marker(tmp_path):
    path = tmp_path / "foo_cot.csv"
    df = pd.DataFrame({"raw_response": ["Answer: C"], "predicted": ["A"], "answer": ["C"]})
    df.to_csv(path, index=False)

    changed = m.rescore_file(str(path))

    assert changed == 1
    after = pd.read_csv(path)
    assert after["predicted"].tolist() == ["C"]


def test_main_rescoring_files_named_on_the_command_line(tmp_path, monkeypatch, capsys):
    p1 = tmp_path / "a.csv"
    pd.DataFrame({"raw_response": ["Answer: A"], "predicted": ["B"], "answer": ["A"]}).to_csv(p1, index=False)
    p2 = tmp_path / "b.csv"
    pd.DataFrame({"raw_response": ["B"], "predicted": ["B"], "answer": ["B"]}).to_csv(p2, index=False)

    monkeypatch.setattr("sys.argv", ["prog", str(p1), str(p2)])
    m.main()

    out = capsys.readouterr().out
    assert "done: 1 row(s) re-scored across 2 file(s)" in out
    assert pd.read_csv(p1)["predicted"].tolist() == ["A"]


def test_main_uses_default_files_when_no_argv_given(tmp_path, monkeypatch, capsys):
    default_path = tmp_path / "default1.csv"
    pd.DataFrame({"raw_response": ["Answer: A"], "predicted": ["B"], "answer": ["A"]}).to_csv(
        default_path, index=False)

    # DEFAULT_FILES entries are made absolute, so main()'s real repo path is never used
    monkeypatch.setattr(m, "DEFAULT_FILES", [str(default_path)])
    monkeypatch.setattr("sys.argv", ["prog"])

    m.main()

    out = capsys.readouterr().out
    assert "done: 1 row(s) re-scored across 1 file(s)" in out
    assert pd.read_csv(default_path)["predicted"].tolist() == ["A"]
