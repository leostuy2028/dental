# Tests for dataio/check_na_roundtrip.py
# Scans every CSV under REPO for cells that are a real non-empty string with
# keep_default_na=False but come back as NaN under pandas' defaults (the "None" ->
# NaN trap). Files listed in RECORD_ONLY are reported but don't fail the scan.

import os

import dataio.check_na_roundtrip as m


def test_scan_flags_a_plain_csv_that_loses_the_string_none(tmp_path, monkeypatch):
    (tmp_path / "bad.csv").write_text("col1,col2\na,None\nb,c\n")
    monkeypatch.setattr(m, "REPO", str(tmp_path))

    bad, benign = m.scan()

    assert bad == [("bad.csv", "col2", 1, {"None": 1})]
    assert benign == []


def test_scan_treats_a_record_only_path_as_benign(tmp_path, monkeypatch):
    # the RECORD_ONLY dict keys are specific relative paths, so the file must be created
    # at exactly that path to match
    record_dir = tmp_path / "results" / "closed_ended"
    record_dir.mkdir(parents=True)
    (record_dir / "blanks38_manifest.csv").write_text("col1,col2\na,None\n")
    monkeypatch.setattr(m, "REPO", str(tmp_path))

    bad, benign = m.scan()

    assert bad == []
    assert benign == [("results/closed_ended/blanks38_manifest.csv", "col2", 1, {"None": 1})]


def test_scan_ignores_a_nested_superseded_dir_and_unparsable_csvs(tmp_path, monkeypatch):
    # nested (not top-level) _superseded: the "/_superseded/" substring check matches
    superseded_dir = tmp_path / "results" / "_superseded"
    superseded_dir.mkdir(parents=True)
    (superseded_dir / "skip.csv").write_text("a,b\n1,None\n")

    # ragged row count makes pandas raise, which scan() catches and skips
    (tmp_path / "bad_parse.csv").write_text("a,b\n1,2,3\n")

    monkeypatch.setattr(m, "REPO", str(tmp_path))

    bad, benign = m.scan()

    assert bad == []
    assert benign == []


def test_scan_does_not_skip_a_top_level_superseded_dir(tmp_path, monkeypatch):
    # CURRENT BEHAVIOR (looks like a bug): the skip test is `"/_superseded/" in rel`,
    # but rel has no leading slash, so a _superseded folder directly under REPO does NOT
    # match the check and its files are scanned like any other CSV.
    superseded_dir = tmp_path / "_superseded"
    superseded_dir.mkdir()
    (superseded_dir / "skip.csv").write_text("a,b\n1,None\n")
    monkeypatch.setattr(m, "REPO", str(tmp_path))

    bad, benign = m.scan()

    assert bad == [("_superseded/skip.csv", "b", 1, {"None": 1})]
    assert benign == []


def test_scan_never_sees_files_inside_a_dot_directory(tmp_path, monkeypatch):
    # CURRENT BEHAVIOR: glob's "**" does not descend into dot-directories like .venv at
    # all (this is Python's glob, not the module's "/.venv/" check), so such files never
    # appear in either list regardless of their content.
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "skip.csv").write_text("a,b\n1,None\n")
    monkeypatch.setattr(m, "REPO", str(tmp_path))

    bad, benign = m.scan()

    assert bad == []
    assert benign == []


def test_scan_finds_no_problem_in_a_clean_csv(tmp_path, monkeypatch):
    (tmp_path / "clean.csv").write_text("col1,col2\na,b\n")
    monkeypatch.setattr(m, "REPO", str(tmp_path))

    bad, benign = m.scan()

    assert bad == [] and benign == []


def test_main_returns_1_and_prints_fail_when_a_bad_file_exists(tmp_path, monkeypatch, capsys):
    (tmp_path / "bad.csv").write_text("col1,col2\na,None\n")
    monkeypatch.setattr(m, "REPO", str(tmp_path))

    rc = m.main()

    assert rc == 1
    printed = capsys.readouterr().out
    assert "FAIL: these CSVs hold values" in printed
    assert "bad.csv  column 'col2': 1 cells -> {'None': 1}" in printed


def test_main_returns_0_and_prints_ok_when_nothing_is_lost(tmp_path, monkeypatch, capsys):
    (tmp_path / "clean.csv").write_text("col1,col2\na,b\n")
    monkeypatch.setattr(m, "REPO", str(tmp_path))

    rc = m.main()

    assert rc == 0
    printed = capsys.readouterr().out
    assert "OK: no CSV" in printed


def test_main_prints_known_and_handled_section_for_record_only_files(tmp_path, monkeypatch, capsys):
    record_dir = tmp_path / "results" / "dentist_audit"
    record_dir.mkdir(parents=True)
    (record_dir / "boneloss_manifest.csv").write_text("col1,col2\na,None\n")
    monkeypatch.setattr(m, "REPO", str(tmp_path))

    rc = m.main()

    printed = capsys.readouterr().out
    assert "Known and handled" in printed
    assert "boneloss_manifest.csv" in printed
    assert rc == 0
