# Tests for utils/results_io.py
# This file writes/reads a result CSV together with a ".meta.json" sidecar file
# holding the experiment metadata, so every result file is self-describing.

import json
import os

import pandas as pd

from utils import results_io


def test_meta_path_appends_meta_json():
    assert results_io.meta_path("results/foo.csv") == "results/foo.csv.meta.json"


def test_git_commit_returns_the_subprocess_output(monkeypatch):
    def fake_check_output(cmd, stderr=None):
        return b"abc1234\n"

    monkeypatch.setattr(results_io.subprocess, "check_output", fake_check_output)
    assert results_io._git_commit() == "abc1234"


def test_git_commit_falls_back_to_unknown_on_error(monkeypatch):
    def fake_check_output(cmd, stderr=None):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(results_io.subprocess, "check_output", fake_check_output)
    assert results_io._git_commit() == "unknown"


def test_write_results_creates_a_pristine_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "_git_commit", lambda: "deadbee")
    df = pd.DataFrame({"index": [1, 2], "option1": ["#44", "#45"]})
    path = str(tmp_path / "out.csv")

    results_io.write_results(df, path, meta={"experiment": "E1"})

    # the CSV itself has no comment lines or metadata mixed in
    with open(path, encoding="utf-8") as f:
        contents = f.read()
    assert contents.splitlines()[0] == "index,option1"
    assert "#44" in contents


def test_write_results_fills_in_default_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "_git_commit", lambda: "deadbee")
    df = pd.DataFrame({"index": [1]})
    path = str(tmp_path / "sub" / "out.csv")  # also checks the dir gets created

    results_io.write_results(df, path, meta={"experiment": "E1"})

    assert os.path.exists(path)
    with open(results_io.meta_path(path), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["experiment"] == "E1"
    assert meta["code_commit"] == "deadbee"
    assert meta["data_file"] == "out.csv"
    assert "generated_utc" in meta


def test_write_results_does_not_overwrite_meta_the_caller_already_set(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "_git_commit", lambda: "should-not-be-used")
    df = pd.DataFrame({"index": [1]})
    path = str(tmp_path / "out.csv")

    results_io.write_results(df, path, meta={"code_commit": "manual-commit"})

    meta = json.load(open(results_io.meta_path(path), encoding="utf-8"))
    assert meta["code_commit"] == "manual-commit"


def test_load_results_without_return_meta_gives_just_the_dataframe(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "_git_commit", lambda: "deadbee")
    df = pd.DataFrame({"index": [1, 2], "answer": ["A", "B"]})
    path = str(tmp_path / "out.csv")
    results_io.write_results(df, path, meta={})

    loaded = results_io.load_results(path)
    assert list(loaded["answer"]) == ["A", "B"]


def test_load_results_with_return_meta_gives_dataframe_and_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "_git_commit", lambda: "deadbee")
    df = pd.DataFrame({"index": [1]})
    path = str(tmp_path / "out.csv")
    results_io.write_results(df, path, meta={"model": "gpt-4o"})

    loaded_df, meta = results_io.load_results(path, return_meta=True)
    assert list(loaded_df["index"]) == [1]
    assert meta["model"] == "gpt-4o"


def test_load_results_with_no_sidecar_gives_empty_meta(tmp_path):
    df = pd.DataFrame({"index": [1]})
    path = str(tmp_path / "bare.csv")
    df.to_csv(path, index=False)

    loaded_df, meta = results_io.load_results(path, return_meta=True)
    assert meta == {}


def test_read_meta_returns_the_sidecar_contents(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "_git_commit", lambda: "deadbee")
    df = pd.DataFrame({"index": [1]})
    path = str(tmp_path / "out.csv")
    results_io.write_results(df, path, meta={"n": 5})

    assert results_io.read_meta(path)["n"] == 5


def test_read_meta_with_no_sidecar_gives_empty_dict(tmp_path):
    path = str(tmp_path / "nothing.csv")
    assert results_io.read_meta(path) == {}
