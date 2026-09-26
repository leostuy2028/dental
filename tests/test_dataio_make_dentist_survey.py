# Tests for dataio/make_dentist_survey.py
# Buckets closed items by how 3 models' predictions relate to the key (T1/T2/T3/C),
# picks a survey sample per bucket, and adds a prose slice of open items.

import pandas as pd

import dataio.make_dentist_survey as m


def test_is_coord_detects_box_json_but_not_plain_text():
    assert m.is_coord('[{"box_2d": [1,2,3,4]}]') == True
    assert m.is_coord("[1, 2, 3, 4]") == True
    assert m.is_coord("plain text answer") == False
    assert m.is_coord('{"foo": 1}') == True


def _mkdf(preds, keys):
    return pd.DataFrame({"answer": keys, "predicted": preds}, index=[1, 2, 3, 4, 5])


def test_buckets_classifies_agreement_patterns():
    # idx1: all 3 agree on B, key is A -> T1 (order-invariant "key may be wrong")
    # idx2: B/A/C all different, only 1 matches key -> T3 (split)
    # idx3: B/C/C -> 2 agree on C (not key), 0 match key -> T2
    # idx4: A/A/B -> 2 match the key -> "B" (agree-with-key majority, not unanimous)
    # idx5: C/B/B -> 2 agree on B (not key), 0 match key -> T2
    gpt4o = _mkdf(["B", "B", "B", "A", "C"], ["A"] * 5)
    g25 = _mkdf(["B", "A", "C", "A", "B"], ["A"] * 5)
    g35 = _mkdf(["B", "C", "C", "B", "B"], ["A"] * 5)
    dd = {"gpt4o": gpt4o, "g25": g25, "g35": g35}

    out = m.buckets(dd, [1, 2, 3, 4, 5])

    assert out.to_dict() == {1: "T1", 2: "T3", 3: "T2", 4: "B", 5: "T2"}


def _write_survey_inputs(tmp_path, closed_options=None):
    closed_p = tmp_path / "closed.parquet"
    open_p = tmp_path / "open.parquet"
    shuf_paths = {k: tmp_path / f"shuf_{k}.csv" for k in ("gpt4o", "g25", "g35")}
    orig_paths = {k: tmp_path / f"orig_{k}.csv" for k in ("gpt4o", "g25", "g35")}

    idx = [1, 2, 3, 4, 5, 6]
    opt1 = closed_options if closed_options else ["a"] * 6
    closed = pd.DataFrame({
        "index": idx, "category": ["Cat"] * 6, "question": [f"Q{i}" for i in idx],
        "option1": opt1, "option2": ["b"] * 6, "option3": ["c"] * 6, "option4": ["d"] * 6,
        "answer": ["A"] * 6,
    })
    closed.to_parquet(closed_p, index=False)

    # 6th item: all 3 models agree WITH the key -> "C" bucket control
    preds = {
        "gpt4o": ["B", "B", "B", "A", "C", "A"],
        "g25": ["B", "A", "C", "A", "B", "A"],
        "g35": ["B", "C", "C", "B", "B", "A"],
    }
    for k in ("gpt4o", "g25", "g35"):
        df = pd.DataFrame({"index": idx, "answer": ["A"] * 6, "predicted": preds[k]})
        df.to_csv(shuf_paths[k], index=False)
        df.to_csv(orig_paths[k], index=False)   # identical shuffled/original -> order-invariant

    op = pd.DataFrame({
        "index": [10, 11, 12],
        "category": ["Patho1", "Other", "Report"],
        "question": ["Q10", "Q11", "Q12"],
        "answer": ["prose ref 1", "prose ref 2", "report text"],
    })
    op.to_parquet(open_p, index=False)

    return closed_p, open_p, shuf_paths, orig_paths


def _patch(monkeypatch, closed_p, open_p, shuf_paths, orig_paths, out_p):
    monkeypatch.setattr(m, "CLOSED", str(closed_p))
    monkeypatch.setattr(m, "OPEN", str(open_p))
    monkeypatch.setattr(m, "SHUF", {k: str(v) for k, v in shuf_paths.items()})
    monkeypatch.setattr(m, "ORIG", {k: str(v) for k, v in orig_paths.items()})
    monkeypatch.setattr(m, "OUT", str(out_p))
    monkeypatch.setattr(m, "N_T1", 1)
    monkeypatch.setattr(m, "N_T2", 1)
    monkeypatch.setattr(m, "N_T3", 1)
    monkeypatch.setattr(m, "N_C", 1)
    monkeypatch.setattr(m, "N_OPEN_PROSE", 6)   # matches the hardcoded "6" patho split in main()
    monkeypatch.setattr(m, "N_OPEN_COORD", 0)


def test_main_builds_a_survey_manifest_with_closed_and_open_items(tmp_path, monkeypatch, capsys):
    closed_p, open_p, shuf_paths, orig_paths = _write_survey_inputs(tmp_path)
    out_p = tmp_path / "man.csv"
    _patch(monkeypatch, closed_p, open_p, shuf_paths, orig_paths, out_p)

    m.main()

    man = pd.read_csv(out_p)
    closed_rows = man[man.task_type == "closed"]
    open_rows = man[man.task_type == "open"]

    # index 1 is T1, index 6 is C; both buckets are present in the sample
    assert set(closed_rows["bucket"]) == {"T1", "T2", "T3", "C"}
    assert 6 in closed_rows["index"].tolist()
    assert 1 in closed_rows["index"].tolist()

    # "Report" category is excluded from the open pool entirely
    assert 12 not in open_rows["index"].tolist()
    # only the Patho-tagged prose item is picked (6 - 6 = 0 requested from the rest)
    assert open_rows["index"].tolist() == [10]

    assert man["item_id"].duplicated().sum() == 0
    assert closed_rows["index"].duplicated().sum() == 0

    printed = capsys.readouterr().out
    assert "unique item_id check (should be 0): 0" in printed
    assert "within-closed dup index (should be 0): 0" in printed


def test_main_excludes_items_with_a_none_option(tmp_path, monkeypatch):
    # index 2 (bucket T3 normally) carries a "None" option and must be excluded
    opts = ["a", "None", "a", "a", "a", "a"]
    closed_p, open_p, shuf_paths, orig_paths = _write_survey_inputs(tmp_path, closed_options=opts)
    out_p = tmp_path / "man_none.csv"
    _patch(monkeypatch, closed_p, open_p, shuf_paths, orig_paths, out_p)

    m.main()

    man = pd.read_csv(out_p)
    assert 2 not in man["index"].tolist()
