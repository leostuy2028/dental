# Tests for dataio/make_plausible_distractors.py
# Rewrites wrong options with real answers drawn (verbatim) from other items of the same
# answer "type", so a model can't eliminate options without looking at the image. main()
# takes all its paths as --closed/--open/--out/--manifest CLI flags, so we just point
# those at tmp_path files.

import pandas as pd

import dataio.make_plausible_distractors as m


def test_answer_type_classifies_common_shapes():
    assert m.answer_type("5") == "count"
    assert m.answer_type("#34") == "tooth_codes"
    assert m.answer_type("mild bone loss") == "bone_loss_severity"
    assert m.answer_type("yes") == "polar"
    assert m.answer_type("maxillary sinus") == "anatomy"
    assert m.answer_type("some random prose answer") == "prose"


def test_main_rewrites_distractors_using_only_real_answers_from_other_items(tmp_path, monkeypatch, capsys):
    closed_p = tmp_path / "closed.parquet"
    open_p = tmp_path / "open.parquet"
    out_p = tmp_path / "out.parquet"
    man_p = tmp_path / "man.csv"

    texts = ["alpha finding here", "beta observation noted", "gamma result seen",
             "delta pattern visible", "epsilon marker present"]
    rows = [{"index": i, "file_name": f"img{i}.jpg", "option1": t,
             "option2": "filler2", "option3": "filler3", "option4": "filler4", "answer": "A"}
            for i, t in enumerate(texts)]
    pd.DataFrame(rows).to_parquet(closed_p, index=False)
    pd.DataFrame({"index": [0], "image_name": ["other.jpg"], "question": ["Q"],
                  "answer": ["unrelated text"]}).to_parquet(open_p, index=False)

    monkeypatch.setattr("sys.argv", [
        "prog", "--closed", str(closed_p), "--open", str(open_p),
        "--out", str(out_p), "--manifest", str(man_p), "--seed", "0",
    ])

    m.main()

    out = pd.read_parquet(out_p)
    man = pd.read_csv(man_p)
    assert len(man) == 5   # every item was rewritten (5 candidates, 4 alternatives each)

    for _, r in out.iterrows():
        opts = [r["option1"], r["option2"], r["option3"], r["option4"]]
        key_text = texts[r["index"]]
        # the key's real text is still present as one of the four options
        assert key_text in opts
        # every option came from the real pool of texts (nothing invented)
        assert all(o in texts for o in opts)
        # the answer letter really points at the key text
        assert opts["ABCD".index(r["answer"])] == key_text

    printed = capsys.readouterr().out
    assert "rewrote the distractors of 5 items; left 0 untouched" in printed
    assert "No option text was authored here." in printed


def test_main_leaves_an_item_untouched_when_too_few_real_alternatives_exist(tmp_path, monkeypatch, capsys):
    closed_p = tmp_path / "closed.parquet"
    open_p = tmp_path / "open.parquet"
    out_p = tmp_path / "out.parquet"
    man_p = tmp_path / "man.csv"

    # only 2 distinct prose answers exist -> at most 1 real alternative per item, so
    # the "need >= 3 picked" rule can never be satisfied
    cl = pd.DataFrame({
        "index": [0, 1], "image_id": ["img0", "img1"],
        "option1": ["alpha finding here", "beta observation noted"],
        "option2": ["f2", "f2"], "option3": ["f3", "f3"], "option4": ["f4", "f4"],
        "answer": ["A", "A"],
    })
    cl.to_parquet(closed_p, index=False)
    pd.DataFrame({"index": [0], "image_name": ["x.jpg"], "question": ["Q"],
                  "answer": ["unrelated"]}).to_parquet(open_p, index=False)

    monkeypatch.setattr("sys.argv", [
        "prog", "--closed", str(closed_p), "--open", str(open_p),
        "--out", str(out_p), "--manifest", str(man_p), "--seed", "1",
    ])

    m.main()

    out = pd.read_parquet(out_p)
    # untouched rows keep their original option text and answer letter exactly
    assert out.loc[0, ["option1", "option2", "option3", "option4", "answer"]].tolist() == \
        ["alpha finding here", "f2", "f3", "f4", "A"]
    assert out.loc[1, ["option1", "option2", "option3", "option4", "answer"]].tolist() == \
        ["beta observation noted", "f2", "f3", "f4", "A"]

    printed = capsys.readouterr().out
    assert "rewrote the distractors of 0 items; left 2 untouched" in printed


def test_main_rejects_a_candidate_that_the_images_own_reference_also_asserts(tmp_path, monkeypatch):
    # candidate "gamma result seen" would normally be eligible for image 0's distractor
    # pool, but image 0's own open-ended reference text also says "gamma result seen",
    # so it must be rejected as "might secretly be true of this image".
    closed_p = tmp_path / "closed.parquet"
    open_p = tmp_path / "open.parquet"
    out_p = tmp_path / "out.parquet"
    man_p = tmp_path / "man.csv"

    texts = ["alpha finding here", "beta observation noted", "gamma result seen",
             "delta pattern visible", "epsilon marker present"]
    rows = [{"index": i, "file_name": f"img{i}.jpg", "option1": t,
             "option2": "filler2", "option3": "filler3", "option4": "filler4", "answer": "A"}
            for i, t in enumerate(texts)]
    pd.DataFrame(rows).to_parquet(closed_p, index=False)
    # image 0's reference text explicitly contains the "gamma result seen" candidate
    pd.DataFrame({"index": [0], "image_name": ["img0.jpg"], "question": ["Q"],
                  "answer": ["notes mention gamma result seen on this film"]}).to_parquet(open_p, index=False)

    monkeypatch.setattr("sys.argv", [
        "prog", "--closed", str(closed_p), "--open", str(open_p),
        "--out", str(out_p), "--manifest", str(man_p), "--seed", "0",
    ])

    m.main()

    out = pd.read_parquet(out_p).set_index("index")
    row0_opts = [out.loc[0, c] for c in ["option1", "option2", "option3", "option4"]]
    assert "gamma result seen" not in row0_opts
