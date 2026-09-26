# Tests for dataio/make_boneloss_survey.py
# Builds the bone-loss key-bias probe: DISAGREE (key says "no loss", both models say
# "loss"), AGREE_NORM (all three agree "no loss"), KEY_POS (key asserts loss), plus fixed
# COUNT_TEST images that are always added if not already selected.

import pandas as pd

import dataio.make_boneloss_survey as m


def test_says_loss_detects_bone_loss_and_respects_negation():
    assert m.says_loss("No apparent bone loss noted") == False
    assert m.says_loss("Mild bone loss present") == True
    assert m.says_loss("Normal architecture") == False
    assert m.says_loss("Moderate periodontal resorption") == True


def test_gt_count_extracts_a_stated_tooth_total():
    assert m.gt_count("28 teeth are visualized in this image") == 28
    assert m.gt_count("30 teeth present") == 30
    assert m.gt_count("no count mentioned") is None


def test_take_never_samples_more_than_the_pool_has(tmp_path):
    import random
    rng = random.Random(0)
    assert m.take(rng, {1, 2}, 5) == {1, 2}


def test_main_builds_the_three_buckets_plus_count_test_images(tmp_path, monkeypatch, capsys):
    open_p = tmp_path / "open.parquet"
    gpt_p = tmp_path / "gpt.csv"
    gem_p = tmp_path / "gem.csv"
    out_p = tmp_path / "out.csv"

    op = pd.DataFrame({
        "index": [1, 2, 3],
        "image_name": ["img1.jpg", "img2.jpg", "img3.jpg"],
        "question": ["bone loss?", "bone architecture?", "jawbone status?"],
        "answer": ["No apparent bone loss", "No apparent bone loss",
                   "Moderate bone loss present, 28 teeth are visualized"],
    })
    op.to_parquet(open_p, index=False)
    pd.DataFrame({"index": [1, 2, 3],
                  "answer": ["There is bone loss", "Normal, no loss", "Bone loss confirmed"]}).to_csv(gpt_p, index=False)
    pd.DataFrame({"index": [1, 2, 3],
                  "answer": ["There is bone loss too", "Normal", "Bone loss too"]}).to_csv(gem_p, index=False)

    monkeypatch.setattr(m, "OPEN", str(open_p))
    monkeypatch.setattr(m, "GPT", str(gpt_p))
    monkeypatch.setattr(m, "GEM", str(gem_p))
    monkeypatch.setattr(m, "OUT", str(out_p))
    monkeypatch.setattr(m, "N_DISAGREE", 1)
    monkeypatch.setattr(m, "N_KEY_POS", 1)
    monkeypatch.setattr(m, "N_AGREE_NORM", 1)

    m.main()

    man = pd.read_csv(out_p, keep_default_na=False).set_index("image_name")
    assert man.loc["img1.jpg", "bucket"] == "DISAGREE"
    assert man.loc["img1.jpg", "key_stance"] == "none"
    assert man.loc["img1.jpg", "gpt_stance"] == "loss"
    assert man.loc["img1.jpg", "gem_stance"] == "loss"

    assert man.loc["img2.jpg", "bucket"] == "AGREE_NORM"
    assert man.loc["img3.jpg", "bucket"] == "KEY_POS"
    assert man.loc["img3.jpg", "gt_count"] == "28"

    # the three fixed count-test images are always added when not already selected
    count_test_rows = man[man["bucket"] == "COUNT_TEST"]
    assert set(count_test_rows.index) == {"016640.jpg", "017148.jpg", "018174.jpg"}
    assert (count_test_rows["key_stance"] == "n/a").all()

    assert len(man) == 6
    printed = capsys.readouterr().out
    assert "wrote" in printed and "6 images" in printed
    assert "available pools: DISAGREE 1, AGREE_NORM 1, KEY_POS 1" in printed
