# Tests for eval_open/build_predictions.py
# This file builds paired "prose" vs "coords" prediction variants for the same
# ground-truth answer, so a judge can be graded on format alone: same clinical
# content, only the presence of coordinates differs.

import random
import pandas as pd

import eval_open.build_predictions as bp


# ---------- is_coord_ref ----------

def test_coord_ref_json_list_with_box_2d():
    assert bp.is_coord_ref('[{"box_2d": [1, 2, 3, 4]}]') is True


def test_coord_ref_json_dict_with_point_2d():
    assert bp.is_coord_ref('{"point_2d": [1, 2]}') is True


def test_plain_json_without_geometry_keys_is_not_coord_ref():
    assert bp.is_coord_ref('[{"a": 1}]') is False


def test_plain_prose_is_not_coord_ref():
    assert bp.is_coord_ref("30 teeth are visualized.") is False


def test_coord_ref_tolerates_leading_whitespace():
    assert bp.is_coord_ref("  [1,2] with box_2d") is True


def test_coord_ref_handles_non_string_input():
    assert bp.is_coord_ref(123) is False


# ---------- _flag_true / _flag_false ----------

def test_flag_true_matches_quoted_true():
    assert bp._flag_true('"is_impacted": "True"', "is_impacted") is True


def test_flag_true_matches_unquoted_lowercase_true():
    assert bp._flag_true('"is_impacted": true', "is_impacted") is True


def test_flag_true_is_false_when_key_absent():
    assert bp._flag_true('"other_key": "True"', "is_impacted") is False


def test_flag_false_matches_quoted_false():
    assert bp._flag_false('"is_impacted": "False"', "is_impacted") is True


def test_flag_false_is_false_when_value_is_true():
    assert bp._flag_false('"is_impacted": "True"', "is_impacted") is False


# ---------- coord_ref_to_prose ----------

def test_coord_ref_to_prose_extracts_finding_from_finding_key_dict():
    gt = ('[{"Teeth position": {"point_2d": [1242, 726]}}, '
          '{"Crown": {"box_2d": [1220, 637, 1266, 741]}}]')
    # no tooth_id in the JSON and no question given -> falls back to the bare-loc message
    assert bp.coord_ref_to_prose(gt) == "findings: Crown"


def test_coord_ref_to_prose_recovers_tooth_from_the_question_when_answer_has_none():
    gt = '[{"point_2d": [1, 2]}]'
    assert bp.coord_ref_to_prose(gt, "condition of tooth #22?") == "Tooth #22"


def test_coord_ref_to_prose_bare_localization_with_no_tooth_anywhere():
    gt = '[{"point_2d": [1, 2]}]'
    assert bp.coord_ref_to_prose(gt) == (
        "The indicated tooth is present and localized on the radiograph.")


def test_coord_ref_to_prose_combines_tooth_finding_and_flags():
    gt = '[{"tooth_id": "22", "label": "Filling", "is_impacted": "True", "box_2d": [1,2,3,4]}]'
    assert bp.coord_ref_to_prose(gt) == "Tooth #22; findings: Filling; impacted"


def test_coord_ref_to_prose_reports_not_impacted():
    gt = '[{"tooth_id":"5", "is_wisdom_tooth": "True", "is_impacted": "False"}]'
    assert bp.coord_ref_to_prose(gt) == "Tooth #5; wisdom tooth, not impacted"


def test_coord_ref_to_prose_sorts_multiple_teeth_numerically_not_lexically():
    gt = '[{"tooth_id": "10"}, {"tooth_id": "9"}]'
    # a string sort would put "10" before "9"; the real code sorts by int value
    assert bp.coord_ref_to_prose(gt) == "Teeth #9, #10"


def test_coord_ref_to_prose_dedupes_a_finding_named_in_both_places():
    gt = '[{"Crown": {"box_2d": [1,2,3,4]}}, {"label": "Crown"}]'
    assert bp.coord_ref_to_prose(gt) == "findings: Crown"


def test_coord_ref_to_prose_is_robust_to_truncated_json():
    # regex-based extraction, not json.loads, so a cut-off string still yields
    # whatever tokens matched before the cut
    gt = '[{"tooth_id": "22", "label": "Filli'
    assert bp.coord_ref_to_prose(gt) == "Tooth #22"


# ---------- _synthetic_box ----------

def test_synthetic_box_is_deterministic_for_a_given_rng_seed():
    assert bp._synthetic_box(random.Random(5)) == [2551, 523, 2705, 628]


def test_synthetic_box_stays_within_the_assumed_image_bounds():
    box = bp._synthetic_box(random.Random(1))
    x1, y1, x2, y2 = box
    assert 0 <= x1 <= bp._IMG_W
    assert 0 <= y1 <= bp._IMG_H
    assert x2 > x1 and y2 > y1


# ---------- prose_ref_add_coords ----------

def test_prose_ref_add_coords_attaches_a_box_per_mentioned_tooth():
    out = bp.prose_ref_add_coords("Teeth #26 and #14 show signs of abscess", 42)
    assert out.startswith("Teeth #26 and #14 show signs of abscess [")
    assert '"tooth_id": "26"' in out
    assert '"tooth_id": "14"' in out


def test_prose_ref_add_coords_attaches_generic_points_when_no_tooth_mentioned():
    out = bp.prose_ref_add_coords("no teeth mentioned here", 42)
    assert out.startswith("no teeth mentioned here [")
    assert "point_2d" in out
    assert "tooth_id" not in out


def test_prose_ref_add_coords_is_reproducible_for_the_same_index():
    a = bp.prose_ref_add_coords("Tooth #11 has a filling", 7)
    b = bp.prose_ref_add_coords("Tooth #11 has a filling", 7)
    assert a == b


def test_prose_ref_add_coords_differs_across_indices():
    a = bp.prose_ref_add_coords("Tooth #11 has a filling", 1)
    b = bp.prose_ref_add_coords("Tooth #11 has a filling", 2)
    assert a != b


# ---------- build() end to end ----------

def test_build_writes_one_row_per_input_item_with_the_right_ref_type(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame([
        {"index": 1, "question": "What features are visible in tooth #31?",
         "answer": ('[{"Teeth position": {"point_2d": [10, 20]}}, '
                    '{"Crown": {"box_2d": [1,2,3,4]}}]'),
         "category": "Teeth"},
        {"index": 2, "question": "What is the condition of teeth #26 and #14?",
         "answer": "Teeth #26 and #14 show signs of abscess.", "category": "Condition"},
    ])
    df.to_parquet(tmp_path / "tiny.parquet", index=False)
    monkeypatch.setattr(bp, "DATA", str(tmp_path / "tiny.parquet"))
    monkeypatch.setattr(bp, "OUT", str(tmp_path / "out.parquet"))

    out = bp.build()

    assert len(out) == 2
    row1 = out[out["index"] == 1].iloc[0]
    assert row1.ref_type == "coord_ref"
    assert row1.pred_prose == "Tooth #31; findings: Crown"
    assert row1.pred_coords == row1.answer  # raw JSON reused verbatim, just stripped

    row2 = out[out["index"] == 2].iloc[0]
    assert row2.ref_type == "prose_ref"
    assert row2.pred_prose == "Teeth #26 and #14 show signs of abscess."
    assert '"tooth_id": "26"' in row2.pred_coords
    assert '"tooth_id": "14"' in row2.pred_coords

    # it really wrote the parquet file, readable back independently
    reread = pd.read_parquet(tmp_path / "out.parquet")
    assert len(reread) == 2

    printed = capsys.readouterr().out
    assert "2 items" in printed
    assert "1 coord_ref" in printed
    assert "1 prose_ref" in printed
