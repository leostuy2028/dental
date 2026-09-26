# Tests for eval_open/prompts_open.py
# This file builds the (system, user) prompt for each of 4 "arms" of a study on
# whether asking a model for coordinates changes its open-ended answer score.
# Each arm adds exactly one thing on top of the previous arm.

from eval_open.prompts_open import build_arm, ARMS, COAX_SYSTEM


# ---------- plain arm ----------

def test_plain_arm_has_no_system_message():
    system, user = build_arm("plain", "What is this?")
    assert system is None


def test_plain_arm_user_text_is_the_faithful_prompt():
    system, user = build_arm("plain", "What is this?")
    assert user == (
        "Question: What is this?\n"
        "Please provide a detailed and accurate answer to the question."
    )


# ---------- coax arm ----------

def test_coax_arm_uses_the_coax_system_persona():
    system, user = build_arm("coax", "What is this?")
    assert system == COAX_SYSTEM


def test_coax_arm_user_text_is_unchanged_from_plain():
    _, plain_user = build_arm("plain", "What is this?")
    _, coax_user = build_arm("coax", "What is this?")
    assert coax_user == plain_user


# ---------- coax_primer arm ----------

def test_coax_primer_requires_primer_text():
    try:
        build_arm("coax_primer", "Q")
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "coax_primer" in str(e)


def test_coax_primer_rejects_empty_primer_text():
    # empty string is falsy, so it is treated the same as "missing"
    try:
        build_arm("coax_primer", "Q", primer_text="")
        assert False, "expected a ValueError"
    except ValueError:
        pass


def test_coax_primer_strips_and_prepends_the_primer():
    system, user = build_arm("coax_primer", "Q", primer_text="  PRIMER TEXT  ")
    assert system == COAX_SYSTEM
    assert user == "PRIMER TEXT\n\nQuestion: Q\nPlease provide a detailed and accurate answer to the question."


# ---------- coax_primer_coords arm ----------

def test_coax_primer_coords_requires_primer_text():
    try:
        build_arm("coax_primer_coords", "Q", img_w=100, img_h=200)
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "coax_primer_coords" in str(e)


def test_coax_primer_coords_requires_image_dimensions():
    try:
        build_arm("coax_primer_coords", "Q", primer_text="P")
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "img_w, img_h" in str(e)


def test_coax_primer_coords_full_text():
    system, user = build_arm(
        "coax_primer_coords", "Q", primer_text="P", img_w=100, img_h=200
    )
    assert system == COAX_SYSTEM
    assert user == (
        "P\n\nQuestion: Q\nPlease provide a detailed and accurate answer to the question."
        "\n\nThe panoramic image is 100 by 200 pixels, with (0, 0) at the top-left corner. "
        "In addition to your written answer, for every finding you report also give its "
        "location in the image as a JSON list, one object per finding, using this exact form:\n"
        '[{"label": "<finding>", "tooth_id": "<FDI code if applicable>", '
        '"box_2d": [x1, y1, x2, y2]}]\n'
        "where box_2d is the bounding box in pixel coordinates. Include the JSON after your "
        "written answer."
    )


def test_coax_primer_coords_accepts_img_w_or_img_h_of_zero():
    # CURRENT BEHAVIOR (looks like a bug): the check is "is None", so 0 (a falsy but
    # valid width/height) is accepted rather than rejected as missing.
    system, user = build_arm(
        "coax_primer_coords", "Q", primer_text="P", img_w=0, img_h=0
    )
    assert "0 by 0 pixels" in user


# ---------- unknown arm ----------

def test_unknown_arm_raises_value_error():
    try:
        build_arm("bogus", "Q")
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "bogus" in str(e)
        assert str(ARMS) in str(e)
