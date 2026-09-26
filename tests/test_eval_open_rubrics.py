# Tests for eval_open/rubrics.py
# This file builds the text prompt sent to an LLM judge, grading a model's open-ended
# answer against a ground truth. Two rubrics ("original", "rephrased") are an A/B pair
# built to isolate a coordinate-reward bias; "v2" is a different, better rubric.
# Getting the prompt text exactly right matters: it's the input that decides scores.

from eval_open.rubrics import (
    build_grading_prompt,
    _prep_gt,
    INSTRUCTION,
    DEBIAS_RULE,
    TABLE_HEADER,
)


# ---------- _prep_gt ----------

def test_prep_gt_spaces_out_and_or_tokens():
    assert _prep_gt("a <AND> b <OR> c") == "a  <AND>  b  <OR>  c"


def test_prep_gt_converts_non_string_to_string():
    assert _prep_gt(123) == "123"


def test_prep_gt_leaves_plain_text_unchanged():
    assert _prep_gt("just plain text") == "just plain text"


# ---------- build_grading_prompt: rubric='original' ----------

def test_original_prompt_starts_with_shared_instruction():
    p = build_grading_prompt("Q?", "GT here", "Pred here", rubric="original")
    assert p.startswith(INSTRUCTION)


def test_original_prompt_has_table_header_and_18_lines():
    p = build_grading_prompt("Q?", "GT here", "Pred here", rubric="original")
    lines = p.split("\n")
    # 5 instruction lines + 1 blank + table header (2 lines) + 9 examples + 1 real row
    assert len(lines) == 18
    assert lines[6] == TABLE_HEADER.split("\n")[0]
    assert lines[7] == TABLE_HEADER.split("\n")[1]


def test_original_prompt_ends_with_the_real_row_and_a_trailing_blank_cell():
    p = build_grading_prompt("Q?", "GT here", "Pred here", rubric="original")
    assert p.endswith("Q? | GT here | Pred here | ")


def test_original_prompt_does_not_include_the_debias_rule():
    p = build_grading_prompt("Q?", "GT here", "Pred here", rubric="original")
    assert DEBIAS_RULE not in p


def test_original_prompt_gives_coordinate_answers_a_bonus_score():
    # this is the exact bias the paper flags: same clinical finding ("Crown"),
    # score goes up as coordinates are added: 0.8 -> 0.9 -> 1.0
    p = build_grading_prompt("Q?", "GT here", "Pred here", rubric="original")
    assert "| Crown | 0.8" in p
    assert "Crown at position: [1230, 627, 1276, 750] | 0.9" in p


# ---------- build_grading_prompt: rubric='rephrased' ----------

def test_rephrased_prompt_includes_the_debias_rule():
    p = build_grading_prompt("Q?", "GT here", "Pred here", rubric="rephrased")
    assert DEBIAS_RULE in p
    assert p.startswith(INSTRUCTION + "\n" + DEBIAS_RULE)


def test_rephrased_prompt_scores_prose_and_coords_the_same():
    # format-invariant: plain "Crown" and "Crown at position: ..." both score 1.0
    p = build_grading_prompt("Q?", "GT here", "Pred here", rubric="rephrased")
    assert "| Crown | 1.0" in p
    assert "Crown at position: [1230, 627, 1276, 750] | 1.0" in p


def test_original_and_rephrased_share_the_six_non_coordinate_examples():
    orig = build_grading_prompt("Q?", "GT here", "Pred here", rubric="original")
    reph = build_grading_prompt("Q?", "GT here", "Pred here", rubric="rephrased")
    orig_lines = orig.split("\n")
    reph_lines = reph.split("\n")
    # lines 8-13 (0-indexed) are the 6 shared example rows in the 'original' prompt
    shared_from_original = orig_lines[8:14]
    # in 'rephrased' the instruction block has one extra line (the debias rule),
    # so everything shifts down by one line
    shared_from_rephrased = reph_lines[9:15]
    assert shared_from_original == shared_from_rephrased


# ---------- build_grading_prompt: rubric='v2' ----------

def test_v2_prompt_starts_with_the_v2_instruction():
    p = build_grading_prompt("Q?", "GT <AND> here", "Pred here", rubric="v2")
    assert p.startswith("You are grading how clinically correct")


def test_v2_prompt_has_worked_examples_and_a_final_question_block():
    p = build_grading_prompt("Q?", "GT <AND> here", "Pred here", rubric="v2")
    assert p.count("Worked examples:") == 1
    assert "Now grade this one:" in p
    assert p.endswith(
        "Now grade this one:\n"
        "Question: Q?\n"
        "Ground truth: GT  <AND>  here\n"
        "Prediction: Pred here\n"
    )


# ---------- unknown rubric ----------

def test_unknown_rubric_raises_value_error():
    try:
        build_grading_prompt("q", "gt", "pred", rubric="bogus")
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "bogus" in str(e)
        assert "original" in str(e) and "rephrased" in str(e) and "v2" in str(e)
