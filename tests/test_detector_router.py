# Tests for detector/router.py
# A regex router: decides whether the detector (not a VLM) should answer a benchmark
# question, based only on the question text. Also has small helpers that pick a
# multiple-choice option, or write out the free-text answer, from what the detector saw.

from router import answer_mcq, answer_text, route


# ---------- route ----------

def test_plain_count_question_routes_to_count():
    assert route("How many teeth are visualized?") == "count"


def test_natural_teeth_wording_still_routes_to_count():
    assert route("How many natural teeth are visible in the radiograph?") == "count"


def test_count_question_without_trailing_question_mark():
    assert route("How many teeth were detected") == "count"


def test_wisdom_teeth_detected_question_routes_to_wisdom():
    assert route("How many wisdom teeth are detected?") == "wisdom"


def test_third_molars_wording_also_routes_to_wisdom():
    assert route("How many third molars are present?") == "wisdom"


def test_eruption_status_question_is_not_routed():
    # eruption is not something a box detector can read, so this must go to the VLM
    assert route("How many wisdom teeth are erupted?") is None


def test_impaction_status_question_is_not_routed_even_if_it_mentions_wisdom_teeth():
    assert route("how many wisdom teeth are impacted?") is None


def test_question_with_a_trailing_clause_after_a_comma_is_not_routed():
    assert route("How many teeth are visualized, excluding wisdom teeth?") is None


def test_condition_question_is_not_routed():
    assert route("How many teeth have crowns?") is None


def test_route_normalises_extra_whitespace():
    assert route("How many   teeth are visualized  in the radiograph?") == "count"


def test_route_handles_none():
    assert route(None) is None


def test_route_is_case_insensitive():
    assert route("HOW MANY WISDOM TEETH ARE DETECTED?") == "wisdom"


# ---------- answer_mcq ----------

def test_answer_mcq_count_picks_the_closest_numeric_option():
    det = {"count": 28, "wisdom": ["18", "28"]}
    assert answer_mcq("count", det, ["26", "27", "28", "29"]) == 2


def test_answer_mcq_wisdom_uses_the_length_of_the_wisdom_list():
    det = {"count": 28, "wisdom": ["18", "28"]}
    assert answer_mcq("wisdom", det, ["0", "1", "2", "3"]) == 2


def test_answer_mcq_ties_break_towards_the_earlier_option():
    det = {"count": 5, "wisdom": []}
    # 4 and 6 are equally close to 5; index 0 (value 4) should win the tie
    assert answer_mcq("count", det, ["4", "6"]) == 0


def test_answer_mcq_returns_none_for_non_numeric_options():
    det = {"count": 28, "wisdom": []}
    assert answer_mcq("count", det, ["twenty-six", "27", "28", "29"]) is None


def test_answer_mcq_returns_none_if_any_single_option_is_not_a_bare_number():
    det = {"count": 1, "wisdom": []}
    assert answer_mcq("count", det, ["1", "2 teeth"]) is None


# ---------- answer_text ----------

def test_answer_text_count():
    assert answer_text("count", {"count": 28, "wisdom": []}) == \
        "28 teeth are visualized in the radiograph."


def test_answer_text_no_wisdom_teeth():
    assert answer_text("wisdom", {"count": 1, "wisdom": []}) == \
        "No wisdom teeth are detected in the radiograph."


def test_answer_text_one_wisdom_tooth_uses_singular_grammar():
    assert answer_text("wisdom", {"count": 1, "wisdom": ["38"]}) == \
        "1 wisdom tooth is detected: #38."


def test_answer_text_multiple_wisdom_teeth_uses_plural_grammar():
    assert answer_text("wisdom", {"count": 1, "wisdom": ["18", "28"]}) == \
        "2 wisdom teeth are detected: #18, #28."
