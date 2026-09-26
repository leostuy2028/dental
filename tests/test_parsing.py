# Tests for clients/parsing.py
# This file pulls the answer letter (A/B/C/D) out of a model's reply.
# Getting this wrong would fake the exact letter bias the paper measures,
# so these tests check a lot of different reply shapes.

from clients.parsing import extract_letter, looks_like_refusal, is_api_failure


# ---------- looks_like_refusal ----------

def test_refusal_is_detected():
    assert looks_like_refusal("I'm sorry, I cannot analyze this image") == True


def test_refusal_is_case_insensitive():
    assert looks_like_refusal("AS AN AI I can't help") == True


def test_normal_answer_is_not_a_refusal():
    assert looks_like_refusal("The answer is B") == False


def test_refusal_handles_none_and_empty():
    assert looks_like_refusal(None) == False
    assert looks_like_refusal("") == False


# ---------- is_api_failure ----------

def test_api_failure_sentinel_is_detected():
    assert is_api_failure("max retries exceeded") == True


def test_api_failure_ignores_spaces_around_it():
    assert is_api_failure("  max retries exceeded \n") == True


def test_api_failure_must_match_the_whole_text():
    assert is_api_failure("Error: max retries exceeded") == False


def test_api_failure_is_case_sensitive():
    # current behavior: only the exact lowercase sentinel counts
    assert is_api_failure("Max Retries Exceeded") == False


def test_api_failure_only_accepts_strings():
    assert is_api_failure(None) == False
    assert is_api_failure(123) == False


# ---------- extract_letter: short replies ----------

def test_bare_letter():
    assert extract_letter("B") == "B"


def test_bare_letter_with_punctuation():
    assert extract_letter("B.") == "B"
    assert extract_letter("(C)") == "C"
    assert extract_letter("[D]") == "D"
    assert extract_letter("A)") == "A"


def test_lowercase_bare_letter_is_accepted():
    assert extract_letter("b") == "B"


def test_letter_followed_by_option_text():
    assert extract_letter("D) #21") == "D"


def test_empty_and_none_give_none():
    assert extract_letter("") is None
    assert extract_letter("   ") is None
    assert extract_letter(None) is None


def test_no_letter_gives_none():
    assert extract_letter("no letter here") is None


def test_letter_E_is_not_an_answer():
    assert extract_letter("E") is None


# ---------- extract_letter: refusals must NOT turn into a letter ----------

def test_refusal_does_not_become_A():
    assert extract_letter("I am unable to view images") is None
    assert extract_letter("As an AI, I cannot analyze") is None


# ---------- extract_letter: longer replies with an answer cue ----------

def test_answer_is_cue():
    assert extract_letter("The answer is B") == "B"


def test_answer_colon_cue():
    assert extract_letter("Based on the radiograph...\r\n\r\nAnswer: A") == "A"


def test_bold_correct_answer():
    assert extract_letter("The correct answer is **D** (which is #38).") == "D"


def test_correct_answer_on_next_line():
    assert extract_letter("The correct answer is:\r\n\r\n**B**") == "B"


def test_option_is_correct_cue_beats_earlier_listed_options():
    text = "* C) #48  * D) #44  ... option A is the correct answer.  **A**"
    assert extract_letter(text) == "A"


def test_cue_beats_a_later_loose_letter():
    assert extract_letter("...the answer is A, not B.") == "A"


def test_last_cue_wins_when_model_changes_its_mind():
    assert extract_letter("Answer: B. Wait, re-checking. Answer: C") == "C"


def test_word_starting_with_a_letter_is_not_read_as_answer():
    # "Based" starts with B and "Answer" starts with A; neither should count
    assert extract_letter("Based on the image, the Answer is D") == "D"


def test_trailing_bare_letter_after_reasoning():
    assert extract_letter("...= 29 teeth.\r\n\r\nD") == "D"


def test_last_standalone_letter_is_used_without_a_cue():
    assert extract_letter("I think it is B or maybe C") == "C"


def test_the_word_a_is_read_as_answer_A():
    # CURRENT BEHAVIOR (looks like a bug): the English word "a" counts as the
    # letter A, so a reply with no answer at all can be scored as choosing A.
    assert extract_letter("not a letter at all") == "A"
    assert extract_letter("Is it a cyst?") == "A"


# ---------- extract_letter: cot=True (per-option prompt) ----------

def test_cot_answer_line():
    assert extract_letter("Answer: C", cot=True) == "C"


def test_cot_lowercase_letter():
    assert extract_letter("...\nAnswer: b", cot=True) == "B"


def test_cot_bold_answer():
    assert extract_letter("reasoning...\n**Answer: B**", cot=True) == "B"


def test_cot_takes_the_last_answer_line():
    assert extract_letter("Answer: A\nhmm no\nAnswer: D", cot=True) == "D"


def test_cot_without_answer_line_gives_none():
    # in cot mode we never grab a loose letter from the reasoning
    assert extract_letter("It is clearly B", cot=True) is None


def test_cot_none_of_the_above_gives_none():
    assert extract_letter("Answer: None of the above", cot=True) is None


def test_cot_handles_none_input():
    assert extract_letter(None, cot=True) is None
