# Tests for utils/vlmeval_parse.py
# get_single_choice_prediction is copied verbatim from the benchmark's own scoring
# code (VLMEvalKit). faithful_predict is our reproducible version of it: same
# matching logic, but the "couldn't parse it, just guess" fallback is seeded.

from utils import vlmeval_parse
from utils.vlmeval_parse import get_single_choice_prediction, faithful_predict

INDEX2ANS = {"A": "Caries", "B": "Impaction", "C": "None", "D": "Bone loss"}
CHOICES = ["A", "B", "C", "D"]


# ---------- get_single_choice_prediction ----------

def test_matches_a_letter_in_parentheses():
    assert get_single_choice_prediction("The answer is (B).", CHOICES, INDEX2ANS) == "B"


def test_matches_a_bare_letter_followed_by_period():
    assert get_single_choice_prediction("I pick B.", CHOICES, INDEX2ANS) == "B"


def test_matches_the_option_text_when_no_letter_is_present():
    assert get_single_choice_prediction(
        "It looks like Impaction to me.", CHOICES, INDEX2ANS) == "B"


def test_parenthesized_letters_win_over_loose_text_and_use_the_earliest_one():
    # CURRENT BEHAVIOR (looks like a bug): "the answer is (D)" is clearly the
    # intended pick, but the function returns whichever candidate appears
    # EARLIEST in the text, so the dismissed option (A) wins instead.
    text = "Option (A) is wrong, the answer is (D)"
    assert get_single_choice_prediction(text, CHOICES, INDEX2ANS) == "A"


def test_when_two_letters_loosely_match_only_the_one_with_a_locatable_position_wins():
    # "B" is followed by a comma (matches the ' B,' loose form) but its own
    # ' B ' / '(B)' / option-text forms are not found later, so it never gets a
    # position and loses to "C", which does have one (" C " has a trailing space).
    assert get_single_choice_prediction("B, C", CHOICES, INDEX2ANS) == "C"


def test_option_text_match_uses_the_position_of_the_text():
    assert get_single_choice_prediction(
        "The condition is Bone loss here", CHOICES, INDEX2ANS) == "D"


def test_trailing_period_is_stripped_before_matching_so_a_bare_letter_still_matches():
    # The whole response has its trailing punctuation stripped up front, so "B."
    # at the very end becomes "B" and matches the plain ' B ' form, not the
    # ' {c}.' loose form.
    assert get_single_choice_prediction("the choice is B.", CHOICES, INDEX2ANS) == "B"


def test_an_internal_period_right_after_the_letter_also_falls_back(monkeypatch):
    # CURRENT BEHAVIOR (looks like a bug): same issue as the comma case below. A
    # period that is NOT at the very end of the message (so it survives the
    # up-front strip) matches the loose ' {c}.' candidate check, but the position
    # lookup never tries ' {c}.', so the candidate can never be located and the
    # function falls back to a random guess.
    monkeypatch.setattr(vlmeval_parse.random, "choice", lambda choices: "Z-forced")
    assert get_single_choice_prediction("the choice is B. It is clear", CHOICES, INDEX2ANS) == "Z-forced"


def test_comma_loose_match_finds_a_candidate_but_then_cannot_locate_its_position(monkeypatch):
    # CURRENT BEHAVIOR (looks like a bug): "B," matches the loose ' {c},' candidate
    # check, but the POSITION lookup right after only tries ' B ', '(B)', and the
    # option text -- never ' B,' -- so it can never find where B is, "positions"
    # stays empty, and the function silently falls through to the random fallback
    # even though a candidate was clearly found.
    monkeypatch.setattr(vlmeval_parse.random, "choice", lambda choices: "Z-forced")
    assert get_single_choice_prediction("the choice is B, right", CHOICES, INDEX2ANS) == "Z-forced"


def test_falls_back_to_random_choice_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(vlmeval_parse.random, "choice", lambda choices: "Z-forced")
    assert get_single_choice_prediction("nothing matches at all", CHOICES, INDEX2ANS) == "Z-forced"


# ---------- faithful_predict ----------

def test_faithful_predict_matches_deterministically_like_the_original():
    letter, used_fallback = faithful_predict("The answer is (B).", INDEX2ANS, seed=42)
    assert (letter, used_fallback) == ("B", False)


def test_faithful_predict_flags_the_fallback():
    letter, used_fallback = faithful_predict("no match here at all", INDEX2ANS, seed=42)
    assert used_fallback is True
    assert letter in CHOICES


def test_faithful_predict_fallback_is_reproducible_for_the_same_seed():
    first = faithful_predict("no match here at all", INDEX2ANS, seed=7)
    second = faithful_predict("no match here at all", INDEX2ANS, seed=7)
    assert first == second


def test_faithful_predict_fallback_can_differ_for_different_seeds():
    a = faithful_predict("no match here at all", INDEX2ANS, seed=1)
    b = faithful_predict("no match here at all", INDEX2ANS, seed=999)
    # both are valid fallback picks; the seeds just need not force the same one
    assert a[1] is True and b[1] is True
    assert a[0] in CHOICES and b[0] in CHOICES


def test_faithful_predict_trailing_period_is_stripped_before_matching():
    letter, used_fallback = faithful_predict("the choice is B.", INDEX2ANS, seed=1)
    assert (letter, used_fallback) == ("B", False)


def test_faithful_predict_an_internal_period_right_after_the_letter_also_falls_back():
    # same current behavior as get_single_choice_prediction above
    letter, used_fallback = faithful_predict("the choice is B. It is clear", INDEX2ANS, seed=1)
    assert used_fallback is True
    assert letter in CHOICES


def test_faithful_predict_matches_the_option_text_when_no_letter_is_present():
    letter, used_fallback = faithful_predict("It looks like Impaction to me.", INDEX2ANS, seed=1)
    assert (letter, used_fallback) == ("B", False)


def test_faithful_predict_comma_loose_match_also_falls_back():
    # Same current behavior as get_single_choice_prediction: a comma-loose match is
    # found as a candidate but its position can never be located, so this falls
    # back to the seeded random choice instead of returning "B".
    letter, used_fallback = faithful_predict("the choice is B, right", INDEX2ANS, seed=1)
    assert used_fallback is True
    assert letter in CHOICES


def test_faithful_predict_handles_none_response():
    letter, used_fallback = faithful_predict(None, INDEX2ANS, seed=1)
    assert used_fallback is True
    assert letter in CHOICES
