# Tests for prompts/claude.py
# Builds the (system, messages) payload sent to the Anthropic API for one
# closed-ended (multiple choice) question.

import base64

from prompts.claude import (
    build_prompt, _image_part, _image_bytes_part, _text_part, SYSTEM, COAX_SYSTEM,
)

IMG_B64 = base64.b64encode(b"imgbytes").decode()


def make_row(**overrides):
    row = {
        "question": "Which tooth is missing?",
        "option1": "opt A", "option2": "opt B", "option3": "opt C", "option4": "opt D",
        "image": IMG_B64,
    }
    row.update(overrides)
    return row


# ---------- small helpers ----------

def test_image_part_wraps_the_base64_string():
    part = _image_part("abc123")
    assert part == {"type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc123"}}


def test_image_bytes_part_base64_encodes_the_bytes():
    part = _image_bytes_part(b"hello")
    assert part["source"]["data"] == base64.b64encode(b"hello").decode()


def test_text_part_wraps_the_string():
    assert _text_part("hi") == {"type": "text", "text": "hi"}


# ---------- build_prompt: basic shape ----------

def test_default_mode_uses_the_house_system_prompt():
    system, messages = build_prompt(make_row())
    assert system == SYSTEM


def test_coax_mode_uses_the_coax_system_prompt():
    system, messages = build_prompt(make_row(), mode="coax")
    assert system == COAX_SYSTEM


def test_messages_is_a_single_user_turn():
    _, messages = build_prompt(make_row())
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_default_prompt_has_the_image_then_the_question_text():
    _, messages = build_prompt(make_row())
    content = messages[0]["content"]
    assert len(content) == 2
    assert content[0] == {"type": "image",
                          "source": {"type": "base64", "media_type": "image/jpeg", "data": IMG_B64}}
    assert content[1]["type"] == "text"
    assert "Which tooth is missing?" in content[1]["text"]
    assert "A) opt A" in content[1]["text"]
    assert content[1]["text"].endswith("Reply with only the letter of the correct answer: A, B, C, or D.")


def test_cot_mode_asks_for_reasoning_then_an_answer_line():
    _, messages = build_prompt(make_row(), cot=True)
    text = messages[0]["content"][-1]["text"]
    assert "write one sentence" in text
    assert text.endswith("Answer: A, Answer: B, Answer: C, or Answer: D.")


def test_coax_mode_text_instructs_never_to_refuse():
    _, messages = build_prompt(make_row(), mode="coax")
    text = messages[0]["content"][-1]["text"]
    assert "never refuse" in text
    assert text.endswith("Output only that letter and nothing else.")


def test_coax_cot_mode_combines_coax_wording_with_per_option_reasoning():
    _, messages = build_prompt(make_row(), mode="coax", cot=True)
    text = messages[0]["content"][-1]["text"]
    assert "never refuse" in text
    assert text.endswith("Answer: A, Answer: B, Answer: C, or Answer: D.")


# ---------- context / examples / visual exemplars ----------

def test_context_is_prepended_as_a_stripped_text_block():
    _, messages = build_prompt(make_row(), context="  some primer text  \n")
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "some primer text\n"}


def test_no_context_means_no_extra_text_block():
    _, messages = build_prompt(make_row(), context=None)
    assert len(messages[0]["content"]) == 2  # just image + question


def test_examples_are_inserted_before_the_test_image_with_answers():
    example = {"question": "Ex Q", "option1": "1", "option2": "2", "option3": "3",
              "option4": "4", "answer": "B", "image": base64.b64encode(b"ex-img").decode()}
    _, messages = build_prompt(make_row(), examples=[example])
    content = messages[0]["content"]
    # "Here are some examples:" intro, example image, example Q+A block, "Now answer" cue,
    # then the real image, then the real question
    texts = [c["text"] for c in content if c["type"] == "text"]
    images = [c["source"]["data"] for c in content if c["type"] == "image"]
    assert texts[0] == "Here are some examples:\n"
    assert "Answer: B" in texts[1]
    assert texts[2] == "\nNow answer the following:\n"
    assert images == [base64.b64encode(b"ex-img").decode(), IMG_B64]


def test_visual_exemplars_are_inserted_with_their_captions_before_examples():
    _, messages = build_prompt(make_row(), visual_exemplars=[(b"vis-bytes", "caption A")])
    content = messages[0]["content"]
    texts = [c["text"] for c in content if c["type"] == "text"]
    images = [c["source"]["data"] for c in content if c["type"] == "image"]
    assert "labeled in FDI tooth numbering" in texts[0]
    assert texts[1] == "caption A"
    assert texts[2] == "\nNow examine the following panoramic X-ray and answer the question:\n"
    assert images[0] == base64.b64encode(b"vis-bytes").decode()


def test_context_visual_exemplars_and_examples_all_appear_in_order():
    example = {"question": "Ex Q", "option1": "1", "option2": "2", "option3": "3",
              "option4": "4", "answer": "C", "image": base64.b64encode(b"ex-img").decode()}
    _, messages = build_prompt(
        make_row(), context="Primer!", examples=[example],
        visual_exemplars=[(b"vis-bytes", "caption A")])
    content = messages[0]["content"]
    texts_and_kinds = [(c["type"], c.get("text", "")) for c in content]
    # context text is first
    assert texts_and_kinds[0] == ("text", "Primer!\n")
    # last item is the real question, second to last is the real image
    assert content[-1]["type"] == "text"
    assert content[-2] == {"type": "image",
                           "source": {"type": "base64", "media_type": "image/jpeg", "data": IMG_B64}}
