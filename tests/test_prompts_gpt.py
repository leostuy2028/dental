# Tests for prompts/gpt.py
# Builds the (system, content) payload sent to the OpenAI chat API. Two prompt
# modes: "faithful" (the benchmark's own verbatim wording, no persona) and
# "coax" (our engineered prompt that forces a single-letter answer).

import base64

import pytest

from prompts.gpt import build_prompt, _fmt, _image_part, _text_part, FAITHFUL_BLOCK

IMG_B64 = base64.b64encode(b"imgbytes").decode()


def make_row(**overrides):
    row = {"question": "Which tooth is missing?", "option1": "opt A", "option2": "opt B",
          "option3": "opt C", "option4": "opt D", "image": IMG_B64}
    row.update(overrides)
    return row


# ---------- small helpers ----------

def test_image_part_defaults_to_high_detail():
    part = _image_part("abc123")
    assert part == {"type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,abc123", "detail": "high"}}


def test_image_part_can_use_a_different_detail():
    part = _image_part("abc123", detail="low")
    assert part["image_url"]["detail"] == "low"


def test_text_part_wraps_the_string():
    assert _text_part("hi") == {"type": "text", "text": "hi"}


def test_fmt_fills_in_the_template_with_stringified_options():
    row = make_row(option1=1, option2=None)
    out = _fmt(row, "{question} | {option1} | {option2}")
    assert out == "Which tooth is missing? | 1 | None"


# ---------- build_prompt: mode selection ----------

def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError, match="unknown prompt mode: bogus"):
        build_prompt(make_row(), mode="bogus")


def test_faithful_mode_has_no_system_persona():
    system, content = build_prompt(make_row(), mode="faithful")
    assert system is None


def test_faithful_mode_uses_the_verbatim_benchmark_wording():
    _, content = build_prompt(make_row(), mode="faithful")
    text = content[-1]["text"]
    assert text == _fmt(make_row(), FAITHFUL_BLOCK)
    assert "Options:\nA. opt A" in text


def test_coax_mode_has_the_radiologist_persona():
    system, _ = build_prompt(make_row(), mode="coax")
    assert system == "You are an expert dental radiologist taking a multiple-choice exam."


def test_coax_mode_instructs_never_to_refuse():
    _, content = build_prompt(make_row(), mode="coax")
    text = content[-1]["text"]
    assert "never refuse" in text
    assert text.endswith("Output only that letter and nothing else.")


def test_coax_cot_mode_asks_for_an_answer_line():
    _, content = build_prompt(make_row(), mode="coax", cot=True)
    text = content[-1]["text"]
    assert text.endswith("Answer: A, Answer: B, Answer: C, or Answer: D.")


def test_faithful_mode_ignores_cot():
    # cot is documented as unused in faithful mode
    _, content_plain = build_prompt(make_row(), mode="faithful", cot=False)
    _, content_cot = build_prompt(make_row(), mode="faithful", cot=True)
    assert content_plain[-1]["text"] == content_cot[-1]["text"]


# ---------- content structure ----------

def test_content_has_the_image_then_the_question():
    _, content = build_prompt(make_row(), mode="coax")
    assert len(content) == 2
    assert content[0] == {"type": "image_url",
                          "image_url": {"url": f"data:image/jpeg;base64,{IMG_B64}", "detail": "high"}}
    assert content[1]["type"] == "text"


def test_context_is_prepended_stripped_as_its_own_text_block():
    _, content = build_prompt(make_row(), mode="coax", context="  primer text  \n")
    assert content[0] == {"type": "text", "text": "primer text\n"}


def test_chart_is_appended_after_the_question_text():
    _, content = build_prompt(make_row(), mode="coax", chart="CHART!")
    assert content[-1]["text"].endswith("\n\nCHART!\n")


def test_no_chart_means_the_question_text_is_unchanged():
    _, with_chart = build_prompt(make_row(), mode="coax", chart=None)
    assert not with_chart[-1]["text"].endswith("\n")


def test_examples_are_inserted_with_answers_before_the_real_image():
    example = {"question": "Ex Q", "option1": "1", "option2": "2", "option3": "3",
              "option4": "4", "answer": "B", "image": base64.b64encode(b"ex-img").decode()}
    _, content = build_prompt(make_row(), mode="coax", examples=[example])
    texts = [c["text"] for c in content if c["type"] == "text"]
    images = [c["image_url"]["url"] for c in content if c["type"] == "image_url"]
    assert texts[0] == "Here are some examples:\n"
    assert "Answer: B" in texts[1]
    assert texts[2] == "\nNow answer the following:\n"
    assert images == [f"data:image/jpeg;base64,{base64.b64encode(b'ex-img').decode()}",
                      f"data:image/jpeg;base64,{IMG_B64}"]


def test_examples_use_the_given_detail_for_their_images():
    example = {"question": "Ex Q", "option1": "1", "option2": "2", "option3": "3",
              "option4": "4", "answer": "B", "image": base64.b64encode(b"ex-img").decode()}
    _, content = build_prompt(make_row(), mode="coax", examples=[example], detail="low")
    image_parts = [c for c in content if c["type"] == "image_url"]
    assert all(p["image_url"]["detail"] == "low" for p in image_parts)
