# Tests for prompts/gemini.py
# Builds the list of content parts sent to the Gemini API for one closed-ended
# (multiple choice) question, including optional resizing, half-crops, a
# reference-text primer, visual exemplars, and a detector "chart" hint.

import base64
import io

from PIL import Image
from google.genai import types

from prompts.gemini import (
    build_prompt, _maybe_downscale, _image_part, _half_crops, SYSTEM, COAX_SYSTEM,
)


def make_jpeg_b64(size=(20, 10), color=(255, 0, 0)):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


IMG_B64 = make_jpeg_b64()


def make_row(**overrides):
    row = {"question": "Which tooth is missing?", "option1": "opt A", "option2": "opt B",
          "option3": "opt C", "option4": "opt D", "image": IMG_B64}
    row.update(overrides)
    return row


# ---------- _maybe_downscale ----------

def test_maybe_downscale_returns_the_same_bytes_when_max_px_is_falsy():
    raw = base64.b64decode(IMG_B64)
    assert _maybe_downscale(raw, None) is raw
    assert _maybe_downscale(raw, 0) is raw


def test_maybe_downscale_leaves_a_small_image_alone():
    raw = base64.b64decode(IMG_B64)  # 20x10
    assert _maybe_downscale(raw, 1000) == raw


def test_maybe_downscale_shrinks_a_large_image():
    raw = base64.b64decode(IMG_B64)  # 20x10, longest side 20
    shrunk = _maybe_downscale(raw, 5)
    im = Image.open(io.BytesIO(shrunk))
    assert max(im.size) <= 5


# ---------- _image_part / _half_crops ----------

def test_image_part_returns_a_genai_part_with_the_bytes():
    part = _image_part(IMG_B64)
    assert isinstance(part, types.Part)
    assert part.inline_data.mime_type == "image/jpeg"
    assert part.inline_data.data == base64.b64decode(IMG_B64)


def test_half_crops_returns_two_parts():
    crops = _half_crops(IMG_B64)
    assert len(crops) == 2
    assert all(isinstance(c, types.Part) for c in crops)


# ---------- build_prompt: basic shape ----------

def test_default_prompt_starts_with_the_house_system_string():
    parts = build_prompt(make_row())
    assert parts[0] == SYSTEM


def test_coax_mode_starts_with_the_coax_system_string():
    parts = build_prompt(make_row(), mode="coax")
    assert parts[0] == COAX_SYSTEM


def test_default_prompt_has_system_image_and_question():
    parts = build_prompt(make_row())
    assert len(parts) == 3
    assert isinstance(parts[1], types.Part)
    assert "Which tooth is missing?" in parts[2]
    assert parts[2].endswith("Reply with only the letter of the correct answer: A, B, C, or D.")


def test_cot_mode_ends_with_the_answer_line_instruction():
    parts = build_prompt(make_row(), cot=True)
    assert parts[-1].endswith("Answer: A, Answer: B, Answer: C, or Answer: D.")


def test_no_image_mode_skips_the_image_and_adds_a_note():
    parts = build_prompt(make_row(), no_image=True)
    assert not any(isinstance(p, types.Part) for p in parts)
    assert any("No radiograph is provided" in p for p in parts)


def test_context_is_prepended_stripped_with_surrounding_newlines():
    parts = build_prompt(make_row(), context="  a primer  \n")
    assert parts[1] == "\na primer\n"


def test_chart_is_appended_after_the_question():
    parts = build_prompt(make_row(), chart="CHART TEXT")
    assert parts[-1] == "\nCHART TEXT\n"
    assert "Which tooth is missing?" in parts[-2]


def test_no_chart_means_no_extra_trailing_part():
    with_chart = build_prompt(make_row(), chart="x")
    without_chart = build_prompt(make_row(), chart=None)
    assert len(without_chart) == len(with_chart) - 1


def test_crops_adds_two_images_and_a_lead_in_and_follow_up_line():
    parts = build_prompt(make_row(), crops=True)
    image_parts = [p for p in parts if isinstance(p, types.Part)]
    assert len(image_parts) == 3  # main image + 2 crops
    assert any("enlarged views of the left and right halves" in p for p in parts if isinstance(p, str))
    assert any("Now answer using all three views" in p for p in parts if isinstance(p, str))


def test_examples_come_with_images_and_answers_before_the_real_question():
    example = {"question": "Ex Q", "option1": "1", "option2": "2", "option3": "3",
              "option4": "4", "answer": "B", "image": IMG_B64}
    parts = build_prompt(make_row(), examples=[example])
    strs = [p for p in parts if isinstance(p, str)]
    assert any("Here are some examples:" in s for s in strs)
    assert any("Answer: B" in s for s in strs)
    assert any("Now answer the following:" in s for s in strs)
    image_parts = [p for p in parts if isinstance(p, types.Part)]
    assert len(image_parts) == 2  # example image + real image


def test_visual_exemplars_come_with_captions_before_the_real_question():
    parts = build_prompt(make_row(), visual_exemplars=[(base64.b64decode(IMG_B64), "caption A")])
    strs = [p for p in parts if isinstance(p, str)]
    assert any("labeled in FDI tooth numbering" in s for s in strs)
    assert "caption A" in strs
    assert any("Now examine the following panoramic X-ray" in s for s in strs)
