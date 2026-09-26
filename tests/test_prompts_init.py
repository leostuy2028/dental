# Tests for prompts/__init__.py
# This file just re-exports each family's build_prompt function under a
# family-specific name, so callers can do e.g. "from prompts import gemini_prompt".

import prompts
from prompts.gemini import build_prompt as gemini_build_prompt
from prompts.gpt import build_prompt as gpt_build_prompt
from prompts.claude import build_prompt as claude_build_prompt


def test_gemini_prompt_is_the_gemini_build_prompt_function():
    assert prompts.gemini_prompt is gemini_build_prompt


def test_gpt_prompt_is_the_gpt_build_prompt_function():
    assert prompts.gpt_prompt is gpt_build_prompt


def test_claude_prompt_is_the_claude_build_prompt_function():
    assert prompts.claude_prompt is claude_build_prompt
