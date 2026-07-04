import os
import re
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.0-flash"
DELAY_SECONDS = 4  # gemini-2.0-flash is NOT on the free tier (quota 0) — requires billing enabled.
                   # On paid tier RPM is high; this delay can be lowered to speed up long runs.

_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def extract_answer(text, cot=False):
    """Extract A/B/C/D from model response."""
    if cot:
        # CoT responses end with 'Answer: X' after the per-option reasoning.
        match = re.search(r'Answer:\s*([ABCD])', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    text = text.strip().upper()
    for letter in ["A", "B", "C", "D"]:
        if text.startswith(letter):
            return letter
    for letter in ["A", "B", "C", "D"]:
        if letter in text:
            return letter
    return None


def call(parts, thinking_budget=None, cot=False, retries=3):
    """
    Send content parts to Gemini and return (predicted_letter, raw_response).
    Retries on transient errors with exponential backoff.

    thinking_budget: if set, enables native thinking with that many thinking
    tokens (use -1 for dynamic; 0 disables thinking). Independent of `cot`.
    cot: if True, parse the answer from a 'Answer: X' line (visible per-option
    chain-of-thought) instead of the leading letter.

    Default max output (8192) leaves room for the answer after any thinking /
    visible reasoning, so we don't cap it here.
    """
    client = get_client()

    config = None
    if thinking_budget is not None:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )

    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=MODEL, contents=parts, config=config)
            raw = response.text or ""
            time.sleep(DELAY_SECONDS)
            return extract_answer(raw, cot=cot), raw
        except Exception as e:
            wait = DELAY_SECONDS * (2 ** attempt)
            print(f"  API error (attempt {attempt + 1}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)

    return None, "max retries exceeded"
