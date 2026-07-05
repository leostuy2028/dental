import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

from clients.parsing import extract_letter, looks_like_refusal  # noqa: F401 (re-exported)
from clients.errors import APICallFailed

load_dotenv()

MODEL = "gemini-2.0-flash"
DELAY_SECONDS = 0  # no artificial pacing between successful calls; backoff below is separate.

_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def extract_answer(text, cot=False):
    """Extract A/B/C/D (shared audit-safe extractor; see clients/parsing.py)."""
    return extract_letter(text, cot=cot)


def call(parts, thinking_budget=None, cot=False, retries=3):
    """
    Send content parts to Gemini and return (predicted_letter, raw_response).
    Raises APICallFailed if all retries are exhausted (the harness then skips the
    item rather than recording an error string as a model answer).

    thinking_budget: if set, enables native thinking with that many thinking
    tokens (use -1 for dynamic; 0 disables thinking). Independent of `cot`.
    cot: if True, parse the answer from a 'Answer: X' line (visible per-option
    chain-of-thought) instead of the leading letter.

    Generation is greedy (temperature=0) for reproducibility — the SDK default is
    non-zero, so it MUST be set explicitly or runs are not deterministic. Default max
    output (8192) leaves room for the answer after any thinking / visible reasoning.
    """
    client = get_client()

    cfg_kwargs = {"temperature": 0.0}
    if thinking_budget is not None:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    config = types.GenerateContentConfig(**cfg_kwargs)

    last_err = None
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=MODEL, contents=parts, config=config)
            raw = response.text or ""
            if DELAY_SECONDS:
                time.sleep(DELAY_SECONDS)
            return extract_answer(raw, cot=cot), raw
        except Exception as e:
            last_err = e
            wait = 2 ** attempt  # 1, 2, 4s — real backoff (DELAY_SECONDS=0 gave none)
            print(f"  API error (attempt {attempt + 1}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)

    raise APICallFailed(f"gemini {MODEL}: {retries} retries exhausted: {last_err}")
