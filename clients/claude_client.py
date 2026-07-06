import os
import time
import anthropic
from dotenv import load_dotenv

from clients.parsing import extract_letter, looks_like_refusal  # noqa: F401 (re-exported)
from clients.errors import APICallFailed

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
DELAY_SECONDS = 1

_client = None


def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def extract_answer(text, cot=False):
    """Extract A/B/C/D (shared audit-safe extractor; see clients/parsing.py)."""
    return extract_letter(text, cot=cot)


def call(system, messages, model=MODEL, cot=False, retries=3):
    """
    Send messages to Claude and return (predicted_letter, raw_response).
    Raises APICallFailed if all retries are exhausted (the harness then skips the
    item rather than recording an error string as a model answer).

    Generation is greedy (temperature=0) for reproducibility — the API default is
    non-zero, so it MUST be set explicitly.
    """
    client = get_client()
    # CoT writes one sentence per option (4) + a trailing "Answer: X"; 800 can truncate a
    # verbose model (e.g. Opus) BEFORE the answer line, which the extractor then reads as a
    # non-pick and scores wrong. 2048 leaves ample room. Direct is a bare letter, but give it
    # headroom so a model that prefixes a word under the coax prompt still reaches the letter.
    max_tokens = 2048 if cot else 64

    last_err = None
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system,
                messages=messages,
            )
            raw = response.content[0].text
            time.sleep(DELAY_SECONDS)
            return extract_answer(raw, cot=cot), raw
        except Exception as e:
            last_err = e
            wait = DELAY_SECONDS * (2 ** attempt)
            print(f"  API error (attempt {attempt + 1}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)

    raise APICallFailed(f"claude {model}: {retries} retries exhausted: {last_err}")
