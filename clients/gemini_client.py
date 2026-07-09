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
# Explicit output cap, pinned (not the SDK default) so it is reproducible and matches the GPT-4o
# harness (clients/gpt_client.py BENCHMARK_MAX_TOKENS). Non-binding for direct/thinking-off coax
# runs (bare-letter replies), but load-bearing for CoT / thinking-on runs, so it must be explicit.
MAX_OUTPUT_TOKENS = 8192

_client = None


def get_client():
    global _client
    if _client is None:
        # per-request HTTP timeout (ms): a silent hang otherwise blocks the run forever
        # (the retry loop only fires on exceptions); a timeout becomes a retryable error.
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                               http_options=types.HttpOptions(timeout=180_000))
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

    Generation is greedy (temperature=0) for reproducibility (the SDK default is
    non-zero, so it MUST be set explicitly or runs are not deterministic). max_output_tokens
    is pinned to MAX_OUTPUT_TOKENS (8192), matching the GPT-4o harness, rather than left to the
    SDK default; this leaves room for the answer after any thinking / visible reasoning.
    """
    client = get_client()

    cfg_kwargs = {"temperature": 0.0, "max_output_tokens": MAX_OUTPUT_TOKENS}
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
