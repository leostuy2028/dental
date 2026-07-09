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


def _extract_text(response):
    """Return the visible answer text, skipping any thinking blocks. With extended
    thinking enabled the first content block is a `thinking` block, so response.content[0]
    is NOT the answer; we take the first `text` block instead."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def call(system, messages, model=MODEL, cot=False, thinking_budget=None, effort=None, retries=3):
    """
    Send messages to Claude and return (predicted_letter, raw_response).
    Raises APICallFailed if all retries are exhausted (the harness then skips the
    item rather than recording an error string as a model answer).

    Three thinking modes (checked in this order):
      - effort set  -> Opus-4.8-style ADAPTIVE thinking (thinking.type=adaptive +
        output_config.effort in {low,medium,high,xhigh,max}). The model self-budgets;
        temperature is left at the API default (adaptive manages sampling). max_tokens
        is generous so long high-effort reasoning is not truncated before the answer.
      - thinking_budget set -> legacy ENABLED extended thinking (Haiku/Sonnet): temp
        must be 1.0 and max_tokens must exceed budget_tokens.
      - neither -> greedy (temperature=0) for reproducibility; the API default is
        non-zero, so it MUST be set explicitly.

    CoT writes one sentence per option + a trailing "Answer: X"; give ample room so a
    verbose model is not truncated BEFORE the answer line (which the extractor then reads
    as a non-pick and scores wrong). Direct is a bare letter but still gets headroom.
    """
    client = get_client()

    # hard per-request timeout: a silent network hang otherwise blocks forever (the retry
    # loop only fires on exceptions), so cap it and let a timeout become a retryable error.
    kwargs = {"model": model, "system": system, "messages": messages, "timeout": 600.0}
    if effort is not None:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": effort}
        kwargs["max_tokens"] = 32000  # headroom for high/xhigh adaptive reasoning + answer
    elif thinking_budget is not None:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        kwargs["temperature"] = 1.0
        kwargs["max_tokens"] = thinking_budget + (2048 if cot else 512)
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = 2048 if cot else 64

    # adaptive high-effort reasoning can exceed the SDK's 10-min non-streaming guard, so
    # stream (and reassemble) whenever thinking is enabled; plain greedy calls stay unary.
    use_stream = effort is not None or thinking_budget is not None

    last_err = None
    for attempt in range(retries):
        try:
            if use_stream:
                with client.messages.stream(**kwargs) as s:
                    response = s.get_final_message()
            else:
                response = client.messages.create(**kwargs)
            raw = _extract_text(response)
            time.sleep(DELAY_SECONDS)
            return extract_answer(raw, cot=cot), raw
        except Exception as e:
            last_err = e
            wait = DELAY_SECONDS * (2 ** attempt)
            print(f"  API error (attempt {attempt + 1}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)

    raise APICallFailed(f"claude {model}: {retries} retries exhausted: {last_err}")
