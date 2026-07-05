"""
OpenAI GPT client for closed-ended MMOral-Bench MCQ.

call(system, user_content, model, cot=False, reasoning_effort="none")
    -> (predicted_letter | None, raw_response, refused_bool)

Handles the two OpenAI parameter regimes:
  - Classic chat models (gpt-4o, gpt-4.1): max_tokens + temperature=0.
  - Reasoning models (gpt-5*, o*): max_completion_tokens + reasoning_effort
    (use "none" for the direct/thinking-off condition); temperature is not sent
    (these models reject non-default temperature).
"""
import os
import re
import time
import openai
from dotenv import load_dotenv

from clients.parsing import looks_like_refusal, extract_letter  # noqa: F401 (single-sourced)
from clients.errors import APICallFailed

load_dotenv()

MODEL = "gpt-4o-2024-11-20"
DELAY_SECONDS = 0.5

_client = None


def get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                timeout=120.0, max_retries=2)
    return _client


def is_reasoning_model(model):
    return model.startswith("gpt-5") or re.match(r"^o\d", model) is not None


def extract_answer(text, cot=False):
    """Extract A/B/C/D for the COAX path via the shared, audit-safe extractor
    (clients.parsing.extract_letter). Verified to reproduce every committed GPT-4o
    coax prediction, and it drops the old substring-scan fallback that could read a
    verbose reply's first stray letter (protects the pending Opus/o-series runs).
    The FAITHFUL reproduction path does not use this — it uses vlmeval_parse."""
    return extract_letter(text, cot=cot)


def call(system, user_content, model=None, cot=False, reasoning_effort="none",
         retries=4):
    """Return the model's RAW output string. Answer extraction is done by the
    harness (mode-dependent: faithful=VLMEvalKit parser, coax=strict). An empty
    string is a real (empty) model reply; a genuine API failure raises APICallFailed
    so the harness can skip the item instead of scoring an error string as wrong."""
    client = get_client()
    model = model or MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})

    kwargs = dict(model=model, messages=messages)
    if is_reasoning_model(model):
        # reasoning models need headroom for hidden reasoning even at effort="none"
        kwargs["max_completion_tokens"] = 4000 if cot else 2000
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        # max_tokens=8192 + temperature=0 match the authors' config_mmoral_opg.json
        # (which overrides the vendored wrapper's 2048 default).
        kwargs["max_tokens"] = 8192
        kwargs["temperature"] = 0.0

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            time.sleep(DELAY_SECONDS)
            return raw
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [gpt:{model}] error (try {attempt+1}/{retries}): "
                  f"{str(e).splitlines()[0][:120]} — {wait}s")
            time.sleep(wait)

    raise APICallFailed(f"gpt {model}: {retries} retries exhausted: {last_err}")
