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


# phrases that indicate the model declined rather than answered
_REFUSAL_MARKERS = (
    "i'm unable", "i am unable", "i cannot", "i can't", "unable to view",
    "unable to analyze", "can't analyze", "cannot analyze", "as an ai",
    "i'm sorry", "i am sorry", "consult a", "seek professional",
    "not able to", "i'm not able", "cannot provide", "can't provide",
    "unable to provide", "i must decline", "i won't",
)


def looks_like_refusal(text):
    t = (text or "").lower()
    return any(m in t for m in _REFUSAL_MARKERS)


def extract_answer(text, cot=False):
    """Extract A/B/C/D. CoT -> trailing 'Answer: X'; else leading/first letter."""
    if cot:
        m = re.search(r"Answer:\s*([ABCD])", text or "", re.IGNORECASE)
        return m.group(1).upper() if m else None
    t = (text or "").strip().upper()
    if not t:
        return None
    # prefer a standalone letter token (e.g. "B", "B.", "B)", "(B)")
    m = re.search(r"(?:^|[^A-Z])([ABCD])(?:[^A-Z]|$)", t)
    if m:
        return m.group(1)
    for L in "ABCD":
        if L in t:
            return L
    return None


def call(system, user_content, model=None, cot=False, reasoning_effort="none",
         retries=4):
    """Return the model's RAW output string. Answer extraction is done by the
    harness (mode-dependent: faithful=VLMEvalKit parser, coax=strict). Returns
    "" only if all retries fail."""
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

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            time.sleep(DELAY_SECONDS)
            return raw
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [gpt:{model}] error (try {attempt+1}/{retries}): "
                  f"{str(e).splitlines()[0][:120]} — {wait}s")
            time.sleep(wait)

    return ""
