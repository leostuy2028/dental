"""
Text-only LLM judges for open-ended grading.

Each judge takes a full grading prompt (from rubrics.build_grading_prompt) and returns
a float score in [0.0, 1.0]. Parsing mirrors VLMEvalKit's MMOral_opg_auxeval:
  - the judge is asked only to "complete the last space" with a number;
  - we parse the last float in [0,1] from its output;
  - on an unparseable reply we retry up to 5x with RISING temperature;
  - if all retries fail, the score is 0.0 (VLMEvalKit's behaviour).

Judges:
  gpt-4o   -> OpenAI  (needs OPENAI_API_KEY)   -- closest successor to the paper's GPT-4-turbo
  gemini   -> google-genai (GEMINI_API_KEY)
  claude   -> anthropic (ANTHROPIC_API_KEY)

Usage:
  from eval_open.judges import grade
  score, raw = grade(prompt, judge="gpt-4o")
"""
import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

# rising temperature schedule across the 5 retries (VLMEvalKit style)
TEMP_SCHEDULE = [0.0, 0.2, 0.4, 0.6, 0.8]

DEFAULT_MODELS = {
    "gpt-4o": "gpt-4o",
    "gemini": "gemini-2.5-flash",
    "claude": "claude-haiku-4-5-20251001",
}

_clients = {}


# --- score parsing ----------------------------------------------------------
def parse_score(text):
    """Return the last float in [0,1] found in `text`, or None if none parses."""
    if not text:
        return None
    # matches 0, 1, 0.0-1.0, .5, etc.
    cands = re.findall(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?![\d])", text.strip())
    for tok in reversed(cands):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
    return None


# --- per-provider single calls ---------------------------------------------
def _call_openai(prompt, model, temperature):
    import openai
    if "gpt-4o" not in _clients:
        # timeout prevents a hung call; the SDK's own retry layer handles 429
        # rate limits (it honours Retry-After) so our wrapper doesn't pile on.
        _clients["gpt-4o"] = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"], timeout=60.0, max_retries=5)
    resp = _clients["gpt-4o"].chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def _call_gemini(prompt, model, temperature):
    from google import genai
    from google.genai import types
    if "gemini" not in _clients:
        _clients["gemini"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = _clients["gemini"].models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=16,
            # thinking off: otherwise 2.5/3.5-flash spend the whole budget on hidden
            # reasoning and return an empty answer. Keeps the judge a bare-number scorer.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text


def _call_claude(prompt, model, temperature):
    import anthropic
    if "claude" not in _clients:
        _clients["claude"] = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = _clients["claude"].messages.create(
        model=model,
        max_tokens=16,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


_CALLERS = {"gpt-4o": _call_openai, "gemini": _call_gemini, "claude": _call_claude}


# --- public grade() ---------------------------------------------------------
def grade(prompt, judge="gpt-4o", model=None, delay=0.0):
    """Grade one prompt with `judge`; return (score in [0,1], raw_last_reply).

    Retries 5x with rising temperature on unparseable output; 0.0 after all fail.
    Transient API errors are retried with backoff within each temperature step.
    """
    if judge not in _CALLERS:
        raise ValueError(f"unknown judge {judge!r} (one of {list(_CALLERS)})")
    model = model or DEFAULT_MODELS[judge]
    caller = _CALLERS[judge]
    last_raw = ""
    for temp in TEMP_SCHEDULE:
        for attempt in range(3):  # transient-error backoff within a temp step
            try:
                last_raw = caller(prompt, model, temp) or ""
                break
            except Exception as e:
                msg = str(e).lower()
                # non-retryable: quota/billing, missing model, auth — fail fast
                if any(t in msg for t in ("insufficient_quota", "model_not_found",
                                          "does not exist", "invalid_api_key",
                                          "authentication", "401", "403", "404")):
                    raise RuntimeError(f"[{judge}/{model}] non-retryable API error: {e}")
                wait = 2 ** attempt
                print(f"  [{judge}] API error (t={temp}, try {attempt+1}/3): {e} — {wait}s")
                time.sleep(wait)
        else:
            continue  # all 3 transient retries failed at this temp; try next temp
        score = parse_score(last_raw)
        if score is not None:
            if delay:
                time.sleep(delay)
            return score, last_raw
    return 0.0, last_raw  # unparseable after all retries -> 0.0


if __name__ == "__main__":
    # parser smoke test (no API)
    for t, exp in [("0.8", 0.8), ("Correctness: 1.0", 1.0), (" .5 ", 0.5),
                   ("the score is 0", 0.0), ("nonsense", None), ("2.0 but 0.7", 0.7)]:
        got = parse_score(t)
        print(f"parse_score({t!r}) = {got}  {'OK' if got == exp else 'FAIL exp '+str(exp)}")
