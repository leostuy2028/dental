"""Shared answer parsing + refusal detection for the closed-ended clients.

Single-sourced so the Claude, Gemini, and GPT clients cannot drift apart on how a
letter is pulled out of a reply (the paper's central metric is the A/B/C/D
distribution, so a sloppy extractor would manufacture the very bias we measure).

`extract_letter` is the audit-safe extractor used by the Claude and Gemini clients.
The GPT client keeps its own `extract_answer` (kept byte-for-byte to preserve the
already-committed GPT-4o reproduction numbers); it imports only `looks_like_refusal`
from here.
"""
import re

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


# legacy sentinel a pre-fix harness wrote into raw_response when all API retries failed
# (the client now raises APICallFailed instead, so new runs never contain this). A row
# carrying it is an infrastructure failure, NOT a model answer, and must be excluded from
# scoring rather than counted wrong.
_API_FAILURE_SENTINEL = "max retries exceeded"


def is_api_failure(text):
    """True if `raw_response` is a stored API-failure marker, not real model output."""
    return isinstance(text, str) and text.strip() == _API_FAILURE_SENTINEL


# explicit "this is my answer" declarations, most specific first. Each captures the
# letter. We take the letter from the LAST declaration in the reply, because a model
# that reasons out loud restates its choice at the end ("...option A is the correct
# answer. **A**"), and often lists the wrong options verbatim earlier ("C) #48  D) #44").
_ANSWER_CUES = [
    r"OPTION\s+([ABCD])\s+IS\s+(?:THE\s+)?CORRECT",          # "option A is the correct answer"
    r"CORRECT\s+(?:OPTION|CHOICE|ANSWER)\s+IS\W{0,6}([ABCD])(?![A-Z])",
    r"\bANSWER\W{0,6}(?:IS\W{0,6})?([ABCD])(?![A-Z])",       # "Answer: C", "answer is D"
]
# a bare / near-bare reply: "B", "B.", "B)", "(B)" — the fast, unambiguous path
_BARE = re.compile(r"^[(\[]?\s*([ABCD])\s*[).\]]?$")
# a standalone A/B/C/D token (letter not glued to other letters), used as last resort
_STANDALONE = re.compile(r"(?<![A-Z])([ABCD])(?![A-Z])")


def extract_letter(text, cot=False):
    """Return A/B/C/D, or None if the reply declares no answer.

    Robust to models that ignore "reply with only a letter" and write a paragraph
    that ends in their choice (gemini-3.5-flash does this). The rule mirrors how a
    human reads the reply: take the model's LAST answer declaration.

      1. bare reply ("B", "B.", "(B)")                      -> that letter
      2. else the letter of the LAST explicit answer cue
         ("Answer: C", "the correct answer is D", "option A is correct")
      3. else the LAST standalone A-D token in the reply
         (handles a trailing "**C**" or "= 29 teeth.  D")
      4. else None (a refusal or a reply with no letter at all)

    Deliberately NOT used: a scan for the first A/B/C/D character anywhere, which
    reads "The **A**nswer is D" or "**B**ased on..." as A / B from the first word and
    silently corrupts the very letter distribution this project measures. Markdown
    emphasis (``**C**``) is stripped first so a bolded letter reads like a plain one.
    A None here is scored wrong (honest); the harness's `refused` flag records why.
    """
    if cot:
        # the per-option prompt asks for a trailing "Answer: X". Strip markdown emphasis
        # first (so "**Answer: B**" / "Answer:** B" read cleanly) and take the LAST such
        # line if the model restates. A colon is optional. If the model instead writes
        # "Answer: None of the above" (a deliberate non-pick), no letter follows the cue
        # and we correctly return None — we never scavenge a letter out of the reasoning,
        # which would fabricate an answer the model declined to give.
        u = re.sub(r"[*_`]", " ", text or "")
        hits = list(re.finditer(r"\bANSWER\b\s*:?\s*([ABCD])(?![A-Za-z])", u, re.IGNORECASE))
        return hits[-1].group(1).upper() if hits else None

    raw = text or ""
    if not raw.strip():
        return None

    bare = _BARE.match(raw.strip().upper())
    if bare:
        return bare.group(1)

    # strip markdown emphasis so "**C**" / "__C__" / "`C`" read as plain letters
    u = re.sub(r"[*_`]", " ", raw).upper()

    cue_hits = []
    for pat in _ANSWER_CUES:
        cue_hits += list(re.finditer(pat, u))
    if cue_hits:
        return max(cue_hits, key=lambda m: m.start()).group(1)

    tokens = _STANDALONE.findall(u)
    if tokens:
        return tokens[-1]
    return None


if __name__ == "__main__":
    # no-API self-test. Cases marked (REAL) are verbatim gemini-3.5-flash direct
    # replies the OLD extractor misread; the expected value is the model's actual choice.
    cases = [
        ("B", "B"), ("B.", "B"), ("(C)", "C"), ("D) #21", "D"),
        ("I am unable to view images", None),          # refusal -> not 'A'
        ("As an AI, I cannot analyze", None),          # refusal -> not 'A'
        ("The answer is B", "B"),
        ("The correct answer is **D** (which corresponds to #38).", "D"),   # REAL idx93 (old->A)
        ("The correct answer is:\r\n\r\n**B**", "B"),                        # REAL idx110 (old->A)
        ("Based on the panoramic radiograph...\r\n\r\nAnswer: A", "A"),      # REAL idx67 (old->B via 'Based')
        ("...Therefore, the only erupted wisdom tooth is #18.\r\n\r\n**Correct Answer:** **C**", "C"),  # REAL idx132 (old->B)
        ("* C) #48  * D) #44  ... option A is the correct answer.  **A**", "A"),  # REAL idx121: lists C/D, answers A
        ("...= 29 teeth.\r\n\r\nD", "D"),               # REAL idx79: trailing bare letter after counting
        ("...the answer is A, not B.", "A"),           # cue beats trailing token
        ("", None), (None, None), ("no letter here", None),
    ]
    ok = 0
    for text, exp in cases:
        got = extract_letter(text)
        flag = "OK" if got == exp else "FAIL exp " + str(exp)
        ok += got == exp
        print(f"[{flag:12}] extract_letter({text!r:60.60}) = {got}")
    for text, exp in [("Answer: C", "C"), ("...\nAnswer: b", "B"), ("no answer", None)]:
        got = extract_letter(text, cot=True)
        flag = "OK" if got == exp else "FAIL exp " + str(exp)
        ok += got == exp
        print(f"[{flag:12}] extract_letter({text!r:60.60}, cot=True) = {got}")
    print(f"\n{ok}/{len(cases)+3} passed")
