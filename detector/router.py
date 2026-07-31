"""
The router: decides WHO answers each question. A regex over the question text, nothing more.

Why a rule and not a model. The benchmark's questions were machine-generated from fixed
templates (paper §2), so their surface forms are highly regular and a rule matches them almost
by construction. A model-based router would also cost an API call per question, which would
spend exactly the saving the pipeline exists to demonstrate. And a regex is auditable: anyone
can read it and check what it claims.

The error costs are ASYMMETRIC, so this is deliberately conservative:

  routing a question the detector cannot answer  -> active damage. The detector answers
                                                    something irrelevant where the VLM might
                                                    have been right.
  missing one it could have answered             -> a lost opportunity, and nothing worse.

So it prefers precision over recall. Anything it is not sure about goes to the VLM.

Two capabilities are routed, and only two, because these are the only two the detector has
been measured on:

  COUNT   "how many teeth are visualized"        -> number of detected boxes
                                                    (81% exact, 94% within one tooth)
  WISDOM  "how many wisdom teeth were detected"  -> codes ending in 8 from the positional
                                                    numbering (58% exact count)

NOT routed, on purpose:

  * "how many wisdom teeth are ERUPTED" — eruption status is not something a box detector can
    read. An impacted third molar is still a detected third molar, so answering these from the
    count would be confidently wrong. This is the single biggest precision trap in the wisdom
    set and it is excluded by an explicit negative rule, not by luck.
  * "how many teeth have crowns / show periapical lesions" — needs a finding, not a count.
  * anything asking WHICH tooth has a condition — that rides on full numbering, which is
    53% on held-out DENTEX and not good enough to answer unaided.
"""
import re

# "how many teeth are visualized/visible/present/detected/seen" and nothing further
COUNT_RE = re.compile(
    r"^how many\s+(?:natural\s+|permanent\s+)?teeth\s+"
    r"(?:are|were|is|was)?\s*"
    r"(?:visualized|visualised|visible|present|detected|seen)\b[^,]*\??$", re.I)

# "how many wisdom teeth were detected/present" — the ones that are a pure enumeration
WISDOM_RE = re.compile(
    r"^how many\s+(?:wisdom teeth|third molars)\s+"
    r"(?:are|were|is|was)?\s*"
    r"(?:detected|present|visible|visualized|visualised|seen)\b[^,]*\??$", re.I)

# eruption/impaction status is NOT derivable from a box. Excludes both word orders:
# "how many wisdom teeth are erupted" and "how many erupted wisdom teeth are present".
STATUS_RE = re.compile(r"\b(erupt|impact|unerupt|partially|status|condition|suspect)", re.I)


def route(question):
    """-> 'count' | 'wisdom' | None (None means the VLM answers it)."""
    q = " ".join(str(question).split()).strip()
    if STATUS_RE.search(q):
        return None
    if COUNT_RE.match(q):
        return "count"
    if WISDOM_RE.match(q):
        return "wisdom"
    return None


def answer_mcq(kind, detector, options):
    """Pick the numeric option closest to what the detector measured.

    detector = {'count': int, 'wisdom': [fdi codes]}. Returns an index into options, or None
    if the options are not plain numbers (in which case the question is handed back to the VLM
    rather than guessed at).
    """
    vals = []
    for o in options:
        s = str(o).strip()
        if not re.fullmatch(r"\d+", s):
            return None
        vals.append(int(s))
    n = detector["count"] if kind == "count" else len(detector["wisdom"])
    return min(range(len(vals)), key=lambda i: (abs(vals[i] - n), i))


def answer_text(kind, detector):
    """The free-text form of the same answer."""
    if kind == "count":
        return f"{detector['count']} teeth are visualized in the radiograph."
    w = detector["wisdom"]
    if not w:
        return "No wisdom teeth are detected in the radiograph."
    names = ", ".join("#" + c for c in w)
    verb = "is" if len(w) == 1 else "are"
    return f"{len(w)} wisdom {'tooth' if len(w) == 1 else 'teeth'} {verb} detected: {names}."
