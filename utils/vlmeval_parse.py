"""
Faithful reproduction of MMOral-Bench's closed-ended answer extraction.

`get_single_choice_prediction` is VERBATIM from VLMEvalKit
(open-compass/VLMEvalKit, vlmeval/dataset/utils/mmoral_opg.py), fetched 2026-07-04
— the exact code the benchmark/paper used to turn a model's free-form output into
an A/B/C/D choice. Note two behaviours that matter for the audit:
  1. It also matches on the OPTION TEXT (index2ans values), not just the letter.
  2. On no match it returns random.choice(all_choices) with NO seed — so refusals
     and garble are laundered into ~25% random guesses rather than scored 0.

`faithful_predict` is a reproducible instantiation: the deterministic candidate
logic is identical; the ONLY divergence is we seed the random fallback per item so
re-runs match, and we return a `used_fallback` flag so we can measure how often the
paper's pipeline is really just guessing.
"""
import random


def get_single_choice_prediction(response, all_choices, index2ans):
    # ---- VERBATIM (do not edit) ----
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = ' ' + response + ' '  # add space to avoid partial match

    candidates = []

    for choice in all_choices:
        if f'({choice})' in response:
            candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f' {choice} ' in response:
                candidates.append(choice)
            elif f' {choice}.' in response:
                candidates.append(choice)
            elif f' {choice},' in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for index, ans in index2ans.items():
            ans_str = str(ans)
            if ans_str in response:
                candidates.append(index)

    if len(candidates) > 0:
        positions = {}
        for c in candidates:
            pos = response.find(f' {c} ')
            if pos == -1:
                pos = response.find(f'({c})')
            if pos == -1:
                pos = response.find(str(index2ans[c]))
            if pos != -1:
                positions[c] = pos

        if positions:
            return min(positions.items(), key=lambda x: x[1])[0]

    return random.choice(all_choices)
    # ---- end verbatim ----


def faithful_predict(response, index2ans, seed=None, all_choices=("A", "B", "C", "D")):
    """Reproducible get_single_choice_prediction.

    Returns (letter, used_random_fallback). Deterministic candidate matching is
    identical to VLMEvalKit; the random fallback is seeded (per-item) for
    reproducibility and flagged so we can report the true parse rate.
    """
    all_choices = list(all_choices)
    resp = response or ""
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        resp = resp.strip(char)
    resp = ' ' + resp + ' '

    candidates = []
    for choice in all_choices:
        if f'({choice})' in resp:
            candidates.append(choice)
    if not candidates:
        for choice in all_choices:
            if f' {choice} ' in resp:
                candidates.append(choice)
            elif f' {choice}.' in resp:
                candidates.append(choice)
            elif f' {choice},' in resp:
                candidates.append(choice)
    if not candidates:
        for index, ans in index2ans.items():
            if str(ans) in resp:
                candidates.append(index)

    if candidates:
        positions = {}
        for c in candidates:
            pos = resp.find(f' {c} ')
            if pos == -1:
                pos = resp.find(f'({c})')
            if pos == -1:
                pos = resp.find(str(index2ans[c]))
            if pos != -1:
                positions[c] = pos
        if positions:
            return min(positions.items(), key=lambda x: x[1])[0], False

    return random.Random(seed).choice(all_choices), True
