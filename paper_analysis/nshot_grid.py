"""
GENERATOR for: PAPER_DRAFT.md §5.3.2 — the few-shot x reasoning grid (F3 / T1).

For every model x mode x k cell it RE-DERIVES the predicted letter from the committed
`raw_response` with the canonical extractor (clients.parsing.extract_letter) — it never
trusts the `predicted` column a run harness wrote, because an older harness mis-parsed
verbose gemini-3.5-flash replies (read "The correct answer is **D**" as 'A'; see
RESEARCH_PLAN parser-fix note). Re-deriving from raw output is the §1.0 rule-7 guarantee
that every paper number regenerates from raw API text with zero new API calls.

Per cell: n, accuracy [95% Wilson CI], %A, and a chi-square of the A/B/C/D counts against
an even 25/25/25/25 split (df=3; >7.815 => lopsided at p<0.05, flagged with *).

Run:   python paper_analysis/nshot_grid.py
Reads: results/nshot/closed_gemini-{2.5,3.5}-flash_k{0,1,3,5}_cleanshuf_think0[_cot].csv
Writes: paper_analysis/_generated/nshot_grid_table.md        (full 16-cell grid, for the appendix)
        paper_analysis/_generated/nshot_grid_condensed.md    (the 10-row §5.3.2 table)
        paper_analysis/_generated/nshot_grid.values.json
"""
import os
import sys
import json
import math
import datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "_generated")
sys.path.insert(0, REPO)
try:  # χ and – print fine on a utf-8 console; force it on Windows cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from clients.parsing import extract_letter, is_api_failure  # noqa: E402  (canonical, no API)

MODELS = ["2.5-flash", "3.5-flash"]
MODES = ["direct", "cot"]
KS = [0, 1, 3, 5]
CHI2_CRIT_05 = 7.815  # chi-square df=3, p=0.05
# the subset of cells the §5.3.2 condensed table shows
CONDENSED = [("2.5-flash", "direct", k) for k in KS] + \
            [("2.5-flash", "cot", 0), ("2.5-flash", "cot", 5),
             ("3.5-flash", "direct", 0), ("3.5-flash", "direct", 5),
             ("3.5-flash", "cot", 0), ("3.5-flash", "cot", 5)]


def path_for(model, mode, k):
    cot = "_cot" if mode == "cot" else ""
    return f"results/nshot/closed_gemini-{model}_k{k}_cleanshuf_think0{cot}.csv"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (c - h)), round(100 * (c + h))


def chi2_uniform(counts):
    """chi-square of A/B/C/D counts vs an even split (over the letters actually emitted)."""
    n = sum(counts)
    if n == 0:
        return 0.0
    exp = n / 4
    return sum((o - exp) ** 2 / exp for o in counts)


def cell(model, mode, k):
    df = pd.read_csv(os.path.join(REPO, path_for(model, mode, k)))
    cot = (mode == "cot")
    raws = df["raw_response"].tolist()
    # exclude infrastructure failures (a stored "max retries exceeded" is not a model
    # answer) — they are reported, not scored as wrong. Genuine "none of the above"
    # replies are real model output and STAY (correctly scored wrong on forced choice).
    n_api_fail = sum(1 for r in raws if is_api_failure(r))
    keep = [not is_api_failure(r) for r in raws]
    raws = [r for r, k_ in zip(raws, keep) if k_]
    ans = [a for a, k_ in zip(df["answer"].tolist(), keep) if k_]
    pred = [extract_letter(str(r), cot=cot) for r in raws]
    pred = [p if p in ("A", "B", "C", "D") else None for p in pred]
    n = len(raws)
    ncorrect = sum(1 for p, a in zip(pred, ans) if p == a)
    counts = {L: pred.count(L) for L in "ABCD"}
    unparse = pred.count(None)
    chi2 = chi2_uniform([counts[L] for L in "ABCD"])
    lo, hi = wilson(ncorrect, n)
    return {
        "model": model, "mode": mode, "k": k, "n": n,
        "acc": round(100 * ncorrect / n, 1), "ci": [lo, hi],
        "counts": counts, "pctA": round(100 * counts["A"] / n),
        "chi2": round(chi2, 1), "sig": chi2 > CHI2_CRIT_05,
        "unparseable": unparse, "api_failures_excluded": n_api_fail,
    }


def _row(v, best_acc):
    mode_lbl = "CoT" if v["mode"] == "cot" else "direct"
    a = f"{v['pctA']}"
    if v["sig"]:
        a = f"**{a}**"
    acc = f"{v['acc']:.1f} [{v['ci'][0]}–{v['ci'][1]}]"
    if abs(v["acc"] - best_acc) < 1e-9:
        acc = f"**{acc}**"
    star = "\\*" if v["sig"] else ""
    return (f"| {v['model']} | {mode_lbl} | {v['k']} | {acc} | {a} | {v['chi2']}{star} |")


def _table(cells, prov):
    best = max(v["acc"] for v in cells)
    lines = [prov,
             "| Model | Mode | k | Acc % [95% CI] | %A | χ² vs uniform |",
             "|-------|------|---|----------------|----|---------------|"]
    lines += [_row(v, best) for v in cells]
    return "\n".join(lines) + "\n"


def main():
    grid = {}
    for model in MODELS:
        for mode in MODES:
            for k in KS:
                grid[(model, mode, k)] = cell(model, mode, k)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    prov = (f"<!-- GENERATED by paper_analysis/nshot_grid.py on {stamp}. "
            f"Predictions RE-DERIVED from raw_response (canonical extract_letter). "
            f"Do not hand-edit; run `python paper_analysis/nshot_grid.py`. -->")

    full = [grid[(m, md, k)] for m in MODELS for md in MODES for k in KS]
    with open(os.path.join(OUT_DIR, "nshot_grid_table.md"), "w", encoding="utf-8") as f:
        f.write(_table(full, prov))
    cond = [grid[key] for key in CONDENSED]
    with open(os.path.join(OUT_DIR, "nshot_grid_condensed.md"), "w", encoding="utf-8") as f:
        f.write(_table(cond, prov))

    vals = {f"{m}|{md}|k{k}": grid[(m, md, k)] for m in MODELS for md in MODES for k in KS}
    vals["_generator"] = "paper_analysis/nshot_grid.py"
    vals["_generated_utc"] = stamp
    vals["_note"] = "predicted re-derived from raw_response via clients.parsing.extract_letter"
    with open(os.path.join(OUT_DIR, "nshot_grid.values.json"), "w", encoding="utf-8") as f:
        json.dump(vals, f, indent=2)

    print(_table(cond, prov))
    print("full 16-cell grid + condensed + values.json written to", OUT_DIR)
    # loud note on unparseable / excluded cells
    for v in full:
        if v["unparseable"]:
            print(f"  note: {v['model']} {v['mode']} k{v['k']} has {v['unparseable']} "
                  f"unparseable reply(ies) (real 'none of the above' output; scored wrong)")
        if v["api_failures_excluded"]:
            print(f"  WARN: {v['model']} {v['mode']} k{v['k']} EXCLUDED "
                  f"{v['api_failures_excluded']} API-failure row(s); scored on n={v['n']}. "
                  f"Re-elicit for a clean n=100.")


if __name__ == "__main__":
    main()
