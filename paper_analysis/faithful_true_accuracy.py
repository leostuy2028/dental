"""
GENERATOR for: PAPER_DRAFT.md §5.2 — parsing vs prompting.

Establishes GPT-4o's TRUE accuracy under the benchmark's own ("faithful") prompt on the
clean 453, by reading the letter the model actually committed to in each verbose reply, and
contrasts it with (a) what the benchmark's own parser (faithful_predict) scores and (b) the
coax prompt, whose bare-letter replies the benchmark's parser reads exactly.

"True" = a high-confidence auto-read of each reply's stated answer (last explicit "answer is
X" cue, else a bolded letter) PLUS a committed hand-label for every reply that has no such
unambiguous marker (paper_analysis/faithful_hand_labels.csv). No random guessing: a reply
that states no letter (a refusal) is scored wrong, not assigned a random one.

Run:   python paper_analysis/faithful_true_accuracy.py
Reads: results/closed_ended/reproduction/...faithful...whole__n491.csv  (verbose replies)
       results/closed_ended/prompt_axis/...coax...whole__n491.csv        (bare-letter replies)
       paper_analysis/faithful_hand_labels.csv                          (committed hand reads)
Writes: paper_analysis/_generated/faithful_true_accuracy.values.json + _table.md
"""
import os
import re
import sys
import json
import math
import datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "_generated")
sys.path.insert(0, REPO)
from utils.vlmeval_parse import faithful_predict  # noqa: E402
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BLANK = {41, 50, 58, 73, 77, 87, 91, 105, 113, 114, 116, 125, 175, 188, 199, 205, 230, 240,
         288, 297, 327, 337, 391, 396, 435, 440, 444, 446, 455, 456, 477, 486,
         45, 48, 162, 204, 243, 325}
FAITHFUL = "results/closed_ended/reproduction/gpt-4o-2024-11-20__faithful-direct-k0__whole__n491.csv"
COAX = "results/closed_ended/prompt_axis/gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv"


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def mcnemar(a_right, b_right):
    """Paired test on the SAME items: a_right/b_right are per-item booleans.
    Returns (b, c, chi2, p) where b = a-wrong-b-right, c = a-right-b-wrong."""
    b = sum(1 for x, y in zip(a_right, b_right) if (not x) and y)
    c = sum(1 for x, y in zip(a_right, b_right) if x and (not y))
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p = math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0
    return b, c, round(chi2, 2), p


def high_conf(raw):
    """The letter the model UNAMBIGUOUSLY committed to, or None (needs a hand-label).
    Only the strongest two signals: the last explicit 'answer is X' cue, then a bolded
    letter. Never scans loose prose (which is what mis-reads an article 'a' as 'A')."""
    u = str(raw)
    cues = list(re.finditer(r'(?:correct\s+)?answer\W{0,4}(?:is\W{0,4})?[\*_\s]{0,4}([ABCD])(?![A-Za-z0-9])', u, re.I))
    if cues:
        return cues[-1].group(1).upper()
    bolds = list(re.finditer(r'\*\*\s*([ABCD])[\.\):\s]', u))
    if bolds:
        return bolds[-1].group(1).upper()
    return None


def paper_pick(i, raw, opts):
    ia = {L: str(opts.loc[i, f"option{n}"]) for L, n in zip("ABCD", [1, 2, 3, 4])}
    return faithful_predict(str(raw), ia, seed=int(i))


def main():
    opts = pd.read_parquet(os.path.join(REPO, "data/closed_ended.parquet")).set_index("index")
    fa = pd.read_csv(os.path.join(REPO, *FAITHFUL.split("/")))
    fa = fa[~fa["index"].isin(BLANK)].set_index("index")
    labels = pd.read_csv(os.path.join(HERE, "faithful_hand_labels.csv")).set_index("index")

    true, paper, misread = {}, {}, 0
    need_label = []
    for i, r in fa.iterrows():
        hc = high_conf(r["raw_response"])
        if hc is None:
            if i not in labels.index:
                need_label.append(i)
                continue
            v = labels.loc[i, "true_letter"]
            hc = None if (isinstance(v, float) or str(v).strip() == "") else str(v).strip().upper()
        true[i] = hc
        pp, _ = paper_pick(i, r["raw_response"], opts)
        paper[i] = pp
        if pp != hc:
            misread += 1
    if need_label:
        raise SystemExit(f"Missing hand-labels for ambiguous rows: {need_label} "
                         f"(add them to faithful_hand_labels.csv)")

    n = len(true)
    ans = fa.loc[list(true), "answer"]
    true_correct = sum(1 for i in true if true[i] == fa.loc[i, "answer"])
    paper_correct = sum(1 for i in true if paper[i] == fa.loc[i, "answer"])
    refusals = sum(1 for i in true if true[i] is None)

    # coax + the SAME (benchmark) parser: bare letters, so the parser is exact
    cx = pd.read_csv(os.path.join(REPO, *COAX.split("/")))
    cx = cx[~cx["index"].isin(BLANK)].set_index("index")
    coax_bare = sum(1 for r in cx["raw_response"] if re.fullmatch(r'\s*[ABCD][.)]?\s*', str(r)))
    coax_fallback = 0
    coax_right = {}
    for i, r in cx.iterrows():
        pp, fb = paper_pick(i, r["raw_response"], opts)
        coax_right[i] = (pp == cx.loc[i, "answer"])
        coax_fallback += bool(fb)
    coax_paper_correct = sum(coax_right.values())

    # McNemar on the SAME items: benchmark pipeline (faithful prompt + its parser) vs
    # our pipeline (coax prompt + its parser). Reproduces the retired prompt_axis test.
    common = [i for i in paper if i in coax_right]
    faith_right = [paper[i] == fa.loc[i, "answer"] for i in common]
    cx_right = [coax_right[i] for i in common]
    mc_b, mc_c, mc_chi2, mc_p = mcnemar(faith_right, cx_right)

    # report the two component effects as differences of the ROUNDED accuracies, so the
    # paper's "5.5 + 3.7 = 9.2" adds up exactly against the displayed table.
    faithful_paper = round(100 * paper_correct / n, 1)
    faithful_true = round(100 * true_correct / n, 1)
    coax_paper = round(100 * coax_paper_correct / len(cx), 1)
    vals = {
        "n_clean": n,
        "faithful_paper_parser_acc": faithful_paper,
        "faithful_paper_parser_ci": wilson(paper_correct, n),
        "faithful_true_acc": faithful_true,
        "faithful_true_ci": wilson(true_correct, n),
        "faithful_refusals": refusals,
        "parser_misreads": misread,
        "parser_misread_pct": round(100 * misread / n, 1),
        "parser_cost_pts": round(faithful_true - faithful_paper, 1),
        "coax_bare_replies": coax_bare, "coax_n": len(cx),
        "coax_paper_parser_acc": coax_paper,
        "coax_paper_parser_ci": wilson(coax_paper_correct, len(cx)),
        "coax_paper_parser_fallbacks": coax_fallback,
        "prompt_effect_true_pts": round(coax_paper - faithful_true, 1),
        "combined_pts": round(coax_paper - faithful_paper, 1),
        "mcnemar_combined": {"coax_right_faithful_wrong": mc_b, "faithful_right_coax_wrong": mc_c,
                             "chi2": mc_chi2, "p": mc_p},
        "_generator": "paper_analysis/faithful_true_accuracy.py",
        "_source_csv": [FAITHFUL, COAX, "paper_analysis/faithful_hand_labels.csv"],
        "_generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    prov = (f"<!-- GENERATED by paper_analysis/faithful_true_accuracy.py on {vals['_generated_utc']}. "
            f"True = high-confidence read + committed hand-labels. Do not hand-edit. -->")
    def ci(key):
        lo, hi = vals[key]
        return f"{lo} to {hi}"
    table = "\n".join([
        prov,
        "| GPT-4o on the clean 453 | Accuracy | 95% Wilson range |",
        "|---|--:|:--|",
        f"| Benchmark prompt, scored by the benchmark's parser | {vals['faithful_paper_parser_acc']}% | {ci('faithful_paper_parser_ci')} |",
        f"| Benchmark prompt, scored by the model's true answer (hand-verified) | {vals['faithful_true_acc']}% | {ci('faithful_true_ci')} |",
        f"| Coax prompt, scored by the benchmark's parser | {vals['coax_paper_parser_acc']}% | {ci('coax_paper_parser_ci')} |",
        "",
    ])
    with open(os.path.join(OUT_DIR, "faithful_true_accuracy_table.md"), "w", encoding="utf-8") as f:
        f.write(table)
    with open(os.path.join(OUT_DIR, "faithful_true_accuracy.values.json"), "w", encoding="utf-8") as f:
        json.dump(vals, f, indent=2)

    print(table)
    print(f"parser misreads: {misread}/{n} ({vals['parser_misread_pct']}%); "
          f"parser cost {vals['parser_cost_pts']} pts (true {vals['faithful_true_acc']} vs parser {vals['faithful_paper_parser_acc']})")
    print(f"coax: {coax_bare}/{len(cx)} bare replies; benchmark parser {vals['coax_paper_parser_acc']}% "
          f"with {coax_fallback} random fallbacks")
    print(f"true prompt effect (coax - faithful, both scored honestly): {vals['prompt_effect_true_pts']:+} pts")
    print(f"McNemar (benchmark pipeline vs coax pipeline): coax-right={mc_b}, faithful-right={mc_c}, "
          f"chi2={mc_chi2}, p={mc_p:.2e}")


if __name__ == "__main__":
    main()
