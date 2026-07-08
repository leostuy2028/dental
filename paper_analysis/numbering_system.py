"""
§5.4 — the FDI / US-Universal tooth-numbering finding.

`main()` runs the NO-API "blast radius" scans that the paper reports (closed + open),
straight from the released data + the committed gemini-3.5 shuffled baseline.

The two model probes that established the behaviour are API-based and kept here for
reproducibility (run with `python paper_analysis/numbering_system.py probe`):
  - knowledge_probe: with the system NAMED, can the model map a tooth to its code?
    (2026-07-07: gpt-4o 100/100, gemini-2.5 100/88, gemini-3.5 100/94, claude-haiku 94/56 — FDI/Universal %)
  - default_probe:   on a BARE ambiguous code (no system stated), which system does it assume?
    (2026-07-07: gpt-4o 8/8 Universal, gemini-3.5 7/8, gemini-2.5 6/8 — models DEFAULT to US Universal)
"""
import os
import re
import sys
import pandas as pd

CLOSED = "data/closed_ended.parquet"
OPEN = "data/open_ended.parquet"
BASELINE = "results/closed_ended/position_bias/gemini-3.5-flash__coax-direct-k0__shuffled__n491.csv"


def fdi_valid(n):
    t, o = divmod(n, 10)
    return n >= 11 and t in range(1, 9) and o in range(1, 9)


def blast_radius_closed(repo):
    cl = pd.read_parquet(os.path.join(repo, CLOSED)).set_index("index", drop=False)
    base = pd.read_csv(os.path.join(repo, BASELINE)).set_index("index")

    def codes(i):
        r = cl.loc[i]
        txt = " ".join(str(r[c]) for c in ["question", "option1", "option2", "option3", "option4"])
        return [int(x) for x in re.findall(r"#(\d{1,2})\b", txt)]

    def tier(cs):
        amb = [c for c in cs if fdi_valid(c) and c <= 32]   # valid in BOTH systems, different teeth
        fdi_only = [c for c in cs if fdi_valid(c) and c > 32]  # >32 -> FDI-only, signals FDI
        if not (amb or fdi_only):
            return "NO_CODES"
        if amb and not fdi_only:
            return "HIGH_RISK"
        if amb and fdi_only:
            return "DISAMBIGUATED"
        return "SAFE_FDI"

    cl["tier"] = [tier(codes(i)) for i in cl.index]
    n = len(cl)
    print(f"CLOSED ({n} questions):")
    for t in ["HIGH_RISK", "DISAMBIGUATED", "SAFE_FDI", "NO_CODES"]:
        idx = cl.index[cl.tier == t]
        acc = base.loc[base.index.isin(idx), "correct"].mean() * 100
        print(f"  {t:14} {len(idx):4} ({100*len(idx)/n:2.0f}%)   gemini-3.5 acc {acc:.1f}%")


def blast_radius_open(repo):
    op = pd.read_parquet(os.path.join(repo, OPEN))

    def ref_codes(ans):
        s = str(ans)
        return ([int(x) for x in re.findall(r"#(\d{1,2})\b", s)]
                + [int(x) for x in re.findall(r'tooth_id"?\s*:\s*"?(\d{1,2})', s)])

    op["codes"] = op["answer"].map(ref_codes)
    has = op["codes"].map(len) > 0
    fdi_only = op["codes"].map(lambda cs: any(fdi_valid(c) and c > 32 for c in cs))
    n = len(op)
    print(f"\nOPEN ({n} questions):")
    print(f"  reference contains a tooth code:        {int(has.sum()):4} ({100*has.mean():.0f}%)")
    print(f"  reference has an FDI-only (>32) code:   {int(fdi_only.sum()):4} ({100*fdi_only.mean():.0f}%)"
          f"  <- a Universal-writing model can never reproduce these")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        print("API probes are recorded in the module docstring / PAPER_DRAFT §5.4; "
              "re-run manually against the clients if needed.")
        return
    blast_radius_closed(repo)
    blast_radius_open(repo)


if __name__ == "__main__":
    main()
