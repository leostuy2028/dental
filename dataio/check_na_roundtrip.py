"""
Guard against the "None" -> NaN round-trip (RESEARCH_PLAN §3.7).

THE TRAP. 38 closed questions have the string "None" as a genuine answer option, and the
data layer deliberately normalises them that way (dataio/convert_to_none.py). The string
"None" is also in pandas' DEFAULT missing-value list. So any plain

    pd.read_csv(path)

silently turns those real options into NaN, and str(NaN) renders "nan". The canonical
parquet being clean does not help: the bug is recreated at every CSV boundary.

This has now been found three separate times: once as the original reproduction bug that
cost us a retracted finding, once in export_survey_bundle.py (fixed locally, with a
comment), and once in the question-quality survey, where it reached a page a dentist was
about to be sent. The data layer is guarded; the CSV boundary was not. This script is the
systemic guard.

WHAT IT DOES. Reads every committed CSV twice, once with pandas' defaults and once with
keep_default_na=False, and reports any cell that is a non-empty string in the second read
and NaN in the first. Those are real values a plain read would destroy.

Exit code 1 if anything is found in a file that feeds an analysis, so it can gate a
commit. Files that are pure records (written, never read back) are listed but allowed.

Run:  python -m dataio.check_na_roundtrip
"""
import glob
import os
import sys
import warnings

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Written as a record of what was built; nothing reads them back as input, so a value lost
# on re-read cannot reach an analysis. Keep this list short and justify every entry.
RECORD_ONLY = {
    "results/closed_ended/blanks38_manifest.csv":
        "record only: written by dataio/make_blank_subset.py, never read back (grep-verified)",
    "results/dentist_audit/boneloss_manifest.csv":
        "read only by paper_analysis/boneloss_audit.py, which passes keep_default_na=False",
    "results/dentist_audit/quality_manifest.csv":
        "read only by dataio/export_quality_survey.py and paper_analysis/quality_audit.py, "
        "both of which pass keep_default_na=False",
}


def scan():
    warnings.filterwarnings("ignore")
    bad, benign = [], []
    for f in sorted(glob.glob(os.path.join(REPO, "**", "*.csv"), recursive=True)):
        rel = os.path.relpath(f, REPO).replace("\\", "/")
        if "/.venv/" in f or "/_superseded/" in rel:
            continue
        try:
            a = pd.read_csv(f, dtype=str)
            b = pd.read_csv(f, dtype=str, keep_default_na=False)
        except Exception:
            continue
        if a.shape != b.shape:
            continue
        for c in a.columns:
            mask = a[c].isna() & (b[c].astype(str).str.strip() != "")
            n = int(mask.sum())
            if n:
                vals = b.loc[mask, c].value_counts().head(3).to_dict()
                (benign if rel in RECORD_ONLY else bad).append((rel, c, n, vals))
    return bad, benign


def main():
    bad, benign = scan()
    if benign:
        print("Known and handled (every reader guards, or the file is never read back):")
        for rel, c, n, vals in benign:
            print(f"  {rel}  column '{c}': {n} cells {vals}")
            print(f"      reason: {RECORD_ONLY[rel]}")
        print()
    if not bad:
        print("OK: no CSV that feeds an analysis loses a real value to pandas' default NA list.")
        return 0
    print("FAIL: these CSVs hold values that a plain pd.read_csv would destroy.")
    print("Read them with keep_default_na=False, or take the text from the parquet instead.\n")
    for rel, c, n, vals in bad:
        print(f"  {rel}  column '{c}': {n} cells -> {vals}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
