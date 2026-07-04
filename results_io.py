"""
Self-describing result CSVs.

Every result file we write carries a leading block of `#`-comment lines holding the full
experiment metadata (experiment id, paper section, model, config, dataset, slice, command,
git commit, timestamp). The goal: open any CSV and know exactly what it is, even without
results/README.md.

IMPORTANT — read these files with `load_results()`, NOT plain `pd.read_csv`:
our data contains '#' (tooth numbers like "#44"), so pandas' `comment='#'` would truncate
real data. `load_results` strips ONLY the leading comment block (lines whose first
non-space char is '#'), so '#' inside the data is never touched.

    from results_io import write_results, load_results
    write_results(df, path, meta={...})
    df, meta = load_results(path, return_meta=True)
"""
import io
import subprocess
import datetime
import pandas as pd

COMMENT = "#"

# Recommended metadata keys (fill what applies; order is preserved in the file):
#   experiment, paper_section, description, model, config, dataset, slice, n,
#   judge (open-ended only), command, code_commit, generated_utc
RECOMMENDED_KEYS = [
    "experiment", "paper_section", "description", "model", "config",
    "dataset", "slice", "n", "judge", "command", "code_commit", "generated_utc",
]


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_results(df, path, meta):
    """Write `df` to `path` with a leading '#'-comment metadata header.

    `meta` is a dict of key -> value. `code_commit` and `generated_utc` are auto-filled
    if absent. Values are stringified; newlines in values are collapsed to spaces.
    """
    meta = dict(meta)
    meta.setdefault("code_commit", _git_commit())
    meta.setdefault("generated_utc", datetime.datetime.now(datetime.timezone.utc)
                    .isoformat(timespec="seconds"))
    header = "".join(
        f"{COMMENT} {k}: {str(v).replace(chr(10), ' ').strip()}\n" for k, v in meta.items()
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(header)
        df.to_csv(f, index=False, lineterminator="\n")


def load_results(path, return_meta=False):
    """Read a result CSV written by `write_results` (or a plain CSV with no header block).

    Strips only the LEADING '#'-comment lines, so '#' inside the data is preserved.
    Returns a DataFrame, or (DataFrame, meta_dict) if return_meta=True.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.readlines()
    meta, i = {}, 0
    while i < len(raw) and raw[i].lstrip().startswith(COMMENT):
        body = raw[i].lstrip()[len(COMMENT):].strip()
        if ":" in body:
            k, v = body.split(":", 1)
            meta[k.strip()] = v.strip()
        i += 1
    df = pd.read_csv(io.StringIO("".join(raw[i:])))
    return (df, meta) if return_meta else df


def read_meta(path):
    """Return just the metadata dict from a file's header block."""
    return load_results(path, return_meta=True)[1]


if __name__ == "__main__":
    # round-trip self-test, deliberately with '#' inside the data
    import os, tempfile
    df = pd.DataFrame({"index": [50, 58], "question": ["Which tooth?", "caries?"],
                       "option1": ["#44", "#45"], "answer": ["A", "D"], "correct": [True, False]})
    meta = {"experiment": "E-selftest", "paper_section": "§5.4",
            "description": "round-trip demo", "model": "gemini-3.5-flash",
            "config": "direct k=3 think-off", "dataset": "clean-shuffled",
            "slice": "idx50-149", "n": 2,
            "command": "python eval_closed_gemini.py --model gemini-3.5-flash --k 3"}
    p = os.path.join(tempfile.gettempdir(), "results_io_selftest.csv")
    write_results(df, p, meta)
    print("--- file on disk ---")
    print(open(p).read())
    back, m = load_results(p, return_meta=True)
    print("--- data survived (note '#44' intact) ---")
    print(back.to_string(index=False))
    assert list(back["option1"]) == ["#44", "#45"], "‼ '#' in data was corrupted"
    assert m["experiment"] == "E-selftest" and m["n"] == "2"
    assert "code_commit" in m and "generated_utc" in m
    print("\nOK: metadata parsed, '#'-in-data preserved, auto fields present.")
