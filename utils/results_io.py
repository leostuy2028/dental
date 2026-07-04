"""
Self-describing result files: a pristine CSV + a JSON metadata sidecar.

Each result <name>.csv is written together with <name>.csv.meta.json holding the full
experiment metadata (experiment id, paper section, model, config, dataset, slice, command,
git commit, timestamp). The CSV itself stays 100% standard — no comment lines — so it opens
correctly in plain pandas, Excel, or grep, and there is NO chance of confusing metadata with
the data (which contains '#', e.g. tooth numbers like "#44").

    from utils.results_io import write_results, load_results
    write_results(df, "results/closed_ended/nshot/....csv", meta={...})
    df, meta = load_results(path, return_meta=True)   # reads the CSV + its sidecar

Goal: any file is fully reconstructable on its own (via its sidecar), and results/README.md
is the searchable index across all of them.
"""
import os
import json
import subprocess
import datetime
import pandas as pd

META_SUFFIX = ".meta.json"

# Recommended metadata keys (fill what applies):
#   experiment, paper_section, description, model, config, dataset, slice, n,
#   judge (open-ended only), command, code_commit, generated_utc
RECOMMENDED_KEYS = [
    "experiment", "paper_section", "description", "model", "config",
    "dataset", "slice", "n", "judge", "command", "code_commit", "generated_utc",
]


def meta_path(csv_path):
    """Sidecar path for a given result CSV: foo.csv -> foo.csv.meta.json."""
    return csv_path + META_SUFFIX


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_results(df, path, meta):
    """Write a pristine CSV at `path` and its metadata sidecar at `path + '.meta.json'`.

    `code_commit` and `generated_utc` auto-fill if absent. Both files must be committed
    together (and share a manifest row in results/README.md).
    """
    meta = dict(meta)
    meta.setdefault("code_commit", _git_commit())
    meta.setdefault("generated_utc",
                    datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    meta.setdefault("data_file", os.path.basename(path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)                     # standard CSV, nothing prepended
    with open(meta_path(path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_results(path, return_meta=False):
    """Read a result CSV (plain `pd.read_csv`) and, if present, its `.meta.json` sidecar.

    Returns a DataFrame, or (DataFrame, meta_dict) if return_meta=True. `meta` is {} when
    no sidecar exists.
    """
    df = pd.read_csv(path)
    if not return_meta:
        return df
    mp = meta_path(path)
    meta = {}
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    return df, meta


def read_meta(path):
    """Return just the metadata dict for a result CSV (from its sidecar), or {} if none."""
    mp = meta_path(path)
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    # round-trip self-test, deliberately with '#' inside the data
    import tempfile
    df = pd.DataFrame({"index": [50, 58], "question": ["Which tooth?", "caries?"],
                       "option1": ["#44", "#45"], "answer": ["A", "D"], "correct": [True, False]})
    meta = {"experiment": "E-selftest", "paper_section": "§5.4",
            "description": "sidecar round-trip demo", "model": "gemini-3.5-flash",
            "config": "direct k=3 think-off", "dataset": "clean-shuffled",
            "slice": "idx50-149", "n": 2,
            "command": "python eval_closed_gemini.py --model gemini-3.5-flash --k 3"}
    p = os.path.join(tempfile.gettempdir(), "results_io_selftest.csv")
    write_results(df, p, meta)
    print("--- CSV on disk (pristine, no comment lines) ---")
    print(open(p).read())
    print("--- sidecar", os.path.basename(meta_path(p)), "---")
    print(open(meta_path(p)).read())
    # the CSV must be readable by *plain* pandas with no special args
    plain = pd.read_csv(p)
    assert list(plain["option1"]) == ["#44", "#45"], "‼ data changed"
    df2, m = load_results(p, return_meta=True)
    assert list(df2["option1"]) == ["#44", "#45"]
    assert m["experiment"] == "E-selftest" and m["n"] == 2
    assert "code_commit" in m and "generated_utc" in m
    print("OK: pristine CSV reads with plain pd.read_csv; '#44' intact; sidecar metadata parsed.")
