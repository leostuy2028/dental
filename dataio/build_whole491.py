"""
Build the two canonical GPT-4o whole-491 result files (§5.1/§5.2), each rendered the way the
benchmark actually renders blank options ("None"), by MERGING committed raw model outputs. No
new API calls.

Why a merge and not a fresh run: the 453 questions with no blank option are byte-identical
under "nan" and "None" rendering (they have no NaN), and were run at temperature 0, so their
raw replies are exactly what a "None" run would produce. Only the 38 blank-option items needed
"None"; those we ran separately. So:

  453 non-"None" rows : raw_response from the original whole-491 run (…__whole__n491, in
                        _superseded/, rendered "nan" but identical on these items)
  + 38 "None" rows    : raw_response from the blanks38-none run (_superseded/blank_answer/)
  = 491 rows, with predicted/correct RE-DERIVED against the canonical "None" data
    (faithful -> vlmeval_parse.faithful_predict with the canonical index2ans; coax -> the
    bare-letter extractor). Fully regenerable from committed raw outputs.

Run:  python dataio/build_whole491.py
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from clients.parsing import extract_letter          # noqa: E402
from utils.vlmeval_parse import faithful_predict     # noqa: E402
from utils import results_io                         # noqa: E402

BLANK_OPTION = {41, 50, 58, 73, 77, 87, 91, 105, 113, 114, 116, 125, 175, 188, 199, 205, 230,
                240, 288, 297, 327, 337, 391, 396, 435, 440, 444, 446, 455, 456, 477, 486,
                45, 48, 162, 204, 243, 325}
SUP = "results/closed_ended/_superseded"
OUT = "results/closed_ended"

RUNS = {
    "faithful": {
        "whole491_nan": f"{SUP}/gpt-4o-2024-11-20__faithful-direct-k0__whole__n491.csv",
        "blanks38_none": f"{SUP}/gpt-4o-2024-11-20__faithful-direct-k0__blanks38-none__n38.csv",
        "out": f"{OUT}/gpt-4o-2024-11-20__faithful-direct-k0__whole__n491.csv",
    },
    "coax": {
        "whole491_nan": f"{SUP}/gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv",
        "blanks38_none": f"{SUP}/gpt-4o-2024-11-20__coax-direct-k0__blanks38-none__n38.csv",
        "out": f"{OUT}/gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv",
    },
}


def build(prompt, cfg, opts):
    def rd(p):
        return pd.read_csv(os.path.join(REPO, *p.split("/")))
    whole = rd(cfg["whole491_nan"]).set_index("index")
    b38 = rd(cfg["blanks38_none"]).set_index("index")

    rows = []
    for i in opts.index:
        i = int(i)
        raw = str(b38.loc[i, "raw_response"]) if i in BLANK_OPTION else str(whole.loc[i, "raw_response"])
        answer = opts.loc[i, "answer"]
        if prompt == "faithful":
            ia = {L: str(opts.loc[i, f"option{k}"]) for L, k in zip("ABCD", [1, 2, 3, 4])}
            pred, fb = faithful_predict(raw, ia, seed=i)
        else:
            pred, fb = extract_letter(raw), False
        rows.append({
            "index": i, "file_name": opts.loc[i, "file_name"], "category": opts.loc[i, "category"],
            "question": opts.loc[i, "question"], "answer": answer, "predicted": pred,
            "raw_response": raw, "correct": pred == answer,
            "used_fallback": bool(fb), "prompt_mode": prompt,
            "source": "blanks38-none" if i in BLANK_OPTION else "whole491",
        })
    df = pd.DataFrame(rows)
    meta = {
        "experiment": "E-whole491-none", "paper_section": "§5.1/§5.2",
        "model": "gpt-4o-2024-11-20", "config": f"{prompt} direct k=0",
        "dataset": "closed_ended (canonical, blanks = 'None')", "n": len(df),
        "accuracy_pct": round(float(df["correct"].mean() * 100), 2),
        "gen_settings": "temperature=0 max_tokens=8192 img_detail=high [benchmark-faithful]",
        "description": ("MERGED from committed raw outputs (no API): 453 non-'None' rows from "
                        "the original whole-491 run (identical under nan/None), 38 'None' rows "
                        "from the blanks38-none run; predicted/correct re-derived on canonical data."),
        "source_runs": [cfg["whole491_nan"], cfg["blanks38_none"]],
        "command": "python dataio/build_whole491.py",
    }
    results_io.write_results(df, os.path.join(REPO, *cfg["out"].split("/")), meta)
    return df


def main():
    opts = pd.read_parquet(os.path.join(REPO, "data/closed_ended.parquet")).set_index("index", drop=False)
    for prompt, cfg in RUNS.items():
        df = build(prompt, cfg, opts)
        n_none = df[df["index"].isin(BLANK_OPTION)]["correct"].mean() * 100
        print(f"{prompt:9}: {cfg['out'].split('/')[-1]}  n={len(df)}  "
              f"acc={df['correct'].mean()*100:.1f}%  (on the 38 'None' items: {n_none:.1f}%)")


if __name__ == "__main__":
    main()
