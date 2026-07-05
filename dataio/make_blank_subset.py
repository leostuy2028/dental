"""
Build the 38-item blank-option subset of MMOral-OPG-Closed, plus a human-readable manifest.

These are the ONLY items whose prompt differs between our old "nan" rendering and the
benchmark's true "None" rendering (MMOral_OPG_CLOSED.post_build -> fillna('None')). We carve
them into their own parquet so a targeted re-run (and later runs of OTHER models on the exact
same questions) costs ~38 calls instead of 491. The subset keeps the raw NaN options intact;
the harness renders them as "None" at prompt-build time (prompts/gpt.py::_opt).

Blank items are derived by content (any option is NaN), not a hardcoded list, so the set is
self-verifying. Run:  python dataio/make_blank_subset.py
"""
import os
import pandas as pd

SRC = "data/closed_ended.parquet"
OUT_PARQUET = "data/closed_ended_blanks38.parquet"
OUT_MANIFEST = "results/closed_ended/blank_answer/blanks38_manifest.csv"
OPTS = ["option1", "option2", "option3", "option4"]
L = ["A", "B", "C", "D"]


def opt(v):
    return "None" if (v is None or (isinstance(v, float) and v != v)) else str(v)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_parquet(os.path.join(repo, SRC))
    blank = df[OPTS].isna().any(axis=1)
    sub = df[blank].reset_index(drop=True).copy()
    assert len(sub) == 38, f"expected 38 blank-option items, got {len(sub)}"

    os.makedirs(os.path.join(repo, os.path.dirname(OUT_PARQUET)) or ".", exist_ok=True)
    sub.to_parquet(os.path.join(repo, OUT_PARQUET), index=False)

    # human-readable manifest: exactly what the benchmark shows the model (options with "None")
    rows = []
    for _, r in sub.iterrows():
        blank_pos = [L[i] for i, c in enumerate(OPTS) if pd.isna(r[c])]
        correct_is_blank = r["answer"] in blank_pos
        rows.append({
            "index": int(r["index"]),
            "category": r["category"],
            "answer_key": r["answer"],
            "correct_is_blank": correct_is_blank,
            "blank_position": ",".join(blank_pos),
            "question": r["question"],
            "A": opt(r["option1"]), "B": opt(r["option2"]),
            "C": opt(r["option3"]), "D": opt(r["option4"]),
        })
    man = pd.DataFrame(rows).sort_values("index")
    os.makedirs(os.path.join(repo, os.path.dirname(OUT_MANIFEST)), exist_ok=True)
    man.to_csv(os.path.join(repo, OUT_MANIFEST), index=False)

    print(f"wrote {OUT_PARQUET}: {len(sub)} items")
    print(f"wrote {OUT_MANIFEST}")
    print(f"  correct answer is the blank option: {int(man['correct_is_blank'].sum())} items")
    print(f"  blank is a distractor only:         {int((~man['correct_is_blank']).sum())} items")
    print(f"  blank position counts: {man['blank_position'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
