"""
Build the 38-item "None"-option subset, STRICTLY from the canonical None-normalized source
`data/closed_ended.parquet`. These are the items that carry a blank option (rendered "None"),
carved out so a targeted run — GPT-4o now, other models later — costs ~38 calls, not 491.

The 38 indices are the source-data artifact verified three ways in RESEARCH_PLAN §3.7. We
select them from the canonical set and assert each really does carry a "None" option, so the
subset is self-checking without ever touching the raw NaN file.

Run:  python dataio/make_blank_subset.py
Writes: data/closed_ended_blanks38.parquet
        results/closed_ended/blank_answer/blanks38_manifest.csv
"""
import os
import pandas as pd

CANONICAL = "data/closed_ended.parquet"
OUT_PARQUET = "data/closed_ended_blanks38.parquet"
OUT_MANIFEST = "results/closed_ended/blank_answer/blanks38_manifest.csv"
OPTS = ["option1", "option2", "option3", "option4"]
L = ["A", "B", "C", "D"]

# the 38 blank-option items (RESEARCH_PLAN §3.7, verified vs 3 independent copies)
BLANK_CORRECT = {41, 50, 58, 73, 77, 87, 91, 105, 113, 114, 116, 125, 175, 188, 199, 205, 230,
                 240, 288, 297, 327, 337, 391, 396, 435, 440, 444, 446, 455, 456, 477, 486}
BLANK_DISTRACTOR = {45, 48, 162, 204, 243, 325}
BLANK_INDICES = BLANK_CORRECT | BLANK_DISTRACTOR


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_parquet(os.path.join(repo, CANONICAL)).set_index("index", drop=False)
    assert int(df[OPTS].isna().sum().sum()) == 0, \
        "canonical source has NaN — run dataio/convert_to_none.py first"

    sub = df.loc[sorted(BLANK_INDICES)].reset_index(drop=True).copy()
    assert len(sub) == 38, f"expected 38 blank items, got {len(sub)}"
    # self-check: every selected item really carries a "None" option in the canonical set
    has_none = (sub[OPTS] == "None").any(axis=1)
    assert has_none.all(), f"items without a 'None' option: {sub[~has_none]['index'].tolist()}"

    os.makedirs(os.path.join(repo, os.path.dirname(OUT_PARQUET)) or ".", exist_ok=True)
    sub.to_parquet(os.path.join(repo, OUT_PARQUET), index=False)

    rows = []
    for _, r in sub.iterrows():
        none_pos = [L[i] for i, c in enumerate(OPTS) if r[c] == "None"]
        rows.append({
            "index": int(r["index"]), "category": r["category"], "answer_key": r["answer"],
            "correct_is_none": r["answer"] in none_pos, "none_position": ",".join(none_pos),
            "question": r["question"],
            "A": r["option1"], "B": r["option2"], "C": r["option3"], "D": r["option4"],
        })
    man = pd.DataFrame(rows).sort_values("index")
    os.makedirs(os.path.join(repo, os.path.dirname(OUT_MANIFEST)), exist_ok=True)
    man.to_csv(os.path.join(repo, OUT_MANIFEST), index=False)

    print(f"wrote {OUT_PARQUET}: {len(sub)} items (0 NaN; blanks rendered 'None')")
    print(f"wrote {OUT_MANIFEST}")
    print(f"  correct answer is the 'None' option: {int(man['correct_is_none'].sum())}")
    print(f"  'None' is a distractor only:         {int((~man['correct_is_none']).sum())}")


if __name__ == "__main__":
    main()
