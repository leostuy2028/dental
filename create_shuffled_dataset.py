import random
import pandas as pd

INPUT_PATH = "data/closed_ended.parquet"
OUTPUT_PATH = "data/closed_ended_shuffled.parquet"

LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def shuffle_row(row, seed):
    options = [row["option1"], row["option2"], row["option3"], row["option4"]]
    correct_text = options[LETTER_TO_IDX[row["answer"]]]

    rng = random.Random(seed)
    rng.shuffle(options)

    new_answer = IDX_TO_LETTER[options.index(correct_text)]

    return {
        "option1": options[0],
        "option2": options[1],
        "option3": options[2],
        "option4": options[3],
        "answer": new_answer,
    }


def create():
    df = pd.read_parquet(INPUT_PATH)

    shuffled_rows = []
    for _, row in df.iterrows():
        updated = shuffle_row(row, seed=int(row["index"]))
        shuffled_rows.append(updated)

    shuffled_df = df.copy()
    for col in ["option1", "option2", "option3", "option4", "answer"]:
        shuffled_df[col] = [r[col] for r in shuffled_rows]

    # verify: correct option text must match between original and shuffled
    for i, (orig, shuf) in enumerate(zip(df.itertuples(), shuffled_df.itertuples())):
        orig_text = getattr(orig, f"option{LETTER_TO_IDX[orig.answer] + 1}")
        shuf_text = getattr(shuf, f"option{LETTER_TO_IDX[shuf.answer] + 1}")
        assert orig_text == shuf_text, f"Answer mismatch at row {i}: '{orig_text}' != '{shuf_text}'"

    shuffled_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(shuffled_df)}")
    print()
    print("Original answer distribution:")
    print(df["answer"].value_counts().sort_index().to_string())
    print()
    print("Shuffled answer distribution:")
    print(shuffled_df["answer"].value_counts().sort_index().to_string())
    print()
    print("Sample row comparison:")
    print(f"  Original : {df.iloc[0]['option1']}(A) {df.iloc[0]['option2']}(B) {df.iloc[0]['option3']}(C) {df.iloc[0]['option4']}(D) -> answer={df.iloc[0]['answer']}")
    print(f"  Shuffled : {shuffled_df.iloc[0]['option1']}(A) {shuffled_df.iloc[0]['option2']}(B) {shuffled_df.iloc[0]['option3']}(C) {shuffled_df.iloc[0]['option4']}(D) -> answer={shuffled_df.iloc[0]['answer']}")


if __name__ == "__main__":
    create()
