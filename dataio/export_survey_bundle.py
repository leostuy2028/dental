"""
Build the static viewer bundle for the dentist survey from the manifest.

Reads results/dentist_audit/survey_manifest.csv + the parquets, and writes a
self-contained bundle the GitHub-Pages viewer loads:
  survey/images/<item_id>.jpg   one radiograph per item
  survey/survey_data.json       items (question, options, reference, image path).
                                closed answer_key is embedded but the viewer never
                                displays it in the blind phase; it is used only to
                                compute disagreements for the in-session second pass.

Run:   python dataio/export_survey_bundle.py
"""
import os
import re
import json
import base64
import pandas as pd

MANIFEST = "results/dentist_audit/survey_manifest.csv"
CLOSED = "data/closed_ended.parquet"
OPEN = "data/open_ended.parquet"
OUT_DIR = "survey"


def decode_jpeg(b64):
    s = str(b64)
    s = re.sub(r"^data:image/\w+;base64,", "", s)  # strip data-URI prefix if present
    return base64.b64decode(s)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    man = pd.read_csv(os.path.join(repo, MANIFEST))
    closed = pd.read_parquet(os.path.join(repo, CLOSED)).set_index("index")
    op = pd.read_parquet(os.path.join(repo, OPEN)).set_index("index")

    img_dir = os.path.join(repo, OUT_DIR, "images")
    if os.path.isdir(img_dir):
        for f in os.listdir(img_dir):  # drop stale images from a previous selection
            if f.endswith(".jpg"):
                os.remove(os.path.join(img_dir, f))
    os.makedirs(img_dir, exist_ok=True)

    items = []
    for _, r in man.iterrows():
        src = closed if r["task_type"] == "closed" else op
        srow = src.loc[int(r["index"])]  # authoritative text from the parquet (never the
        # CSV: pd.read_csv coerces the string "None" to NaN -> str() renders "nan")
        with open(os.path.join(img_dir, f"{r['item_id']}.jpg"), "wb") as f:
            f.write(decode_jpeg(srow["image"]))

        item = {
            "item_id": r["item_id"],
            "order": int(r["survey_order"]),
            "task_type": r["task_type"],
            "question": str(srow["question"]),
            "image": f"images/{r['item_id']}.jpg",
        }
        if r["task_type"] == "closed":
            item["options"] = {L: str(srow[f"option{n}"]) for L, n in zip("ABCD", (1, 2, 3, 4))}
            item["_key"] = str(srow["answer"])  # not shown in blind phase; used for 2nd pass
        else:
            item["reference"] = str(srow["answer"])
        items.append(item)

    items.sort(key=lambda x: x["order"])
    with open(os.path.join(repo, OUT_DIR, "survey_data.json"), "w", encoding="utf-8") as f:
        json.dump({"items": items,
                   "n_closed": int((man["task_type"] == "closed").sum()),
                   "n_open": int((man["task_type"] == "open").sum())}, f, ensure_ascii=False)

    print(f"wrote {OUT_DIR}/survey_data.json ({len(items)} items) + {len(items)} images in {OUT_DIR}/images/")


if __name__ == "__main__":
    main()
