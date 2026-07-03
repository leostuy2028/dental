"""
E-open-1: coordinate-bias isolation run.

Grades each item's two format-only prediction variants (prose, coords) under both
rubrics (original, rephrased) with each judge, and writes one row per
(index, judge, rubric, variant) to a resumable CSV.

Primary metric (see analyze.py): coordinate bonus = score(coords) - score(prose),
paired per item. Expect > 0 under the original rubric (F-open-1) and ~0 under the
rephrased rubric, concentrated on coord_ref items.

Examples:
  # quick pilot: 20 coord + 20 prose, gemini+claude
  python eval_open/run_isolation.py --judges gemini,claude --coord-sample 20 --prose-sample 20
  # full run, all three judges (needs OPENAI_API_KEY)
  python eval_open/run_isolation.py --judges gpt-4o,gemini,claude
"""
import os
import argparse
import pandas as pd
from pathlib import Path

from eval_open.judges import grade

PRED = "eval_open/predictions.parquet"
RUBRICS = ["original", "rephrased"]
VARIANTS = {"prose": "pred_prose", "coords": "pred_coords"}
KEY = ["index", "judge", "rubric", "variant"]


def load_done(out_path):
    if not Path(out_path).exists():
        return set(), []
    df = pd.read_csv(out_path)
    done = set(tuple(r) for r in df[KEY].itertuples(index=False, name=None))
    return done, df.to_dict("records")


def sample_items(preds, coord_n, prose_n, seed):
    coord = preds[preds.ref_type == "coord_ref"]
    prose = preds[preds.ref_type == "prose_ref"]
    if coord_n is not None:
        coord = coord.sample(min(coord_n, len(coord)), random_state=seed)
    if prose_n is not None:
        prose = prose.sample(min(prose_n, len(prose)), random_state=seed)
    return pd.concat([coord, prose]).sort_values("index")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="gemini,claude",
                    help="comma list of gpt-4o,gemini,claude")
    ap.add_argument("--model", default=None,
                    help="override the judge's default model id (e.g. gemini-3.5-flash); "
                         "applies to every judge in --judges")
    ap.add_argument("--coord-sample", type=int, default=None, help="cap coord_ref items")
    ap.add_argument("--prose-sample", type=int, default=None, help="cap prose_ref items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.0, help="per-call delay (rate limiting)")
    ap.add_argument("--out", default="results/open/isolation.csv")
    args = ap.parse_args()

    from eval_open.rubrics import build_grading_prompt

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    if "gpt-4o" in judges and "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("gpt-4o requested but OPENAI_API_KEY is not set in .env")

    preds = pd.read_parquet(PRED)
    items = sample_items(preds, args.coord_sample, args.prose_sample, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done, records = load_done(args.out)

    total = len(items) * len(judges) * len(RUBRICS) * len(VARIANTS)
    print(f"{len(items)} items x {len(judges)} judges x {len(RUBRICS)} rubrics x "
          f"{len(VARIANTS)} variants = {total} cells ({len(done)} already done)")

    n = 0
    for _, it in items.iterrows():
        for judge in judges:
            for rubric in RUBRICS:
                for variant, col in VARIANTS.items():
                    n += 1
                    k = (int(it["index"]), judge, rubric, variant)
                    if k in done:
                        continue
                    prompt = build_grading_prompt(it["question"], it["answer"],
                                                  it[col], rubric)
                    score, raw = grade(prompt, judge=judge, model=args.model,
                                       delay=args.delay)
                    records.append({
                        "index": int(it["index"]), "ref_type": it["ref_type"],
                        "category": it["category"], "judge": judge,
                        "model": args.model or "default", "rubric": rubric,
                        "variant": variant, "score": score,
                        "raw": str(raw).replace("\n", " ")[:120],
                    })
                    if len(records) % 20 == 0:
                        pd.DataFrame(records).to_csv(args.out, index=False)
                        print(f"  [{n}/{total}] idx={k[0]} {judge}/{rubric}/{variant} -> {score}")
    pd.DataFrame(records).to_csv(args.out, index=False)
    print(f"done -> {args.out} ({len(records)} rows)")


if __name__ == "__main__":
    main()
