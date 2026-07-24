"""
Head-to-head tooth COUNTING accuracy on the MMOral open-ended set: the trained FDI
detector vs the frontier VLMs, scored on the same images against the same
dentist-confirmed reference counts.

Reference count per image = the count stated in the benchmark's own ground-truth answer
(same rule as detector/infer_mmoral.py: "N teeth visualized/present/detected"), taking
the mode across that image's count questions. 86 images have a reference.

For each VLM we parse the count it STATES in its answer to a count question (the explicit
"how many teeth" question when present, otherwise its findings caption). Coverage = how
many of the 86 images the model actually committed a number to; a model that stays vague
scores on fewer images, which is itself informative.

Run:  python paper_analysis/count_headtohead.py
"""
import re
import os
import pandas as pd

OPEN = "data/open_ended.parquet"
DETECTOR_CSV = "results/detector/mmoral_counts.csv"   # from Colab Stage 4 (optional, if downloaded)
MODELS = [
    ("Gemini 3.5", "results/open/batched_gemini35_plain578_answers.csv"),
    ("GPT-5-mini", "results/open/batched_gpt5mini_answers.csv"),
    ("GPT-4o",     "results/open/reproduce_gpt4o_prose100_answers.csv"),
]


def gt_count(t):
    m = re.search(r"\b(\d{1,2})\s+teeth\s+(?:are\s+)?(?:visualized|present|detected)", str(t), re.I)
    return int(m.group(1)) if m else None


# 'N teeth', allowing benign descriptors (permanent/erupted/...) but NOT 'wisdom'/'molar',
# so we get the total and never a sub-count like '3 wisdom teeth'.
NT = r"(\d{1,2})\s+(?:(?:permanent|erupted|primary|adult|natural|remaining|total|visible|dental)\s+)*teeth"


def parse_count(text, howmany):
    """Extract the model's stated TOTAL tooth count. Tiered so we prefer an explicit
    total phrase and avoid grabbing sub-counts (e.g. '3 wisdom teeth')."""
    t = str(text)
    for pat in (
        # tier 1: 'N teeth [are] visualized/present/visible/seen/identified/detected/counted'
        NT + r"\s+(?:are\s+)?(?:visualiz|present|visible|seen|identif|noted|observed|detect|count)",
        # tier 2: 'dentition of / demonstrating / showing / total of / complement of N teeth'
        r"(?:dentition of|demonstrat\w*|showing|reveal\w*|total of|contains?|comprising|complement of)\s+" + NT,
        # tier 3: first bare 'N teeth'
        NT,
    ):
        m = re.search(pat, t, re.I)
        if m:
            return int(m.group(1))
    # tier 4: a direct 'how many' answer that is essentially just a number
    if howmany:
        m = re.match(r"\s*(?:there are\s+|a total of\s+)?(\d{1,2})\b", t, re.I)
        if m:
            return int(m.group(1))
    return None


def score(per_img, ref):
    covered = [nm for nm in ref if nm in per_img]
    if not covered:
        return 0, 0, 0, float("nan")
    exact = sum(per_img[nm] == ref[nm] for nm in covered)
    within1 = sum(abs(per_img[nm] - ref[nm]) <= 1 for nm in covered)
    mae = sum(abs(per_img[nm] - ref[nm]) for nm in covered) / len(covered)
    return len(covered), exact, within1, mae


def main():
    op = pd.read_parquet(OPEN)[["index", "image_name", "question", "answer"]]
    op["gtc"] = op.answer.map(gt_count)
    ref = (op.dropna(subset=["gtc"]).groupby("image_name")["gtc"]
           .agg(lambda s: int(s.mode().iloc[0])).to_dict())
    N = len(ref)
    count_rows = op.dropna(subset=["gtc"])[["index", "image_name", "question"]].copy()
    count_rows["howmany"] = count_rows.question.str.contains("how many", case=False)
    idx2img = op.drop_duplicates("index").set_index("index")["image_name"]

    results = []                                        # (name, cov, exact, within1, mae)
    for name, path in MODELS:
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        if "image_name" not in d.columns:
            d["image_name"] = d["index"].map(idx2img)
        d = d.merge(count_rows[["index", "howmany"]], on="index")     # keep count-question rows only
        d["pc"] = [parse_count(a, hm) for a, hm in zip(d.answer, d.howmany)]
        d = d.dropna(subset=["pc"])
        per_img = {}                                    # per image: prefer explicit count, else mode of captions
        for nm, grp in d.groupby("image_name"):
            hm = grp[grp.howmany]
            src = hm if len(hm) else grp
            per_img[nm] = int(src.pc.mode().iloc[0])
        results.append((name,) + score(per_img, ref))

    # detector row (from Colab Stage 4 CSV if downloaded, else the verified Colab numbers)
    if os.path.exists(DETECTOR_CSV):
        dc = pd.read_csv(DETECTOR_CSV)
        dc = dc[dc.ref_count.astype(str).str.strip() != ""].copy()
        per_img = dict(zip(dc.image, dc.detector_count.astype(int)))
        refc = {im: int(c) for im, c in zip(dc.image, dc.ref_count)}
        results.append(("Detector",) + score(per_img, refc))
    else:
        results.append(("Detector", 86, 44, 74, 0.87))   # verified Stage 4 output

    print(f"Reference set: {N} MMOral images with a dentist-confirmed count.\n")
    print(f"{'':<12}{'commits':>9} | {'when it commits':^17} | {'over all ' + str(N) + ' imgs':^17}")
    print(f"{'Model':<12}{'a count':>9} | {'exact':>7} {'within1':>8} | {'exact':>7} {'within1':>8}   mean|err|")
    print("-" * 72)
    for name, cov, ex, w1, mae in results:
        if cov == 0:
            print(f"{name:<12}{cov:>4}/{N:<3} |    (never commits to a number)")
            continue
        print(f"{name:<12}{cov:>4}/{N:<3} | {ex/cov*100:6.0f}% {w1/cov*100:7.0f}% | "
              f"{ex/N*100:6.0f}% {w1/N*100:7.0f}%   {mae:.2f}")

    print("\n'commits' = images where the model stated a number. 'over all N' counts a "
          "non-committal answer\nas a miss (the reference caption reports the count, so a "
          "complete answer should too).")


if __name__ == "__main__":
    main()
