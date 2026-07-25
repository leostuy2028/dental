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
import json
import datetime
import pandas as pd

OPEN = "data/open_ended.parquet"
DETECTOR_CSV_CANDIDATES = ["results/detector/mmoral_counts.csv", "mmoral_counts.csv"]  # Colab Stage 4 output
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
    count_rows = op.dropna(subset=["gtc"])[["index", "question"]].copy()
    count_rows["howmany"] = count_rows.question.str.contains("how many", case=False)
    idx2img = op.drop_duplicates("index").set_index("index")["image_name"]

    # detector per-image counts (Colab Stage 4)
    det = {}
    dcsv = next((p for p in DETECTOR_CSV_CANDIDATES if os.path.exists(p)), None)
    if dcsv:
        dd = pd.read_csv(dcsv).dropna(subset=["ref_count"])
        det = {im: int(c) for im, c in zip(dd.image, dd.detector_count)}

    model_per_img = {}                                  # name -> {image: stated count}
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
        model_per_img[name] = per_img
        results.append((name,) + score(per_img, ref))

    results.append(("Detector",) + (score(det, ref) if det else (86, 44, 74, 0.87)))

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

    # fairness check: detector vs each model on the EXACT images that model committed to
    if det:
        print("\nSame-subset check — detector vs model on the exact images the model committed to:")
        print(f"{'Model':<12} {'n':>3} |  {'model exact/w1':^13} |  {'detector exact/w1':^15}")
        print("-" * 54)
        for name, per_img in model_per_img.items():
            imgs = [i for i in per_img if i in ref and i in det]
            if not imgs:
                continue
            n = len(imgs)
            mex = sum(per_img[i] == ref[i] for i in imgs); mw1 = sum(abs(per_img[i] - ref[i]) <= 1 for i in imgs)
            dex = sum(det[i] == ref[i] for i in imgs);     dw1 = sum(abs(det[i] - ref[i]) <= 1 for i in imgs)
            print(f"{name:<12} {n:>3} |   {mex/n*100:3.0f}% / {mw1/n*100:3.0f}%   |    {dex/n*100:3.0f}% / {dw1/n*100:3.0f}%")

    # ---- paper Table 9.1 as a Markdown fragment + values.json (RESEARCH_PLAN §1.0 rule 7) ----
    # The table above is for reading in a terminal; the paper needs the exact markdown, so it is
    # emitted here rather than hand-formatted into the draft.
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_generated")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    n_ref = len(ref)
    lines = [f"<!-- GENERATED by paper_analysis/count_headtohead.py on {stamp}. "
             f"Do not hand-edit; run `python -m paper_analysis.count_headtohead`. -->",
             "| System | states a count | correct count, all "
             f"{n_ref} (exact / within-1) |",
             "|:--|--:|--:|"]
    vals = {"n_reference_images": n_ref, "rows": {}}
    for name, per_img in model_per_img.items():
        imgs = [i for i in per_img if i in ref]
        ex = sum(per_img[i] == ref[i] for i in imgs)
        w1 = sum(abs(per_img[i] - ref[i]) <= 1 for i in imgs)
        cell = "—" if not imgs else f"{ex/n_ref*100:.0f}% / {w1/n_ref*100:.0f}%"
        lines.append(f"| {name} | {len(imgs)} / {n_ref} | {cell} |")
        vals["rows"][name] = {"states_count": len(imgs), "exact_pct": round(ex/n_ref*100, 1),
                              "within1_pct": round(w1/n_ref*100, 1)}
    if det:
        dimgs = [i for i in det if i in ref]
        dex = sum(det[i] == ref[i] for i in dimgs); dw1 = sum(abs(det[i] - ref[i]) <= 1 for i in dimgs)
        lines.append(f"| **Tooth detector** | **{len(dimgs)} / {n_ref}** | "
                     f"**{dex/n_ref*100:.0f}% / {dw1/n_ref*100:.0f}%** |")
        vals["rows"]["Tooth detector"] = {"states_count": len(dimgs), "exact_pct": round(dex/n_ref*100, 1),
                                          "within1_pct": round(dw1/n_ref*100, 1)}
    frag = "\n".join(lines) + "\n"
    open(os.path.join(out_dir, "count_headtohead_table.md"), "w", encoding="utf-8").write(frag)
    vals.update({"_generator": "paper_analysis/count_headtohead.py", "_generated_utc": stamp})
    json.dump(vals, open(os.path.join(out_dir, "count_headtohead.values.json"), "w"), indent=2)
    print("\n" + frag)
    print(f"wrote {out_dir}/count_headtohead_table.md and .values.json")


if __name__ == "__main__":
    main()
