"""
GENERATOR for: PAPER_DRAFT.md  §7  "Toward a more trustworthy benchmark"

Classifies every open-ended question by how CONCRETE it is, then reports each model's
free-text score per class. The point: ~1 in 5 free-text questions is a vague whole-image
"caption / summarize" prompt with no determinate answer, and models score HIGHEST on
exactly those — so a chunk of the free-text score rewards fluent essays over precise
reading. The concrete questions (a specific tooth, a count, which-teeth, detect a named
structure) are the checkable core worth evaluating and improving on.

Classes:
  CONCRETE : specific tooth #N; how many; which tooth/teeth; detect/identify a named
             structure; list the restorations. A determinate, checkable answer.
  BROAD    : a bounded topic with no specific target ("general condition of the teeth",
             "status of the wisdom teeth", "what structures are visible", recommendations).
  VAGUE    : open composition over the whole image ("caption / summarize / describe the
             findings"). No determinate answer.

Run:   python paper_analysis/question_concreteness.py
Reads: data/open_ended.parquet,
       results/open/batched_gpt5mini_scores.csv,
       results/open/batched_gemini35_plain578_scores.csv
Writes: paper_analysis/_generated/question_concreteness_table.md
        paper_analysis/_generated/question_concreteness.values.json
"""
import os
import re
import json
import math
import datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "_generated")


def classify(q):
    ql = str(q).lower()
    if re.search(r"#?\b[1-4][1-8]\b", str(q)):        return "Concrete"   # a specific tooth
    if re.search(r"how many", ql):                    return "Concrete"   # a count
    if re.search(r"which (tooth|teeth)", ql):         return "Concrete"   # which teeth have X
    if re.search(r"output the position|^detect |please detect|identify (the )?(teeth|tooth|"
                 r"mandibular|maxillary|historical|areas)|accurately identify", ql): return "Concrete"
    if re.search(r"which (mandibular canal|areas)", ql):                  return "Concrete"
    if re.search(r"historical (intervention|treatment)", ql):             return "Concrete"
    if re.search(r"list the teeth|number of teeth|which wisdom tooth|which periapical|"
                 r"identify the periapical|are the mandibular canals visible", ql):  return "Concrete"
    if re.search(r"caption|summar|describe|description|what (is|are) the findings|"
                 r"findings (can be )?(observed|presented)|findings of (this|the) panoramic|"
                 r"most significant|anomalies", ql):   return "Vague"
    if re.search(r"general condition|status of|wisdom teeth|structures (are )?(visible|observed)|"
                 r"visible structures|bone|jaw|pathological finding|priority concern|preventive|"
                 r"recommend|clinical|condition of|observed in|signs (were|are) present", ql):
        return "Broad"
    return "Concrete"   # remaining are list/detect-type -> concrete


def acc_ci(s):
    n = len(s)
    mu = s.mean() * 100
    se = (s.std(ddof=1) / math.sqrt(n) * 100) if n > 1 else 0.0
    return round(mu, 1), round(mu - 1.96 * se, 1), round(mu + 1.96 * se, 1), n


def main():
    op = pd.read_parquet(os.path.join(REPO, "data/open_ended.parquet"))[["index", "question"]]
    gpt = pd.read_csv(os.path.join(REPO, "results/open/batched_gpt5mini_scores.csv")).rename(columns={"score": "gpt"})
    gem = pd.read_csv(os.path.join(REPO, "results/open/batched_gemini35_plain578_scores.csv")).rename(columns={"score": "gem"})
    m = op.merge(gpt, on="index").merge(gem, on="index")
    m["bucket"] = m.question.map(classify)

    order = ["Concrete", "Broad", "Vague"]
    vals = {"total": len(m), "buckets": {}}
    for b in order:
        sub = m[m.bucket == b]
        gpt_m, gpt_lo, gpt_hi, n = acc_ci(sub["gpt"])
        gem_m, gem_lo, gem_hi, _ = acc_ci(sub["gem"])
        vals["buckets"][b] = {"n": n, "share_pct": round(100 * n / len(m), 1),
                              "gpt5mini": [gpt_m, gpt_lo, gpt_hi],
                              "gemini35": [gem_m, gem_lo, gem_hi]}
    all_gpt = acc_ci(m["gpt"]); all_gem = acc_ci(m["gem"])
    vals["all"] = {"n": len(m), "gpt5mini": list(all_gpt[:3]), "gemini35": list(all_gem[:3])}

    # templating fingerprints in the REFERENCE answers (§7 "same flaws in the training data"):
    # stock phrases that recur verbatim across images betray template-minting, not per-image reading.
    refs = pd.read_parquet(os.path.join(REPO, "data/open_ended.parquet"))["answer"].astype(str)
    ex = pd.read_parquet(os.path.join(REPO, "data/open_ended.parquet"))
    ex = ex[ex["answer"].astype(str).str.contains(
        "teeth visualized with clear anatomical definition", case=False)]["image_name"].head(3).tolist()
    vals["templating"] = {
        "n_refs": int(len(refs)),
        "clear_anatomical_definition": int(refs.str.contains("clear anatomical definition", case=False).sum()),
        "no_apparent_bone_loss": int(refs.str.contains("no apparent bone loss", case=False).sum()),
        "coordinate_box_refs": int(refs.str.contains("box_2d|point_2d", case=False).sum()),
        "verbatim_count_template_example_images": ex,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    prov = (f"<!-- GENERATED by paper_analysis/question_concreteness.py on {stamp}. "
            f"Do not hand-edit; run `python paper_analysis/question_concreteness.py`. -->")
    desc = {"Concrete": "specific tooth / count / which-teeth / detect",
            "Broad": "a topic, but no specific target",
            "Vague": "caption / summarize the whole image"}
    lines = [prov,
             "| Question type | n | share | gpt-5-mini | gemini-3.5 |",
             "|:--|--:|--:|--:|--:|"]
    for b in order:
        v = vals["buckets"][b]
        lines.append(f"| **{b}** ({desc[b]}) | {v['n']} | {v['share_pct']}% | "
                     f"{v['gpt5mini'][0]}% | {v['gemini35'][0]}% |")
    v = vals["all"]
    lines.append(f"| All | {v['n']} | 100% | {v['gpt5mini'][0]}% | {v['gemini35'][0]}% |")
    table = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "question_concreteness_table.md"), "w", encoding="utf-8") as f:
        f.write(table)
    vals["_generator"] = "paper_analysis/question_concreteness.py"
    vals["_source_csv"] = ["results/open/batched_gpt5mini_scores.csv",
                           "results/open/batched_gemini35_plain578_scores.csv"]
    vals["_generated_utc"] = stamp
    with open(os.path.join(OUT_DIR, "question_concreteness.values.json"), "w", encoding="utf-8") as f:
        json.dump(vals, f, indent=2)
    print(table)
    t = vals["templating"]
    print(f"templating fingerprints in {t['n_refs']} refs: "
          f"'clear anatomical definition' {t['clear_anatomical_definition']}, "
          f"'no apparent bone loss' {t['no_apparent_bone_loss']}, "
          f"coordinate-box refs {t['coordinate_box_refs']}; "
          f"verbatim count-template example images {t['verbatim_count_template_example_images']}")
    print(f"wrote {OUT_DIR}/question_concreteness_table.md and .values.json")


if __name__ == "__main__":
    main()
