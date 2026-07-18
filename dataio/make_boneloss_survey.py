"""
Bone-loss key-bias probe (dentist audit, round 2). Continuation of the Sandeep
survey (dataio/make_dentist_survey.py), focused on ONE question: is the benchmark's
Jaw key systematically UNDER-reporting alveolar bone loss?

Motivation (RESEARCH_PLAN, open-ended analysis 2026-07-18): 88 of 106 Jaw open items
carry the same reference finding, "no apparent bone loss." Our frontier models assert
bone loss ANYWAY on 73-93% of them (gpt-5-mini 93%, gemini-3.5 73%). Either the models
over-call, or the auto-generated key under-reports. A dentist is the tie-breaker, and
the round-1 audit already certified two key errors that "denied bone loss the dentist
could see" (PAPER_DRAFT §5.8) -- so this is worth measuring properly.

DESIGN (blinded, calibration-anchored; the dentist never sees the key, the model
answers, or the bucket). We sample UNIQUE IMAGES (a dentist reads a radiograph once),
into three buckets:

  DISAGREE   key = "no bone loss"  AND  BOTH models call bone loss   -> the adjudication
             set. Dentist verdict breaks the model-vs-key tie per image.
  AGREE_NORM key = "no bone loss"  AND  both models agree "normal"    -> negative control
             (guards against a dentist who just always calls bone loss).
  KEY_POS    key ASSERTS bone loss                                     -> positive control
             (confirms the dentist detects bone loss when it is really there).

This is a LEARNING-MAXIMIZING probe, not a base-rate sample (same philosophy as round 1):
the DISAGREE bucket is deliberately adversarial, so it tests whether the key bias EXISTS
and its direction, not its exact population rate. Analysis + power in the README.

Run:   python dataio/make_boneloss_survey.py
Writes: results/dentist_audit/boneloss_manifest.csv   (our records: bucket + key + model
        stances + reference text; the hidden ground truth the blind HTML must NOT reveal)
"""
import os
import re
import random
import pandas as pd

OPEN = "data/open_ended.parquet"
GPT = "results/open/batched_gpt5mini_answers.csv"
GEM = "results/open/batched_gemini35_plain578_answers.csv"
OUT = "results/dentist_audit/boneloss_manifest.csv"
SEED = 20260718
N_DISAGREE, N_KEY_POS, N_AGREE_NORM = 12, 5, 3     # 20 bone images

# Tooth-count artifact (added 2026-07-18): the key over-counts teeth on non-standard mouths.
# Verified in-context (Fable blind read): these three are keyed ~30-31 despite visibly missing
# teeth/implants (016640 key "31 teeth" on a sparse arch). Added as dedicated count-test images
# so the dentist adjudicates the count key; ALL images also get a "how many teeth?" field.
COUNT_TEST = ["016640.jpg", "017148.jpg", "018174.jpg"]


def says_loss(t):
    """True if the text asserts alveolar bone loss (and does not negate it)."""
    t = str(t).lower()
    has = bool(re.search(r"bone loss|resorption|periodont", t))
    neg = bool(re.search(r"no (apparent )?(bone loss|resorption|significant)", t))
    return has and not neg


def gt_count(t):
    """Extract a stated total tooth count ('N teeth visualized/present/detected'), or None."""
    m = re.search(r"\b(\d{1,2})\s+teeth\s+(?:are\s+)?(?:visualized|present|detected)", str(t), re.I)
    return int(m.group(1)) if m else None


def take(rng, pool, n):
    pool = sorted(pool)
    return set(rng.sample(pool, min(n, len(pool))))


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rng = random.Random(SEED)
    op = pd.read_parquet(os.path.join(repo, OPEN))[["index", "image_name", "question", "answer"]]
    ga = pd.read_csv(os.path.join(repo, GPT))[["index", "answer"]].rename(columns={"answer": "gpt"})
    ma = pd.read_csv(os.path.join(repo, GEM))[["index", "answer"]].rename(columns={"answer": "gem"})
    m = op.merge(ga, on="index").merge(ma, on="index")

    # bone-relevant jaw questions only
    bl = m[m.question.str.contains("bone loss|bone architecture|jawbone|bone", case=False)].copy()
    bl["key_loss"] = bl.answer.map(says_loss)
    bl["gpt_loss"] = bl.gpt.map(says_loss)
    bl["gem_loss"] = bl.gem.map(says_loss)

    # collapse to one row per image; keep the reference bone-answer(s) for our records
    def joinrefs(s):
        return " || ".join(sorted(set(str(x) for x in s)))
    img = bl.groupby("image_name").agg(
        key_loss=("key_loss", "max"), gpt_loss=("gpt_loss", "max"), gem_loss=("gem_loss", "max"),
        refs=("answer", joinrefs), nq=("index", "size")).reset_index()

    # per-image ground-truth tooth count stated in any reference (the count key we audit)
    op["gtc"] = op.answer.map(gt_count)
    countmap = (op.dropna(subset=["gtc"]).groupby("image_name")["gtc"]
                .agg(lambda s: int(s.mode().iloc[0])).to_dict())

    keynone = img[~img.key_loss]
    disagree = set(keynone[keynone.gpt_loss & keynone.gem_loss].image_name)
    agree = set(keynone[~keynone.gpt_loss & ~keynone.gem_loss].image_name)
    keypos = set(img[img.key_loss].image_name)

    sel = {
        "DISAGREE": take(rng, disagree, N_DISAGREE),
        "AGREE_NORM": take(rng, agree, N_AGREE_NORM),
        "KEY_POS": take(rng, keypos, N_KEY_POS),
    }

    selected, rows = set(), []
    for bucket, names in sel.items():
        for name in sorted(names):
            r = img[img.image_name == name].iloc[0]
            selected.add(name)
            rows.append({
                "image_name": name, "bucket": bucket,
                "key_stance": "loss" if r.key_loss else "none",
                "gpt_stance": "loss" if r.gpt_loss else "none",
                "gem_stance": "loss" if r.gem_loss else "none",
                "n_bone_q": int(r.nq), "gt_count": countmap.get(name, ""),
                "reference_answers": r.refs,
            })
    # dedicated tooth-count-artifact images (compromised mouths); skip any already picked
    for name in COUNT_TEST:
        if name in selected:
            continue
        sub = bl[bl.image_name == name]
        rows.append({
            "image_name": name, "bucket": "COUNT_TEST",
            "key_stance": ("loss" if bool(sub.key_loss.max()) else "none") if len(sub) else "n/a",
            "gpt_stance": "n/a", "gem_stance": "n/a",
            "n_bone_q": int(len(sub)), "gt_count": countmap.get(name, ""),
            "reference_answers": " || ".join(sorted(set(str(x) for x in sub.answer))) if len(sub) else "",
        })
        selected.add(name)

    man = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    man.insert(0, "survey_order", range(1, len(man) + 1))
    man.insert(1, "item_id", ["bl%02d" % i for i in range(1, len(man) + 1)])

    os.makedirs(os.path.join(repo, os.path.dirname(OUT)), exist_ok=True)
    man.to_csv(os.path.join(repo, OUT), index=False)
    print(f"wrote {OUT}: {len(man)} images")
    print("  buckets:", man["bucket"].value_counts().to_dict())
    print("  gt_count present on %d/%d images" % (int((man.gt_count != '').sum()), len(man)))
    print("  (available pools: DISAGREE %d, AGREE_NORM %d, KEY_POS %d)"
          % (len(disagree), len(agree), len(keypos)))


if __name__ == "__main__":
    main()
