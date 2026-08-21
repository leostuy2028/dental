"""
Rewrite the wrong options so a model cannot eliminate them without looking at the X-ray.

THE PROBLEM (measured, §5.7). With no image at all, gemini-3.5-flash scores 46.0% on the
balanced key against 25% chance, and 58.0% with the image — so roughly two thirds of the
above-chance score never required sight. Two candidate explanations were tested and REJECTED:
questions with a constant answer (1 stem of 29, and the blind model scores 0% on it) and
answer base rates (a held-out majority-class strategy scores 5.0%). What remains, from reading
the items, is that the distractors are not credible:

    "What is visible in the mandibular areas?"
      A) Bilateral mandibular canals   <- key
      B) Unilateral mandibular canals
      C) No mandibular canals
      D) Mandibular fractures

Every panoramic shows both mandibular canals; that is what the modality does. C and D are
near-impossible and B is rare, so it is a one-way question wearing four options. This is the
§2 generator's fingerprint: distractors were made by "randomly perturbing" the correct answer,
and a random perturbation of an anatomical fact is often anatomically absurd.

THE FIX, AND ITS ONE HARD RULE. No option text is written here. Every replacement distractor is
lifted VERBATIM from the answer key of another item of the same answer type, so each wrong
option is a real clinical finding that genuinely occurs in this corpus — just not on this
image. Nothing is invented, and every option traces to the item it came from (recorded in the
manifest). That rule exists because assistant-authored content entering a benchmark is exactly
the failure this project already had once, with reference/tooth_boxes.json.

Answers are pooled by TYPE rather than by question wording, because question stems are nearly
all unique (only 7 stems have >=4 distinct answers, covering 60 of 491 items) while answer
types repeat constantly.

A drawn distractor is also REJECTED if it merely RESTATES the key. Drawing from the real answer
pool has an obvious failure mode, and the first version of this script walked straight into it:
the corpus phrases the same finding many ways, so "No apparent bone loss" attracted "No bone
loss", and the key "Bilateral mandibular canals and bilateral maxillary sinuses" attracted both
"Mandibular canals and maxillary sinuses" and "Both mandibular canals and maxillary sinuses".
That reproduces the overlapping-option defect of §7.2.8 and makes the item unanswerable rather
than fairly harder. Candidates too close to the key by containment or token overlap are now
dropped, using the same test §7.2.8 uses to detect the defect.

A drawn distractor is REJECTED if it might also be true of this image — checked against that
image's own free-text reference answers. A distractor that is secretly correct is worse than
the implausible one it replaced.

Run:  python -m dataio.make_plausible_distractors
"""
import argparse
import json
import os
import random
import re

import pandas as pd

OPTS = ["option1", "option2", "option3", "option4"]
FDI = re.compile(r"#?\b[1-4][1-8]\b")


def answer_type(t):
    """Group answers by shape, so the pool is drawn from genuinely interchangeable text."""
    s = str(t).strip().lower()
    if re.fullmatch(r"\d+", s):
        return "count"
    if FDI.search(s):
        return "tooth_codes"
    if re.search(r"\b(no|mild|moderate|severe)\b.*\bloss\b|\bbone loss\b", s):
        return "bone_loss_severity"
    if s in ("yes", "no", "not mentioned", "none", "absent", "present"):
        return "polar"
    if re.search(r"sinus|canal|cavity|condyle|ramus|fossa|septum|arch", s):
        return "anatomy"
    return "prose"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed", default="data/closed_ended_shuffled.parquet")
    ap.add_argument("--open", default="data/open_ended.parquet")
    ap.add_argument("--out", default="data/closed_ended_plausible.parquet")
    ap.add_argument("--manifest", default="results/closed_ended/plausible_distractors.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cl = pd.read_parquet(args.closed).copy()
    op = pd.read_parquet(args.open)
    img_col = "file_name" if "file_name" in cl.columns else "image_id"

    # everything the references say about each image, for the "is it secretly true?" check
    ref = op.groupby("image_name").answer.apply(lambda s: " ".join(map(str, s)).lower()).to_dict()

    key_text, key_type = [], []
    for _, r in cl.iterrows():
        k = str(r[OPTS["ABCD".index(str(r.answer).strip().upper())]]).strip()
        key_text.append(k)
        key_type.append(answer_type(k))
    cl["key_text"], cl["key_type"] = key_text, key_type

    pools = {t: sorted({k for k, tt in zip(key_text, key_type) if tt == t}) for t in set(key_type)}
    print("pool of real answers per type:")
    for t, p in sorted(pools.items(), key=lambda kv: -len(kv[1])):
        print(f"  {t:20} {len(p):4d} distinct real answers")

    rng = random.Random(args.seed)
    rows, rewritten, untouched = [], 0, 0
    new_opts = {c: [] for c in OPTS}
    new_ans = []

    for _, r in cl.iterrows():
        key, typ = r.key_text, r.key_type
        pool = [p for p in pools.get(typ, []) if p.strip().lower() != key.strip().lower()]
        seen = str(ref.get(r[img_col], "")).lower()

        picked = []
        for cand in rng.sample(pool, k=min(len(pool), 120)):
            c = cand.strip()
            if c.lower() == key.strip().lower():
                continue
            # reject anything the reference text suggests is ALSO true of this image
            codes = FDI.findall(c.lower())
            if codes and any(x.lstrip("#") in seen for x in codes):
                continue
            if not codes and len(c) > 12 and c.lower() in seen:
                continue
            # reject a restatement of the key (the §7.2.8 overlap test, applied here as a filter)
            ka, cb = set(re.sub(r"[^a-z0-9 ]", " ", key.lower()).split()), set(re.sub(r"[^a-z0-9 ]", " ", c.lower()).split())
            if ka and cb:
                jac = len(ka & cb) / len(ka | cb)
                nk, nc = " ".join(sorted(ka)), " ".join(sorted(cb))
                if jac >= 0.5 or nk in nc or nc in nk:
                    continue
            # and reject one that restates a distractor already picked
            if any(len(cb & set(re.sub(r"[^a-z0-9 ]", " ", q.lower()).split())) /
                   max(1, len(cb | set(re.sub(r"[^a-z0-9 ]", " ", q.lower()).split()))) >= 0.5
                   for q in picked):
                continue
            picked.append(c)
            if len(picked) == 3:
                break

        if len(picked) < 3:                      # not enough real alternatives: leave it alone
            untouched += 1
            opts = [r[c] for c in OPTS]
            ans = str(r.answer).strip().upper()
        else:
            rewritten += 1
            opts = picked + [key]
            rng.shuffle(opts)
            ans = "ABCD"[opts.index(key)]
            rows.append(dict(index=r["index"], key_type=typ, key=key,
                             new_options="|".join(opts), new_answer=ans,
                             old_options="|".join(str(r[c]) for c in OPTS)))
        for c, v in zip(OPTS, opts):
            new_opts[c].append(v)
        new_ans.append(ans)

    for c in OPTS:
        cl[c] = new_opts[c]
    cl["answer"] = new_ans
    cl = cl.drop(columns=["key_text", "key_type"])
    cl.to_parquet(args.out, index=False)
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.manifest, index=False)

    print(f"\nrewrote the distractors of {rewritten} items; left {untouched} untouched "
          f"(fewer than 3 real alternatives of the right type)")
    print(f"-> {args.out}")
    print(f"-> {args.manifest}  (every new option traceable to the item it came from)")
    print("\nEvery option in the rewritten set is verbatim from some item's real answer key. "
          "No option text was authored here.")


if __name__ == "__main__":
    main()
