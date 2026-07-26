"""
Stage 3b — choose the detector's two inference settings, and measure what they buy.

The detector shipped with ultralytics' defaults: imgsz=1024, conf=0.25. Neither was ever
chosen. This sweeps both against the benchmark's own dentist-confirmed tooth counts and
reports three things:

  1. the grid — exact-count rate for every (imgsz, conf) pair;
  2. a held-out check — pick the setting on half the images, score it on the other half,
     repeated over many random splits, so the reported gain is not just the grid maximum;
  3. the payoff — the detector answering the benchmark's own 28 pure-count multiple-choice
     questions, against what the frontier models score on the same 28 (committed CSVs, no
     API calls). The setting used there is chosen ONLY on images carrying no such question.

Why one inference pass per resolution is enough for the whole conf axis: NMS sorts boxes by
confidence and lets higher-confidence boxes suppress lower ones. A box below threshold t can
never suppress a box above t, so predict(conf=0.01) filtered to conf>=t gives the same
surviving set as predict(conf=t). The conf sweep is therefore free.

Ground truth is the count stated in the benchmark's own reference answers ("N teeth are
visualized"), the same source detector/infer_mmoral.py scores against.

Run (CPU is fine; the six inference passes take ~20 min):
  python detector/tune_inference.py --weights best.pt
Re-runs reuse results/detector/inference_raw_confs.json unless --refresh is passed.
"""
import argparse
import base64
import glob
import io
import json
import os
import random
import re
import statistics as st

import pandas as pd

RES = [768, 1024, 1280, 1536, 1792, 2048]
CONF = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
SHIPPED = (1024, 0.25)
PURE = re.compile(r"^how many (natural |permanent )?teeth (are|were|is|was)?\s*"
                  r"(visualized|visible|present|detected|seen)[^,]*\??$", re.I)
OPTS = ["option1", "option2", "option3", "option4"]


def gt_count(text):
    m = re.search(r"\b(\d{1,2})\s+teeth\s+(?:are\s+)?(?:visualized|present|detected)", str(text), re.I)
    return int(m.group(1)) if m else None


def reference_counts(open_parquet):
    op = pd.read_parquet(open_parquet)
    op["gtc"] = op.answer.map(gt_count)
    ref = (op.dropna(subset=["gtc"]).groupby("image_name")["gtc"]
           .agg(lambda s: int(s.mode().iloc[0])).to_dict())
    return op, ref


def load_raw(args, op):
    """(imgsz, image) -> descending list of surviving box confidences."""
    if os.path.exists(args.raw) and not args.refresh:
        with open(args.raw) as f:
            return {(int(k.split("|", 1)[0]), k.split("|", 1)[1]): v for k, v in json.load(f).items()}

    from PIL import Image
    from ultralytics import YOLO

    images = op.drop_duplicates("image_name").set_index("image_name")["image"]
    decoded = {}
    for name, b64 in images.items():
        blob = base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", str(b64)))
        decoded[name] = Image.open(io.BytesIO(blob)).convert("RGB")

    model = YOLO(args.weights)
    raw = {}
    for imgsz in RES:
        for name, im in decoded.items():
            r = model.predict(im, imgsz=imgsz, conf=0.01, agnostic_nms=True, verbose=False)[0]
            raw[(imgsz, name)] = sorted((float(c) for c in r.boxes.conf.tolist()), reverse=True)
        print(f"  imgsz={imgsz} done", flush=True)

    os.makedirs(os.path.dirname(args.raw) or ".", exist_ok=True)
    with open(args.raw, "w") as f:
        json.dump({f"{i}|{n}": v for (i, n), v in raw.items()}, f)
    return raw


def count_at(raw, setting, name):
    return sum(1 for c in raw[(setting[0], name)] if c >= setting[1])


def exact_rate(raw, ref, setting, subset):
    hit = sum(1 for n in subset if count_at(raw, setting, n) == ref[n])
    return 100 * hit / len(subset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--open", default="data/open_ended.parquet")
    ap.add_argument("--closed", default="data/closed_ended.parquet")
    ap.add_argument("--raw", default="results/detector/inference_raw_confs.json")
    ap.add_argument("--out", default="results/detector/inference_sweep.csv")
    ap.add_argument("--splits", type=int, default=400)
    ap.add_argument("--refresh", action="store_true", help="re-run inference, ignore the cache")
    args = ap.parse_args()

    op, ref = reference_counts(args.open)
    raw = load_raw(args, op)
    names = sorted(ref)
    print(f"\n{len(names)} images carry a dentist-confirmed reference count\n")

    # ---- 1. the grid -------------------------------------------------------------
    rows = []
    for imgsz in RES:
        for conf in CONF:
            errs = [abs(count_at(raw, (imgsz, conf), n) - ref[n]) for n in names]
            rows.append(dict(imgsz=imgsz, conf=conf, n=len(names),
                             exact=100 * sum(e == 0 for e in errs) / len(errs),
                             within1=100 * sum(e <= 1 for e in errs) / len(errs),
                             mae=sum(errs) / len(errs)))
    grid = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    grid.to_csv(args.out, index=False)

    print("exact-count %  (rows = imgsz, cols = conf)")
    print(grid.pivot(index="imgsz", columns="conf", values="exact").round(0).to_string())
    best = max(((i, c) for i in RES for c in CONF), key=lambda s: exact_rate(raw, ref, s, names))
    for label, s in (("shipped ", SHIPPED), ("best    ", best)):
        r = grid[(grid.imgsz == s[0]) & (grid.conf == s[1])].iloc[0]
        print(f"\n  {label} imgsz={s[0]}, conf={s[1]}:  "
              f"exact {r.exact:.1f}%   within-1 {r.within1:.1f}%   mean err {r.mae:.2f}")

    # ---- 2. held-out check -------------------------------------------------------
    rng = random.Random(0)
    held, base_held, chosen = [], [], {}
    for _ in range(args.splits):
        sh = names[:]
        rng.shuffle(sh)
        a, b = sh[:len(sh) // 2], sh[len(sh) // 2:]
        for tune, test in ((a, b), (b, a)):
            pick = max(((i, c) for i in RES for c in CONF),
                       key=lambda s: exact_rate(raw, ref, s, tune))
            chosen[pick] = chosen.get(pick, 0) + 1
            held.append(exact_rate(raw, ref, pick, test))
            base_held.append(exact_rate(raw, ref, SHIPPED, test))
    print(f"\n{2 * args.splits} tune/test splits — tuned on half, scored on the unseen half:")
    print(f"  tuned setting, held out : {st.mean(held):.1f}%")
    print(f"  shipped setting, same   : {st.mean(base_held):.1f}%")
    print(f"  gain that survives      : {st.mean(held) - st.mean(base_held):+.1f} points")
    top = sorted(chosen.items(), key=lambda kv: -kv[1])[0]
    print(f"  most-chosen setting     : imgsz={top[0][0]}, conf={top[0][1]} "
          f"({100 * top[1] / len(held):.0f}% of splits)")

    # ---- 3. payoff on the benchmark's own counting questions ---------------------
    cl = pd.read_parquet(args.closed)
    cl["q"] = cl.question.astype(str)
    mcq = cl[cl.q.str.strip().str.lower().str.match(PURE)].copy()
    mcq = mcq[mcq[OPTS].map(lambda v: bool(re.fullmatch(r"\s*\d+\s*", str(v)))).all(axis=1)]
    test_imgs = set(mcq.file_name)
    tune_imgs = [n for n in names if n not in test_imgs]
    clean = max(((i, c) for i in RES for c in CONF),
                key=lambda s: exact_rate(raw, ref, s, tune_imgs))

    def detector_mcq(setting):
        hit = 0
        for _, r in mcq.iterrows():
            n = count_at(raw, setting, r.file_name)
            o = [int(str(r[c]).strip()) for c in OPTS]
            pick = min(range(4), key=lambda i: (abs(o[i] - n), i))
            hit += "ABCD"[pick] == str(r.answer).strip().upper()
        return 100 * hit / len(mcq)

    print(f"\n{len(mcq)} pure-count multiple-choice questions on {len(test_imgs)} images.")
    print(f"setting chosen on the {len(tune_imgs)} images carrying none of them: "
          f"imgsz={clean[0]}, conf={clean[1]}")
    print(f"  detector, tuned   : {detector_mcq(clean):.1f}%")
    print(f"  detector, shipped : {detector_mcq(SHIPPED):.1f}%")
    idx = set(mcq["index"])
    scores = []
    for f in glob.glob("results/closed_ended/**/*.csv", recursive=True):
        if "superseded" in f or "n491" not in f:
            continue
        d = pd.read_csv(f, keep_default_na=False)
        if "correct" not in d.columns:
            continue
        s = d[d["index"].isin(idx)]
        if len(s):
            scores.append((100 * s.correct.astype(str).str.lower().isin(["true", "1"]).mean(),
                           os.path.basename(f)[:58]))
    print("  models on the same questions (committed runs, no API calls):")
    for a, nm in sorted(scores, reverse=True):
        print(f"    {a:5.1f}%  {nm}")
    print("    25.0%  chance")
    print(f"\ngrid -> {args.out}")


if __name__ == "__main__":
    main()
