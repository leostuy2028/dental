"""
Detection pass (VLM-as-detector PROXY): a model produces an FDI tooth map for each
image — tooth presence, numbering, and location ONLY, with NO findings — so it can
be fed to the answerer as a localization aid without leaking any answer.

This is an honest LOWER BOUND on a real trained tooth-numbering detector: the map is
made by a general VLM, which has the same weaknesses we're studying, so if it helps a
real detector helps more; if it doesn't, it's inconclusive (may just be a weak map).
Independent of the Q&A answers (reads the image only) -> no contamination.

  python -m eval_open.detect_teeth --model gpt-5-mini --effort medium --workers 8 \
      --out reference/teeth_detections.json
"""
import argparse
import json
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from dataio.data_loader import decode_image
import eval_open.run_batched as rb   # reuse its _openai provider + reasoning-effort switch

load_dotenv(".env")
DATA = "data/open_ended.parquet"

DET_PROMPT = (
    "You are a dental tooth-charting tool. This is a panoramic dental X-ray, {w} by {h} pixels, "
    "with (0,0) at the top-left. IMPORTANT: it is a panoramic view, so the patient's RIGHT side is "
    "on the LEFT of the image and the patient's LEFT is on the RIGHT of the image.\n"
    "List EVERY tooth that is present, by its FDI two-digit code. FDI: first digit is the quadrant "
    "(1=upper-right, 2=upper-left, 3=lower-left, 4=lower-right, using the patient's own right/left); "
    "second digit is 1..8 counting from the midline (1=central incisor ... 8=third molar).\n"
    "For each present tooth give a short location word and an approximate bounding box in pixels.\n"
    "Report ONLY which teeth are present, their FDI number, and where they are. Do NOT report caries, "
    "lesions, restorations, implants, or any finding or condition.\n"
    "Output STRICT JSON and nothing else:\n"
    '{{"total_present": <int>, "teeth": [{{"fdi": "18", "loc": "upper-right third molar", '
    '"box_2d": [x1,y1,x2,y2]}}, ...]}}'
)


def detect_one(image_b64, model):
    W, H = decode_image(image_b64).size
    return rb._openai(image_b64, "", DET_PROMPT.format(w=W, h=H), model, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--effort", default="medium", choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--out", default="reference/teeth_detections.json")
    args = ap.parse_args()
    rb.REASONING_EFFORT = args.effort
    print(f"detection: model={args.model} effort={args.effort} workers={args.workers}")

    df = pd.read_parquet(DATA)
    first = df.drop_duplicates("image_name")
    images = first.head(args.n_images)[["image_name", "image"]].values.tolist()

    out = {}
    if os.path.exists(args.out):
        out = json.load(open(args.out, encoding="utf-8"))

    def work(name, b64):
        return name, detect_one(b64, args.model)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, n, b) for n, b in images if n not in out]
        for i, fut in enumerate(as_completed(futs)):
            name, raw = fut.result()
            out[name] = raw
            json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
            print(f"  [{len(out)}/{len(images)}] {name}: {raw[:60].strip()}")
    print(f"wrote {len(out)} tooth maps -> {args.out}")


if __name__ == "__main__":
    main()
