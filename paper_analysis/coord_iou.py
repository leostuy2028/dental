"""
Coordinate ACCURACY check for the §6.2 coordinate-elicitation arm (NO API).

Reads the coordinate arm's real answers and the reference boxes, and measures how
well the model's emitted boxes actually match the ground truth (IoU). This is the
half of the finding that separates the two explanations:

  - if the coordinate arm scores HIGHER than the no-coordinate arm, AND
  - the emitted boxes barely overlap the reference boxes (low IoU),
  then the grader is rewarding coordinate PRESENCE/format, not localization.

Both the model answer and the reference may use several JSON shapes, so we extract
every `box_2d: [x1,y1,x2,y2]` array by regex regardless of surrounding structure.

  python -m paper_analysis.coord_iou results/open/coordarms_gpt4o_n30_answers.csv
"""
import re
import sys
import pandas as pd

BOX_RE = re.compile(r'box_2d"?\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')


def boxes(text):
    return [tuple(map(int, m)) for m in BOX_RE.findall(str(text))]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def main(path, arm="coax_primer_coords"):
    df = pd.read_csv(path)
    df = df[df.arm == arm]
    best_ious, n_emit, n_ref, n_pairs = [], 0, 0, 0
    for _, r in df.iterrows():
        mb, rb = boxes(r["answer"]), boxes(r["gt"])
        n_emit += len(mb)
        n_ref += len(rb)
        if not mb or not rb:
            continue
        # best IoU for each reference box against any emitted box
        for gt_box in rb:
            best = max((iou(gt_box, m) for m in mb), default=0.0)
            best_ious.append(best)
            n_pairs += 1
    s = pd.Series(best_ious) if best_ious else pd.Series([0.0])
    print(f"file: {path}   arm: {arm}   items: {df['index'].nunique()}")
    print(f"model emitted boxes: {n_emit}   reference boxes: {n_ref}")
    print(f"comparable reference boxes (item had both): {n_pairs}")
    print(f"best-match IoU vs reference boxes:")
    print(f"  mean   {s.mean():.3f}   median {s.median():.3f}   max {s.max():.3f}")
    print(f"  IoU>=0.5 (a real hit): {(s >= 0.5).mean()*100:.0f}%   "
          f"IoU>=0.3: {(s >= 0.3).mean()*100:.0f}%   IoU==0 (miss): {(s == 0).mean()*100:.0f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "results/open/coordarms_gpt4o_n30_answers.csv")
