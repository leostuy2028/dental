"""
Construct the paired prediction variants for the coordinate-bias isolation study.

For every open-ended item we build two predictions that carry the SAME clinical
content but differ ONLY in format:

  prose  : clinical findings in plain text, all coordinates removed.
  coords : the same findings WITH coordinate/bbox data.

- coord_ref items (reference answer is a JSON/bbox dump):
    coords = the raw JSON reference (exactly what a coordinate-emitting model such
             as OralGPT outputs), prose = the findings with coordinates stripped.
- prose_ref items (reference answer is plain text):
    prose  = the reference text, coords = the reference text + DETERMINISTIC SYNTHETIC
             boxes attached to each mentioned tooth (flagged synthetic; tests whether
             the coordinate halo bleeds beyond the 101 coord_ref items).

Output: eval_open/predictions.parquet with columns
  index, question, answer (=ground truth), category, ref_type, pred_prose, pred_coords
Run:  python eval_open/build_predictions.py
"""
import re
import json
import random
import pandas as pd
from pathlib import Path

DATA = "data/open_ended.parquet"
OUT = "eval_open/predictions.parquet"

# keys that are pure geometry / bookkeeping, never a clinical finding label
_GEOM_KEYS = {"point_2d", "box_2d", "tooth_id", "label", "is_impacted",
              "is_wisdom_tooth", "Teeth position"}
# plausible OPG pixel bounds (from observed reference coordinates)
_IMG_W, _IMG_H = 2900, 1300


def is_coord_ref(answer):
    """coord_ref := the reference answer is a JSON/bbox coordinate dump."""
    s = str(answer).strip()
    return (s.startswith("[") or s.startswith("{")) and ("box_2d" in s or "point_2d" in s)


def _flag_true(s, key):
    """True if `"key": true/True/"True"` appears (tolerant of quoting/case)."""
    return bool(re.search(rf'"{key}"\s*:\s*"?[Tt]rue"?', s))


def _flag_false(s, key):
    return bool(re.search(rf'"{key}"\s*:\s*"?[Ff]alse"?', s))


def coord_ref_to_prose(raw, question=""):
    """Regex-extract clinical content from a JSON coordinate reference, dropping all
    numbers/coordinates. Robust to the 13 truncated/invalid-JSON references."""
    s = str(raw)
    tooth_ids = re.findall(r'"tooth_id"\s*:\s*"?(\d+)"?', s)
    labels = re.findall(r'"label"\s*:\s*"([^"]+)"', s)
    # single-key finding dicts:  "Crown": { "box_2d": ... }  /  "Filling": {...}
    finding_keys = re.findall(r'"([^"]+)"\s*:\s*\{\s*"(?:box_2d|point_2d)"', s)
    findings = [k for k in finding_keys if k not in _GEOM_KEYS]
    findings += [l for l in labels if l not in findings]
    # impaction / wisdom flags (values are quoted "True"/"False" in this data)
    flags = []
    if _flag_true(s, "is_wisdom_tooth"):
        flags.append("wisdom tooth")
    if _flag_true(s, "is_impacted"):
        flags.append("impacted")
    elif _flag_false(s, "is_impacted"):
        flags.append("not impacted")

    teeth = sorted(set(f"#{t}" for t in tooth_ids), key=lambda x: int(x[1:]))
    # pure-localization references carry no tooth_id in the answer; recover it from
    # the question (e.g. "condition of tooth #22") so the prose still names the tooth.
    if not teeth:
        q_teeth = re.findall(r"#(\d{1,2})", str(question))
        teeth = sorted(set(f"#{t}" for t in q_teeth), key=lambda x: int(x[1:]))

    parts = []
    if teeth:
        parts.append(("Teeth " + ", ".join(teeth)) if len(teeth) > 1 else ("Tooth " + teeth[0]))
    if findings:
        parts.append("findings: " + ", ".join(dict.fromkeys(findings)))
    if flags:
        parts.append(", ".join(dict.fromkeys(flags)))
    prose = "; ".join(parts).strip()
    if prose:
        return prose
    # only a bare localization point and no tooth in the question
    return "The indicated tooth is present and localized on the radiograph."


def _synthetic_box(rng):
    x1 = rng.randint(0, _IMG_W - 200)
    y1 = rng.randint(0, _IMG_H - 200)
    return [x1, y1, x1 + rng.randint(60, 180), y1 + rng.randint(60, 180)]


def prose_ref_add_coords(text, index):
    """Append deterministic SYNTHETIC boxes for each tooth # mentioned in a prose
    reference. Reproducible via a per-item seed."""
    rng = random.Random(index)
    teeth = re.findall(r"#(\d{1,2})", str(text))
    teeth = list(dict.fromkeys(teeth))  # unique, order-preserving
    if not teeth:
        # no explicit teeth -> attach a couple of generic point localizations
        pts = [{"point_2d": [rng.randint(0, _IMG_W), rng.randint(0, _IMG_H)]} for _ in range(2)]
        return f"{text} {json.dumps(pts)}"
    boxes = [{"tooth_id": t, "box_2d": _synthetic_box(rng)} for t in teeth]
    return f"{text} {json.dumps(boxes)}"


def build():
    df = pd.read_parquet(DATA)
    rows = []
    for _, r in df.iterrows():
        gt = r["answer"]
        coord = is_coord_ref(gt)
        if coord:
            pred_prose = coord_ref_to_prose(gt, r["question"])
            pred_coords = str(gt).strip()  # raw JSON = OralGPT-style output
        else:
            pred_prose = str(gt).strip()
            pred_coords = prose_ref_add_coords(gt, int(r["index"]))
        rows.append({
            "index": int(r["index"]),
            "question": r["question"],
            "answer": str(gt),
            "category": r["category"],
            "ref_type": "coord_ref" if coord else "prose_ref",
            "pred_prose": pred_prose,
            "pred_coords": pred_coords,
        })
    out = pd.DataFrame(rows)
    Path("eval_open").mkdir(exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} items "
          f"({(out.ref_type=='coord_ref').sum()} coord_ref, "
          f"{(out.ref_type=='prose_ref').sum()} prose_ref)")
    return out


if __name__ == "__main__":
    out = build()
    print("\n=== coord_ref example ===")
    ex = out[out.ref_type == "coord_ref"].iloc[0]
    print("Q     :", ex.question[:90])
    print("GT    :", ex.answer[:120])
    print("prose :", ex.pred_prose)
    print("coords:", ex.pred_coords[:120])
    print("\n=== prose_ref example ===")
    ex = out[out.ref_type == "prose_ref"].iloc[1]
    print("Q     :", ex.question[:90])
    print("GT    :", ex.answer[:120])
    print("prose :", ex.pred_prose)
    print("coords:", ex.pred_coords)
