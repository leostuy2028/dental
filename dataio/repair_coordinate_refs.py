"""
Turn coordinate-list reference answers back into sentences.

THE PROBLEM. For 82 free-text questions the recorded "correct answer" is not a sentence
but a list of pixel boxes, like:

    [{"box_2d": [1531, 698, 1649, 769], "tooth_id": "36", "label": "Crown"},
     {"box_2d": [1741, 569, 1871, 768], "tooth_id": "38", "label": "Filling"}]

The question that goes with it is an ordinary clinical one ("Which historical treatments
can be detected in the panoramic image?"). So the item is fine and the answer is fine; the
answer is just wrapped in formatting no grader can read and no dentist would write. A
text-only judge scores a model's plain-English answer against a wall of numbers (§6.1).

THE FIX. The boxes carry the clinical labels alongside the coordinates. Drop the
coordinates, keep the labels and tooth numbers, and write the sentence they encode:

    "Tooth #36: crown. Tooth #38: filling."

Nothing is invented. Every word of the repaired reference comes from the released data;
this only discards the pixel coordinates and renders the rest as prose. Items whose JSON
does not parse (a stray curly quote in the released file) are left alone and reported.

Run:    python -m dataio.repair_coordinate_refs
Reads:  data/open_ended.parquet
        results/corrected_benchmark/manifest_open.csv   (which items are REPAIR)
Writes: results/corrected_benchmark/repaired_references.csv
          index, question, original_reference, repaired_reference
"""
import json
import os
import re

import pandas as pd

DATA = "data/open_ended.parquet"
MANIFEST = "results/corrected_benchmark/manifest_open.csv"
OUT = "results/corrected_benchmark/repaired_references.csv"

# keys that carry meaning vs keys that are pure geometry
GEOMETRY = {"box_2d", "point_2d"}
BOOL_LABEL = {"is_impacted": "impacted", "is_wisdom_tooth": "wisdom tooth",
              "is_missing": "missing", "is_erupted": "erupted"}


def findings(node, tooth=None, out=None):
    """Walk a reference's JSON and collect (tooth, finding) pairs. Coordinates are dropped."""
    out = [] if out is None else out
    if isinstance(node, list):
        for item in node:
            findings(item, tooth, out)
        return out
    if not isinstance(node, dict):
        return out

    t = node.get("tooth_id", tooth)
    for key, val in node.items():
        if key in GEOMETRY or key == "tooth_id":
            continue
        if key == "label":
            out.append((t, str(val)))
        elif key in BOOL_LABEL:
            if str(val).strip().lower().strip('"') in ("true", "yes"):
                out.append((t, BOOL_LABEL[key]))
        elif isinstance(val, (dict, list)):
            # nested shape: {"Crown": {"box_2d": [...]}} -> the KEY is the finding
            if not (isinstance(val, dict) and set(val) <= GEOMETRY):
                findings(val, t, out)
            if key.lower() not in ("teeth position",):
                out.append((t, key))
        else:
            out.append((t, f"{key} {val}"))
    return out


def to_sentence(pairs, teeth_seen=()):
    """(tooth, finding) pairs -> a plain clinical sentence, grouped by tooth.

    `teeth_seen` are tooth numbers that appear in the reference with no finding attached.
    Those still answer a "which tooth" question, so they are reported as bare identifications
    rather than discarded."""
    by_tooth, loose = {}, []
    for tooth, finding in pairs:
        f = str(finding).strip().strip('"').strip()
        if not f:
            continue
        if tooth:
            key = str(tooth).strip().strip('"')
            by_tooth.setdefault(key, [])
            if f not in by_tooth[key]:
                by_tooth[key].append(f)
        elif f not in loose:
            loose.append(f)
    parts = [f"Tooth #{t}: {', '.join(v).lower()}." for t, v in by_tooth.items() if v]
    bare = [str(t).strip().strip('"') for t in teeth_seen
            if str(t).strip().strip('"') not in by_tooth]
    if bare:
        parts.append(f"Tooth {', '.join('#' + t for t in dict.fromkeys(bare))}.")
    if loose:
        parts.append(("Also noted: " + ", ".join(loose).lower() + ".") if parts
                     else ("Findings: " + ", ".join(loose).lower() + "."))
    return " ".join(parts)


def teeth_in(node, out=None):
    """Every tooth_id mentioned anywhere in the reference."""
    out = [] if out is None else out
    if isinstance(node, list):
        for x in node:
            teeth_in(x, out)
    elif isinstance(node, dict):
        if "tooth_id" in node:
            out.append(node["tooth_id"])
        for v in node.values():
            if isinstance(v, (dict, list)):
                teeth_in(v, out)
    return out


def repair(reference):
    """Return the repaired sentence, or None if the reference cannot be parsed."""
    try:
        obj = json.loads(str(reference))
    except Exception:
        return None
    text = to_sentence(findings(obj), teeth_seen=teeth_in(obj))
    return text or None


def main():
    op = pd.read_parquet(DATA).set_index("index")
    man = pd.read_csv(MANIFEST).set_index("index")["disposition"]
    targets = man[man == "REPAIR"].index

    rows, failed = [], []
    for idx in targets:
        ref = op.loc[idx, "answer"]
        fixed = repair(ref)
        if fixed:
            rows.append({"index": idx, "question": op.loc[idx, "question"],
                         "original_reference": str(ref), "repaired_reference": fixed})
        else:
            failed.append(idx)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"REPAIR items: {len(targets)}   repaired: {len(rows)}   no recoverable content: {len(failed)}")
    if failed:
        print(f"  indices with nothing to recover: {failed}")
    print("\nexamples:")
    for r in rows[:4]:
        print(f"\n  Q  : {r['question'][:88]}")
        print(f"  was: {r['original_reference'][:88]}...")
        print(f"  now: {r['repaired_reference'][:88]}")
    print(f"\nwrote {OUT}")
    print("Every word of a repaired reference comes from the released data; only the pixel "
          "coordinates were discarded.")


if __name__ == "__main__":
    main()
