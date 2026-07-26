"""
Tests for the data-hygiene gate, in BOTH class modes.

The gate exists for one reason: a counter must never learn from a mis-counted image. Flattening
to a single class destroys the FDI codes, so this checks that the same invariant still holds
when the written labels can no longer prove it by themselves.

A check that cannot fail is worth nothing, so every case here plants a specific defect and
asserts the build refuses it.

Run:  python detector/test_prepare_qc.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import FDI2CLS, MIN_BOXES, qc_image, verify_build

FULL = [f"{q}{t}" for q in (1, 2, 3, 4) for t in range(1, 9)]   # 32 unique codes


def build(tmp, images, single_class):
    """Write a fake dataset. images = {stem: (codes_in_sidecar, codes_written_as_labels)}."""
    os.makedirs(f"{tmp}/labels/train", exist_ok=True)
    sidecar = {}
    for stem, (codes, written) in images.items():
        sidecar[stem + ".png"] = codes
        with open(f"{tmp}/labels/train/{stem}.txt", "w") as f:
            f.write("\n".join(f"{0 if single_class else FDI2CLS[c]} 0.5 0.5 0.1 0.1"
                              for c in written))
    with open(f"{tmp}/fdi_codes.json", "w") as f:
        json.dump(sidecar, f)


def check(name, images, single_class, should_pass):
    tmp = tempfile.mkdtemp()
    try:
        build(tmp, images, single_class)
        bad = verify_build(tmp, single_class)
        ok = (not bad) == should_pass
        mode = "1-class" if single_class else "32-class"
        print(f"  [{'PASS' if ok else 'FAIL'}] {mode:8} {name}")
        if not ok:
            print(f"         expected {'clean' if should_pass else 'a rejection'}, got {bad[:2]}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("qc_image — which images are let in at all:")
    gate = [
        ("32 unique codes (full mouth)", FULL, "ok"),
        ("33 boxes, so one code repeats", FULL + ["11"], "excluded"),
        ("same tooth boxed twice", ["11", "12", "13", "14", "11"], "excluded"),
        ("40 boxes (the worst DENTEX case)", FULL + ["11"] * 8, "excluded"),
        ("genuine sparse mouth", ["11", "12", "21", "22", "31", "41"], "ok"),
        ("broken near-empty annotation", ["11", "21"], "excluded"),
    ]
    ok = True
    for name, codes, want in gate:
        got = qc_image(codes)[0]
        good = got == want
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] n={len(codes):3d}  {want:8} {name}")

    print("\nverify_build — does a defect that slipped past the gate get caught:")
    good_pair = (FULL, FULL)
    ok &= check("a clean full mouth is accepted", {"a": good_pair}, True, True)
    ok &= check("a clean full mouth is accepted", {"a": good_pair}, False, True)

    # the case this whole exercise is about: a duplicate tooth that flattening would hide
    dup = (FULL[:-1] + ["11"], FULL[:-1] + ["11"])
    ok &= check("duplicate tooth is REJECTED even when flattened to class 0",
                {"a": dup}, True, False)

    over = (FULL + ["11"] * 8, FULL + ["11"] * 8)
    ok &= check("40-box image is rejected", {"a": over}, True, False)

    few = (["11", "21"], ["11", "21"])
    ok &= check("near-empty image is rejected", {"a": few}, True, False)

    # label file and sidecar disagreeing means the writer dropped or added boxes
    ok &= check("label/sidecar count mismatch is rejected",
                {"a": (FULL, FULL[:20])}, True, False)

    # a label file with no sidecar entry never passed the gate
    tmp = tempfile.mkdtemp()
    try:
        build(tmp, {"a": good_pair}, True)
        with open(f"{tmp}/labels/train/stowaway.txt", "w") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
        bad = verify_build(tmp, True)
        good = any("not in the kept set" in b for b in bad)
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] 1-class  an unlisted label file is rejected")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL PASS" if ok else "\nSOME TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
