"""
Render the detector's output as the plain-text tooth chart injected into a prompt.

The chart tells the model three things beyond the numbers themselves: HOW they were produced,
HOW ACCURATE each part of the output has been measured to be, and that it is free to ignore
them. That is deliberate on all three counts.

Stating the accuracy lets the model weight the hint instead of deferring to it. The two halves
of the output are very differently reliable — counting is 81% exact, numbering only 54% — and a
model told nothing has no way to know it should trust one more than the other. The earlier
injection experiments handed over v1's 32-class numbering, which was ~75% wrong on MMOral, with
no reliability statement at all; they could not separate "a model cannot use a tooth map" from
"a model cannot use a WRONG tooth map presented as fact".

Every figure quoted in the preamble is measured and traceable:
  counting on MMOral        results/detector/inference_sweep_v2.csv   (81.4% exact, 94.2% w/1)
  numbering, held-out DENTEX  detector/calibrate_arch.py             (53.6% all, 76.9% 3rd molars)
  wisdom sets on MMOral     results/detector/numbering_wisdom_v2_positional.csv  (58%)
"""

ANAT = {"1": "central incisor", "2": "lateral incisor", "3": "canine",
        "4": "first premolar", "5": "second premolar", "6": "first molar",
        "7": "second molar", "8": "third molar (wisdom tooth)"}
QUAD = {"1": "upper right", "2": "upper left", "3": "lower left", "4": "lower right"}

PREAMBLE = (
    "A tooth-detection tool was run on this X-ray. You may use its output if you find it "
    "helpful, and you should ignore it wherever it disagrees with what you can see in the "
    "image yourself.\n"
    "\n"
    "How it was produced: a small object detector (YOLOv8-small, 11M parameters) trained on "
    "DENTEX, a public set of panoramic X-rays with expert tooth annotations. It locates each "
    "tooth but does not name it. The FDI numbers below are then assigned by geometry, not by "
    "the network: each arch is fitted, the midline is found, and each tooth is matched to the "
    "position whose typical distance from the midline it sits closest to.\n"
    "\n"
    "How accurate it is, measured on held-out data:\n"
    "  - The TOTAL COUNT is its most reliable output: exactly right on 81% of X-rays, and "
    "within one tooth on 94%.\n"
    "  - The FDI NUMBERS are considerably less reliable: about 54% of teeth get the correct "
    "number on held-out images. Third molars are the strongest case at 77%, because they sit "
    "at the end of the arch. Numbering degrades most where teeth are missing or crowded.\n"
    "  - It was trained on a different X-ray machine than this one. Counting transferred "
    "across that change; numbering transferred less well.\n"
    "\n"
    "So: treat the count as strong evidence, and the individual numbers as a suggestion to "
    "check against the image rather than a fact to rely on.\n")


def anat(fdi):
    fdi = str(fdi)
    return f"{QUAD.get(fdi[0], '?')} {ANAT.get(fdi[1], '?')}"


def build_chart(entry):
    """entry = {'count': int, 'teeth': [{fdi, box, conf}]} -> plain text for the prompt."""
    teeth = sorted(entry.get("teeth", []), key=lambda t: str(t["fdi"]))
    lines = [f"  #{t['fdi']} = {anat(t['fdi'])}" for t in teeth]
    return (PREAMBLE
            + "\nIts output for this X-ray:\n"
            + f"Total teeth detected: {entry['count']}.\n"
            + f"Of these, {len(teeth)} were numbered:\n"
            + "\n".join(lines)
            + "\n(Teeth located but not confidently numbered are not listed. A tooth absent "
              "from this list is not necessarily missing from the mouth.)")
