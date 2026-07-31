"""
Step 1 — assign FDI codes from GEOMETRY instead of from the network's 32-way guess.

Why not let the detector do it. An FDI code is decided by where a tooth sits in the arch,
not by how it looks: a lower-left second molar and a lower-right second molar are near
mirror images, so the pixels do not carry the answer. Asking a classifier for it means it
latches onto scanner-specific texture, which is exactly why this detector's counting
transferred DENTEX -> MMOral while its numbering did not. Position is scanner-invariant.

This mirrors what the published pipelines do: Tuzoff et al. 2019 (dmfr.20180051) run a
class-agnostic detector and then a separate numbering stage with a heuristic enforcing the
spatial arrangement of teeth; the 2023 scoping review (isd.20230058) finds detect-then-number
dominant. Nobody does flat 32-way.

The rules, in order:

  1. split the detections into upper and lower arch, by fitting one smooth curve per arch and
     letting each tooth join the curve it is closer to (iterated, so a bad start recovers);
  2. find the midline;
  3. quadrant = arch x side. Panoramics are read facing the patient, so image-left is the
     PATIENT'S RIGHT:  upper-left-of-image = Q1, upper-right = Q2, lower-right = Q3,
     lower-left = Q4;
  4. within a quadrant, walk outward from the midline numbering 1..8;
  5. when the step to the next tooth is much wider than the teeth on either side of it, a
     tooth is missing there — skip that index rather than shifting every later tooth by one.

Rule 5 is the one that matters on real mouths. Without it a single missing premolar renames
every tooth behind it, which is the documented failure mode of numbering systems on
edentulous and disrupted arches.

Needs no ground-truth labels and no GPU: it is arithmetic on the boxes the detector already
produces.
"""

QUADRANT = {("upper", "left"): 1, ("upper", "right"): 2,
            ("lower", "right"): 3, ("lower", "left"): 4}
GAP_FACTOR = 1.55   # step/expected above which a missing tooth is inferred


def _centre(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _fit_quadratic(pts):
    """Least-squares y = a x^2 + b x + c. Falls back to a flat line when under-determined."""
    n = len(pts)
    if n == 0:
        return None
    if n < 3:
        return (0.0, 0.0, sum(y for _, y in pts) / n)
    sx = [0.0] * 5
    sy = [0.0] * 3
    for x, y in pts:
        for k in range(5):
            sx[k] += x ** k
        for k in range(3):
            sy[k] += y * x ** k
    m = [[sx[2], sx[1], sx[0]],
         [sx[3], sx[2], sx[1]],
         [sx[4], sx[3], sx[2]]]
    v = [sy[0], sy[1], sy[2]]
    for i in range(3):                      # Gaussian elimination with partial pivoting
        p = max(range(i, 3), key=lambda r: abs(m[r][i]))
        if abs(m[p][i]) < 1e-12:
            return (0.0, 0.0, sum(y for _, y in pts) / n)
        m[i], m[p] = m[p], m[i]
        v[i], v[p] = v[p], v[i]
        for r in range(i + 1, 3):
            f = m[r][i] / m[i][i]
            for c in range(i, 3):
                m[r][c] -= f * m[i][c]
            v[r] -= f * v[i]
    z = [0.0] * 3
    for i in (2, 1, 0):
        z[i] = (v[i] - sum(m[i][c] * z[c] for c in range(i + 1, 3))) / m[i][i]
    # the normal-equation rows above are built with the x powers descending, so the solved
    # vector is already (a, b, c) for y = a x^2 + b x + c. Do not reverse it.
    return (z[0], z[1], z[2])


def _eval(q, x):
    a, b, c = q
    return a * x * x + b * x + c


def split_arches(centres, iters=8):
    """Return a list of 'upper'/'lower', one per centre. Two fitted curves, nearest wins."""
    if len(centres) < 2:
        return ["upper"] * len(centres)
    ys = sorted(y for _, y in centres)
    cut = ys[len(ys) // 2]
    side = ["upper" if y < cut else "lower" for _, y in centres]
    for _ in range(iters):
        up = [p for p, s in zip(centres, side) if s == "upper"]
        lo = [p for p, s in zip(centres, side) if s == "lower"]
        if not up or not lo:
            break
        qu, ql = _fit_quadratic(up), _fit_quadratic(lo)
        new = ["upper" if abs(y - _eval(qu, x)) <= abs(y - _eval(ql, x)) else "lower"
               for x, y in centres]
        if new == side:
            break
        side = new
    return side


def find_midline(centres, side):
    """x of the arch midline.

    The estimator is the VERTEX of the fitted arch curve, not the median x. An arch is a
    curve whose turning point sits at the front of the mouth, so the vertex is the anatomical
    midline; the median is merely the middle *detection*, and it swings whenever one side
    loses a tooth. Using the median put up to 9 teeth in a quadrant, which cannot happen.

    Both arches vote when both are well conditioned. The median is the fallback only.
    """
    xs = sorted(x for x, _ in centres)
    if not xs:
        return 0.0
    med = xs[len(xs) // 2]
    lo, hi = xs[0], xs[-1]
    span = hi - lo
    votes = []
    for arch in ("upper", "lower"):
        pts = [p for p, s in zip(centres, side) if s == arch]
        if len(pts) < 5:
            continue
        a, b, _ = _fit_quadratic(pts)
        if abs(a) < 1e-9:
            continue
        vertex = -b / (2 * a)
        # a vertex outside the middle half of the detected span is a degenerate fit
        if lo + 0.25 * span < vertex < hi - 0.25 * span:
            votes.append(vertex)
    return sum(votes) / len(votes) if votes else med


def assign_fdi(boxes):
    """boxes: [(x1,y1,x2,y2), ...] -> ['11', '12', ...] aligned with boxes (None if unplaced).

    Codes above 8 in a quadrant cannot exist, so any tooth that would land past the third
    molar is left unnumbered rather than given an impossible code."""
    if not boxes:
        return []
    centres = [_centre(b) for b in boxes]
    side = split_arches(centres)
    mid = find_midline(centres, side)

    out = [None] * len(boxes)
    for arch in ("upper", "lower"):
        for hand in ("left", "right"):
            members = [i for i, s in enumerate(side)
                       if s == arch and ((centres[i][0] < mid) == (hand == "left"))]
            if not members:
                continue
            members.sort(key=lambda i: abs(centres[i][0] - mid))
            q = QUADRANT[(arch, hand)]

            # Expected spacing is calibrated from the steps actually seen in this quadrant,
            # NOT from box width: the detector's boxes overlap, so width overstates true
            # centre-to-centre spacing by about half and the gap test could never fire.
            steps = [abs(centres[members[n]][0] - centres[members[n - 1]][0])
                     for n in range(1, len(members))]
            base = sorted(steps)[len(steps) // 2] if steps else 0.0

            idx = 1
            for n, i in enumerate(members):
                if n and base > 0:
                    step = abs(centres[i][0] - centres[members[n - 1]][0])
                    if step > GAP_FACTOR * base:
                        idx += max(1, int(round(step / base)) - 1)
                if idx > 8:
                    break
                out[i] = f"{q}{idx}"
                idx += 1
    return out


def wisdom_teeth(boxes):
    """The FDI codes ending in 8 that this image actually has (the third molars)."""
    return sorted(c for c in assign_fdi(boxes) if c and c.endswith("8"))


# ---------------------------------------------------------------------------------------
# Positional numbering: assign the FDI index from WHERE a tooth sits, not from how many
# teeth precede it.
#
# assign_fdi() above counts ordinally outward from the midline, so any upstream error --
# one undetected tooth, or a midline off by a couple of percent -- shifts every index
# behind it and the third molar lands on 7. Measured: 39% on wisdom teeth, and better
# detection did not move it (37% -> 39% from v1 to v2), which rules out box quality as
# the cause.
#
# A third molar instead sits at a characteristic DISTANCE from the midline whether or not
# the premolar in front of it is present. So: measure arc length along the fitted arch,
# express it in tooth-widths (self-scaling, so it survives a change of scanner without
# recalibration), and match against a prior calibrated on DENTEX's real expert FDI labels.
# A missing tooth then leaves a hole rather than renaming everything behind it.

def _arc_len(q, x0, x1, steps=24):
    """Arc length along y = ax^2+bx+c between two x values. Sign follows x1-x0."""
    a, b, _ = q
    lo, hi = (x0, x1) if x1 >= x0 else (x1, x0)
    h = (hi - lo) / steps
    total = 0.0
    for i in range(steps):
        t = lo + h * (i + 0.5)
        slope = 2 * a * t + b
        total += h * (1.0 + slope * slope) ** 0.5
    return total if x1 >= x0 else -total


def tooth_positions(boxes):
    """[(quadrant, normalised arc distance from the midline), ...] aligned with boxes.

    Distance is measured along the fitted arch curve and divided by the median box width in
    this image, so the unit is 'tooth-widths' and does not depend on image size or scanner.
    """
    if not boxes:
        return []
    centres = [_centre(b) for b in boxes]
    side = split_arches(centres)
    mid = find_midline(centres, side)
    widths = sorted(abs(b[2] - b[0]) for b in boxes)
    unit = widths[len(widths) // 2] or 1.0

    fits = {}
    for arch in ("upper", "lower"):
        pts = [p for p, s in zip(centres, side) if s == arch]
        fits[arch] = _fit_quadratic(pts) if len(pts) >= 3 else (0.0, 0.0, 0.0)

    out = []
    for (cx, _), s in zip(centres, side):
        hand = "left" if cx < mid else "right"
        d = abs(_arc_len(fits[s], mid, cx)) / unit
        out.append((QUADRANT[(s, hand)], d))
    return out


def assign_fdi_positional(boxes, prior):
    """boxes -> ['11', ...] using a calibrated position prior.

    prior maps an index '1'..'8' to the typical distance in tooth-widths. Within a quadrant
    the teeth are matched to indices by the cheapest assignment that keeps indices strictly
    increasing outward -- a small dynamic program, so one odd tooth cannot scramble the rest.
    """
    pos = tooth_positions(boxes)
    out = [None] * len(boxes)
    for q in (1, 2, 3, 4):
        members = sorted((i for i, (qq, _) in enumerate(pos) if qq == q),
                         key=lambda i: pos[i][1])
        if not members:
            continue
        d = [pos[i][1] for i in members]
        exp = [prior[str(k)] for k in range(1, 9)]
        n, m = len(d), 8
        INF = float("inf")
        # best[i][k] = cheapest way to place the first i teeth using indices up to k
        best = [[INF] * (m + 1) for _ in range(n + 1)]
        back = [[None] * (m + 1) for _ in range(n + 1)]
        for k in range(m + 1):
            best[0][k] = 0.0
        for i in range(1, n + 1):
            for k in range(1, m + 1):
                skip = best[i][k - 1]                       # index k unused (missing tooth)
                take = best[i - 1][k - 1] + abs(d[i - 1] - exp[k - 1])
                if take <= skip:
                    best[i][k], back[i][k] = take, "take"
                else:
                    best[i][k], back[i][k] = skip, "skip"
        i, k = n, m
        while i > 0 and k > 0:
            if back[i][k] == "take":
                out[members[i - 1]] = f"{q}{k}"
                i -= 1
            k -= 1
    return out


def wisdom_teeth_positional(boxes, prior):
    return sorted(c for c in assign_fdi_positional(boxes, prior) if c and c.endswith("8"))
