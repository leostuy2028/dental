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
