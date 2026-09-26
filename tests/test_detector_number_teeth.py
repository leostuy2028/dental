# Tests for detector/number_teeth.py
#
# This file assigns FDI tooth codes (like "11", "48") to detected boxes purely from
# geometry: fit a curve through each arch, find the midline, then walk outward from
# it numbering 1..8 per quadrant, and skip an index when a gap says a tooth is missing.
# There is also a second "positional" method that instead matches each tooth's distance
# from the midline against a calibrated prior. This is the most important file for a
# rewrite because it is pure arithmetic with no model in the loop, so every number here
# should come straight from running the real functions.

from number_teeth import (
    _arc_len, _centre, _eval, _fit_quadratic, assign_fdi, assign_fdi_positional,
    find_midline, split_arches, tooth_positions, wisdom_teeth, wisdom_teeth_positional,
)


def arch_boxes(y_base, curve, n_side=8, spacing=20, width=18):
    """Build one arch of small boxes either side of x=0, curving up or down with x^2."""
    boxes = []
    for side in (-1, 1):
        for i in range(1, n_side + 1):
            x = side * i * spacing
            y = y_base + curve * (x ** 2) / 10000.0
            boxes.append((x - width / 2, y - 10, x + width / 2, y + 10))
    return boxes


def full_mouth():
    """32 boxes: an upper arch (curving down, like a smile seen from outside) at y~100
    and a lower arch (curving up) at y~300, 8 teeth either side of the midline in each."""
    return arch_boxes(y_base=100, curve=1.0) + arch_boxes(y_base=300, curve=-1.0)


# ---------- small helpers ----------

def test_centre_is_the_box_midpoint():
    assert _centre((0, 0, 10, 20)) == (5.0, 10.0)


def test_eval_quadratic():
    assert _eval((2.0, 3.0, 4.0), 1.0) == 2.0 + 3.0 + 4.0
    assert _eval((0.0, 0.0, 5.0), 100.0) == 5.0


def test_fit_quadratic_no_points_returns_none():
    assert _fit_quadratic([]) is None


def test_fit_quadratic_one_point_is_a_flat_line_at_that_y():
    assert _fit_quadratic([(1, 2)]) == (0.0, 0.0, 2.0)


def test_fit_quadratic_two_points_falls_back_to_flat_mean_line():
    # fewer than 3 points can't determine a quadratic, so it falls back to a flat
    # line at the mean y (not a straight line through the two points)
    assert _fit_quadratic([(1, 2), (3, 4)]) == (0.0, 0.0, 3.0)


def test_fit_quadratic_singular_system_falls_back_to_flat_mean_line():
    # all three points share the same x, so the normal-equation matrix is singular;
    # the solver detects that and falls back to a flat line at the mean y
    assert _fit_quadratic([(5, 1), (5, 2), (5, 3)]) == (0.0, 0.0, 2.0)


def test_fit_quadratic_recovers_exact_parabola():
    # y = x^2 exactly on these three points
    a, b, c = _fit_quadratic([(0, 0), (1, 1), (2, 4)])
    assert round(a, 6) == 1.0
    assert round(b, 6) == 0.0
    assert round(c, 6) == 0.0


def test_arc_len_zero_for_same_point():
    assert _arc_len((0.0, 0.0, 0.0), 5.0, 5.0) == 0.0


def test_arc_len_flat_line_equals_straight_distance():
    # a=0,b=0 line is flat, so arc length along it is just the x distance
    assert round(_arc_len((0.0, 0.0, 0.0), 0.0, 10.0), 6) == 10.0


def test_arc_len_sign_follows_direction():
    q = (1.0, 0.0, 0.0)
    forward = _arc_len(q, 0.0, 5.0)
    backward = _arc_len(q, 5.0, 0.0)
    assert forward > 0
    assert backward == -forward


# ---------- split_arches ----------

def test_split_arches_empty():
    assert split_arches([]) == []


def test_split_arches_single_point_is_upper():
    assert split_arches([(0, 0)]) == ["upper"]


def test_split_arches_separates_two_clear_rows():
    centres = [(x, 10) for x in range(-80, 80, 20)] + [(x, 200) for x in range(-80, 80, 20)]
    sides = split_arches(centres)
    n = len(centres) // 2
    assert sides[:n] == ["upper"] * n
    assert sides[n:] == ["lower"] * n


# ---------- find_midline ----------

def test_find_midline_empty_is_zero():
    assert find_midline([], []) == 0.0


def test_find_midline_falls_back_to_median_with_few_points():
    # fewer than 5 points per arch means find_midline can't fit+vote, so it uses the
    # median x of everything
    centres = [(-10, 0), (0, 0), (10, 0)]
    side = ["upper", "upper", "upper"]
    assert find_midline(centres, side) == 0.0


def test_find_midline_ignores_a_perfectly_flat_arch_vote():
    # 5+ points but a==0 (dead flat), so the vertex formula -b/2a is undefined; that
    # arch's vote is skipped and the median is used instead
    centres = [(x, 10.0) for x in range(-40, 60, 20)]
    side = ["upper"] * len(centres)
    assert find_midline(centres, side) == 0


def test_find_midline_of_symmetric_full_mouth_is_near_zero():
    centres = [_centre(b) for b in full_mouth()]
    side = split_arches(centres)
    mid = find_midline(centres, side)
    assert abs(mid) < 1.0


# ---------- assign_fdi ----------

def test_assign_fdi_empty_boxes():
    assert assign_fdi([]) == []


def test_assign_fdi_full_mouth_gets_every_code_once():
    boxes = full_mouth()
    codes = assign_fdi(boxes)
    assert None not in codes
    expected = sorted(f"{q}{t}" for q in (1, 2, 3, 4) for t in range(1, 9))
    assert sorted(codes) == expected


def test_assign_fdi_quadrant_layout_matches_the_docstring():
    # image-left, upper arch -> quadrant 1; image-right, upper -> quadrant 2;
    # image-right, lower -> quadrant 3; image-left, lower -> quadrant 4
    boxes = full_mouth()
    codes = assign_fdi(boxes)
    for b, c in zip(boxes, codes):
        cx, cy = _centre(b)
        if cy < 200 and cx < 0:
            assert c.startswith("1")
        elif cy < 200 and cx > 0:
            assert c.startswith("2")
        elif cy > 200 and cx > 0:
            assert c.startswith("3")
        elif cy > 200 and cx < 0:
            assert c.startswith("4")


def test_assign_fdi_skips_the_index_of_a_missing_tooth():
    # remove the 3rd tooth from the image-left upper quadrant (would-be "13"): the
    # teeth after the gap keep their true positional numbers (14..18), they are not
    # shifted down to fill the hole
    boxes = full_mouth()
    del boxes[2]
    codes = assign_fdi(boxes)
    q1_codes = [c for c in codes if c and c.startswith("1")]
    assert q1_codes == ["11", "12", "14", "15", "16", "17", "18"]


def test_assign_fdi_extra_tooth_past_the_eighth_is_left_unnumbered():
    # a 9th tooth in one quadrant has no valid FDI slot (max is 8), so it gets None
    # rather than an impossible code like "19"
    boxes = arch_boxes(y_base=100, curve=1.0, n_side=9)
    codes = assign_fdi(boxes)
    assert codes.count(None) == 2  # the outermost tooth on each side
    numbered = [c for c in codes if c]
    assert sorted(numbered) == [f"1{t}" for t in range(1, 9)] + [f"2{t}" for t in range(1, 9)]


def test_wisdom_teeth_are_the_four_codes_ending_in_8():
    boxes = full_mouth()
    assert wisdom_teeth(boxes) == ["18", "28", "38", "48"]


def test_wisdom_teeth_empty_when_no_boxes():
    assert wisdom_teeth([]) == []


# ---------- tooth_positions ----------

def test_tooth_positions_empty():
    assert tooth_positions([]) == []


def test_tooth_positions_increase_outward_from_midline():
    boxes = full_mouth()
    pos = tooth_positions(boxes)
    # quadrant 1 boxes appear first (8 of them); their distance should increase
    # monotonically the farther out (more negative x) the tooth is
    q1 = [d for q, d in pos if q == 1]
    assert q1 == sorted(q1)
    assert all(q in (1, 2, 3, 4) for q, _ in pos)


# ---------- assign_fdi_positional / wisdom_teeth_positional ----------

def calibrated_prior():
    """A prior whose expected distances line up with full_mouth()'s own geometry,
    read straight off tooth_positions() for a clean mouth."""
    boxes = full_mouth()
    pos = tooth_positions(boxes)
    # boxes are built in FDI order per quadrant (index 1..8 outward), so pos[0:8] are
    # quadrant-1 indices 1..8 in order
    q1 = [d for q, d in pos if q == 1]
    return {str(i + 1): q1[i] for i in range(8)}


def test_assign_fdi_positional_full_mouth_matches_assign_fdi():
    boxes = full_mouth()
    prior = calibrated_prior()
    assert assign_fdi_positional(boxes, prior) == assign_fdi(boxes)


def test_assign_fdi_positional_leaves_a_hole_for_a_missing_tooth():
    # unlike ordinal assign_fdi, the positional method should still recognise the
    # missing "13" as a hole rather than renumbering, because it matches by distance
    boxes = full_mouth()
    prior = calibrated_prior()
    del boxes[2]
    codes = assign_fdi_positional(boxes, prior)
    q1_codes = [c for c in codes if c and c.startswith("1")]
    assert q1_codes == ["11", "12", "14", "15", "16", "17", "18"]


def test_assign_fdi_positional_empty_quadrant_is_skipped():
    prior = calibrated_prior()
    boxes = arch_boxes(y_base=100, curve=1.0, n_side=8)  # only image-left+right upper -> Q1/Q2 only
    codes = assign_fdi_positional(boxes, prior)
    assert all(c is None or c[0] in ("1", "2") for c in codes)


def test_wisdom_teeth_positional_matches_ordinal_on_a_clean_mouth():
    boxes = full_mouth()
    prior = calibrated_prior()
    assert wisdom_teeth_positional(boxes, prior) == wisdom_teeth(boxes)


def test_wisdom_teeth_positional_empty():
    assert wisdom_teeth_positional([], {str(k): float(k) for k in range(1, 9)}) == []
