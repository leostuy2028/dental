# Tests for paper_analysis/coord_iou.py
# This script pulls "box_2d: [x1,y1,x2,y2]" coordinate boxes out of free text with
# a regex (both the model's answer and the reference), and reports how well the
# model's boxes overlap (IoU) the reference boxes, to check whether a coordinate
# arm is actually pointing at the right place or just emitting well-formed JSON.

import pandas as pd

import coord_iou as m


# ---------- boxes() ----------

def test_boxes_extracts_a_quoted_box_2d():
    assert m.boxes('{"box_2d": [1,2,3,4]}') == [(1, 2, 3, 4)]


def test_boxes_extracts_an_unquoted_spaced_box_2d():
    assert m.boxes("box_2d : [ 10 , 20 , 30 , 40 ]") == [(10, 20, 30, 40)]


def test_boxes_finds_multiple_boxes_in_one_string():
    text = '{"box_2d": [0,0,10,10]} and {"box_2d": [20,20,30,30]}'
    assert m.boxes(text) == [(0, 0, 10, 10), (20, 20, 30, 30)]


def test_boxes_returns_empty_list_when_no_match():
    assert m.boxes("nothing here") == []


def test_boxes_handles_none():
    assert m.boxes(None) == []


# ---------- iou() ----------

def test_iou_full_overlap_is_one():
    assert m.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_no_overlap_is_zero():
    assert m.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    # verified against the real function
    assert m.iou((0, 0, 10, 10), (5, 5, 15, 15)) == 0.14285714285714285


def test_iou_zero_area_boxes_give_zero_not_a_division_error():
    assert m.iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


# ---------- main() ----------

def _write_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_main_prints_iou_summary_for_the_matching_arm(tmp_path, capsys):
    rows = [
        {"index": 1, "arm": "coax_primer_coords", "answer": '{"box_2d": [0,0,10,10]}',
         "gt": '{"box_2d": [0,0,10,10]}'},
        {"index": 2, "arm": "coax_primer_coords", "answer": "box_2d: [0, 0, 10, 10]",
         "gt": "box_2d: [5, 5, 15, 15]"},
        {"index": 3, "arm": "coax_primer_coords", "answer": "no coordinates here",
         "gt": "box_2d: [0,0,5,5]"},
        {"index": 4, "arm": "coax_primer_coords",
         "answer": '{"box_2d": [0,0,10,10]} and {"box_2d": [20,20,30,30]}',
         "gt": '{"box_2d": [0,0,10,10]}'},
        {"index": 5, "arm": "faithful", "answer": '{"box_2d": [1,1,2,2]}', "gt": '{"box_2d": [1,1,2,2]}'},
    ]
    path = tmp_path / "fix.csv"
    _write_csv(path, rows)

    m.main(str(path))

    out = capsys.readouterr().out
    assert f"file: {path}   arm: coax_primer_coords   items: 4" in out
    assert "model emitted boxes: 4   reference boxes: 4" in out
    assert "comparable reference boxes (item had both): 3" in out
    assert "mean   0.714   median 1.000   max 1.000" in out
    assert "IoU>=0.5 (a real hit): 67%   IoU>=0.3: 67%   IoU==0 (miss): 0%" in out


def test_main_with_no_rows_for_the_arm_reports_all_zero(tmp_path, capsys):
    rows = [
        {"index": 1, "arm": "coax_primer_coords", "answer": '{"box_2d": [0,0,10,10]}',
         "gt": '{"box_2d": [0,0,10,10]}'},
    ]
    path = tmp_path / "fix.csv"
    _write_csv(path, rows)

    m.main(str(path), arm="nonexistent")

    out = capsys.readouterr().out
    assert f"file: {path}   arm: nonexistent   items: 0" in out
    assert "model emitted boxes: 0   reference boxes: 0" in out
    assert "mean   0.000   median 0.000   max 0.000" in out
    assert "IoU==0 (miss): 100%" in out
