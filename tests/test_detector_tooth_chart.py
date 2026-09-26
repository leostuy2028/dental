# Tests for detector/tooth_chart.py
# Turns the detector's raw output (a count + a list of numbered teeth) into the
# plain-text paragraph that gets pasted into a model's prompt.

from tooth_chart import anat, build_chart


def test_anat_known_code():
    assert anat("26") == "upper left first molar"


def test_anat_accepts_an_int_fdi():
    assert anat(38) == "lower left third molar (wisdom tooth)"


def test_anat_unknown_quadrant_and_tooth_digit():
    assert anat("99") == "? ?"


def test_anat_unknown_quadrant_only():
    assert anat("91") == "? central incisor"


def test_build_chart_lists_teeth_sorted_by_fdi_string():
    entry = {"count": 30, "teeth": [
        {"fdi": "18", "box": [0, 0, 1, 1], "conf": 0.9},
        {"fdi": "11", "box": [0, 0, 1, 1], "conf": 0.8},
    ]}
    text = build_chart(entry)
    assert "Total teeth detected: 30." in text
    assert "Of these, 2 were numbered:" in text
    # sorted by the fdi string, so "11" (starts with '1') comes before "18"
    i11 = text.index("#11")
    i18 = text.index("#18")
    assert i11 < i18
    assert "#11 = upper right central incisor" in text
    assert "#18 = upper right third molar (wisdom tooth)" in text


def test_build_chart_missing_teeth_key_defaults_to_empty_list():
    text = build_chart({"count": 5})
    assert "Total teeth detected: 5." in text
    assert "Of these, 0 were numbered:" in text


def test_build_chart_mentions_the_caveat_about_unlisted_teeth():
    text = build_chart({"count": 0, "teeth": []})
    assert "not necessarily missing from the mouth" in text


def test_build_chart_includes_the_reliability_preamble():
    text = build_chart({"count": 1, "teeth": []})
    assert "81%" in text
    assert "54%" in text
