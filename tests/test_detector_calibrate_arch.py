# Tests for detector/calibrate_arch.py
# Fits the "how far from the midline, in tooth-widths" prior used by positional
# numbering, from DENTEX's real expert FDI labels, then compares ordinal vs positional
# numbering on held-out images. No network: load_annotations() reads a local cache file
# in these tests instead of streaming the DENTEX zip.

import json
import random

from calibrate_arch import load_annotations, per_image, positions_with_truth


def arch_boxes_xyxy(y_base, curve, n_side=8, spacing=20, width=18, jitter=0.0, rng=None):
    boxes = []
    for side in (-1, 1):
        for i in range(1, n_side + 1):
            x = side * i * spacing
            y = y_base + curve * (x ** 2) / 10000.0
            j = (lambda: rng.uniform(-jitter, jitter)) if rng else (lambda: 0.0)
            boxes.append((x - width / 2 + j(), y - 10 + j(), x + width / 2 + j(), y + 10 + j()))
    return boxes


def make_annotations(n_images, jitter=0.0):
    """n_images clean full-mouth annotations, DENTEX-shaped."""
    ann = {
        "categories_1": [{"id": i, "name": str(i)} for i in (1, 2, 3, 4)],
        "categories_2": [{"id": i, "name": str(i)} for i in range(1, 9)],
        "images": [], "annotations": [],
    }
    # full_mouth() in number_teeth builds: upper-left(Q1), upper-right(Q2), then
    # lower-left(Q4), lower-right(Q3), each side outward 1..8 -- match that FDI order
    codes_order = ([f"1{t}" for t in range(1, 9)] + [f"2{t}" for t in range(1, 9)] +
                  [f"4{t}" for t in range(1, 9)] + [f"3{t}" for t in range(1, 9)])
    for img_id in range(1, n_images + 1):
        ann["images"].append({"id": img_id, "file_name": f"{img_id}.png"})
        rng = random.Random(img_id) if jitter else None
        boxes = (arch_boxes_xyxy(100, 1.0, jitter=jitter, rng=rng) +
                arch_boxes_xyxy(300, -1.0, jitter=jitter, rng=rng))
        for code, (x1, y1, x2, y2) in zip(codes_order, boxes):
            q, t = int(code[0]), int(code[1])
            ann["annotations"].append({"image_id": img_id, "category_id_1": q,
                                       "category_id_2": t, "bbox": [x1, y1, x2 - x1, y2 - y1]})
    return ann


# ---------- load_annotations ----------

def test_load_annotations_reads_the_cache_when_present(tmp_path):
    cache = tmp_path / "ann.json"
    cache.write_text(json.dumps({"hello": "world"}))
    assert load_annotations(str(cache), "http://example.com/never-used") == {"hello": "world"}


def test_load_annotations_streams_and_writes_the_cache_when_missing(tmp_path, monkeypatch):
    import remotezip

    class FakeRemoteZip:
        def __init__(self, source):
            self.source = source
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self, name):
            return json.dumps({"from": "remote", "name": name}).encode()

    monkeypatch.setattr(remotezip, "RemoteZip", FakeRemoteZip)
    cache = tmp_path / "subdir" / "ann.json"
    out = load_annotations(str(cache), "http://example.com/data.zip")
    assert out["from"] == "remote"
    # the cache file must now exist, holding exactly what was streamed
    assert json.loads(cache.read_text()) == out


# ---------- per_image ----------

def test_per_image_keeps_only_qc_clean_images():
    ann = make_annotations(n_images=1)
    # add a second, broken image: a duplicated FDI code
    ann["images"].append({"id": 2, "file_name": "2.png"})
    ann["annotations"] += [
        {"image_id": 2, "category_id_1": 1, "category_id_2": 1, "bbox": [0, 0, 5, 5]},
        {"image_id": 2, "category_id_1": 1, "category_id_2": 1, "bbox": [10, 10, 5, 5]},
    ]
    per, clean = per_image(ann)
    assert clean == [1]
    assert len(per[1]) == 32
    assert len(per[2]) == 2  # still recorded in `per`, just excluded from `clean`


def test_per_image_codes_and_boxes_are_xyxy():
    ann = make_annotations(n_images=1)
    per, clean = per_image(ann)
    code, box = per[1][0]
    assert code == "11"
    x, y, w, h = ann["annotations"][0]["bbox"]
    assert box == (x, y, x + w, y + h)


# ---------- positions_with_truth ----------

def test_positions_with_truth_returns_one_row_per_tooth():
    ann = make_annotations(n_images=1)
    per, clean = per_image(ann)
    out = positions_with_truth(per[1])
    assert len(out) == 32
    codes_used = {code for code, _, _ in out}
    expected = {f"{q}{t}" for q in (1, 2, 3, 4) for t in range(1, 9)}
    assert codes_used == expected
    for _, q, d in out:
        assert q in (1, 2, 3, 4)
        assert d >= 0


def test_positions_with_truth_distance_grows_outward_within_a_quadrant():
    ann = make_annotations(n_images=1)
    per, clean = per_image(ann)
    out = positions_with_truth(per[1])
    q1 = [(code, d) for code, q, d in out if code.startswith("1")]
    q1_sorted_by_code = sorted(q1, key=lambda cd: cd[0])
    distances = [d for _, d in q1_sorted_by_code]
    assert distances == sorted(distances)


# ---------- main() end to end, against a cached annotation file ----------

def test_main_writes_a_monotonic_prior_and_scores_both_methods(tmp_path, monkeypatch, capsys):
    ann = make_annotations(n_images=30, jitter=2.0)
    cache = tmp_path / "ann.json"
    cache.write_text(json.dumps(ann))
    out_path = tmp_path / "arch_prior.json"

    monkeypatch.setattr("sys.argv", ["calibrate_arch.py", "--cache", str(cache),
                                     "--out", str(out_path), "--val-frac", "0.2",
                                     "--seed", "0"])
    import calibrate_arch
    calibrate_arch.main()

    prior = json.load(open(out_path))
    assert prior["calibrated_on"] == 24  # 80% of 30 clean images
    median = prior["median"]
    assert list(median) == [str(k) for k in range(1, 9)]
    # a real arch prior must be monotonically increasing outward from the midline
    values = [median[str(k)] for k in range(1, 9)]
    assert values == sorted(values)

    out = capsys.readouterr().out
    assert "30 clean images -> 24 calibrate / 6 held out" in out
    assert "ordinal" in out
    assert "positional" in out
    # this synthetic data has no missing teeth and no noise in the *codes*, so on the
    # TRUE boxes both numbering methods should recover every code exactly
    assert "every tooth 100.0%" in out


def test_main_warns_about_a_non_monotonic_prior(tmp_path, monkeypatch, capsys):
    # build data where quadrant-1 index-1 teeth sit FARTHER out than index 2..5, so the
    # calibrated prior comes out non-monotonic and the script must say so. All 8 tooth
    # indices are present so the print loop over range(1, 9) doesn't KeyError.
    ann = {
        "categories_1": [{"id": i, "name": str(i)} for i in (1, 2, 3, 4)],
        "categories_2": [{"id": i, "name": str(i)} for i in range(1, 9)],
        "images": [], "annotations": [],
    }
    boxes = {1: (-200, 100), 2: (-20, 100), 3: (-40, 100), 4: (-60, 100),
             5: (-80, 100), 6: (-100, 100), 7: (-120, 100), 8: (-140, 100)}
    for img_id in range(1, 11):
        ann["images"].append({"id": img_id, "file_name": f"{img_id}.png"})
        for t, (x, y) in boxes.items():
            ann["annotations"].append({"image_id": img_id, "category_id_1": 1,
                                       "category_id_2": t, "bbox": [x, y, 10, 10]})
    cache = tmp_path / "ann.json"
    cache.write_text(json.dumps(ann))
    out_path = tmp_path / "arch_prior.json"

    monkeypatch.setattr("sys.argv", ["calibrate_arch.py", "--cache", str(cache),
                                     "--out", str(out_path), "--val-frac", "0.2",
                                     "--seed", "0"])
    import calibrate_arch
    calibrate_arch.main()
    out = capsys.readouterr().out
    assert "non-monotonic prior" in out
