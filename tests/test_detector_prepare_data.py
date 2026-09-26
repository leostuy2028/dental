# Tests for detector/prepare_data.py
# Builds a YOLO tooth-detection dataset from the raw DENTEX zip. The important part is
# the QC gate: an image with a duplicated FDI code (the same tooth boxed twice) or a
# wildly wrong box count must be excluded, never silently kept. verify_build() then
# re-derives that invariant from the files actually written, independently of qc_image,
# which matters most in --single-class mode where the written labels alone (all "0")
# can no longer prove "no duplicate tooth" by themselves.
#
# No network and no real DENTEX data: main() is run against a tiny local zip file built
# inside the test, with 2-3 images and hand-written annotations.

import io
import json
import zipfile

from PIL import Image

from prepare_data import FDI2CLS, MIN_BOXES, open_zip, qc_image, verify_build

FULL = [f"{q}{t}" for q in (1, 2, 3, 4) for t in range(1, 9)]  # 32 unique codes


# ---------- qc_image ----------

def test_qc_image_full_mouth_is_ok():
    assert qc_image(FULL) == ("ok", "")


def test_qc_image_duplicate_code_is_excluded():
    status, reason = qc_image(["11", "12", "13", "14", "11"])
    assert status == "excluded"
    assert reason.startswith("duplicate_fdi:")
    assert "11" in reason


def test_qc_image_over_32_boxes_excluded():
    status, reason = qc_image(FULL + ["11"] * 8)
    # a 40-box image also has a duplicate, and the duplicate check runs first
    assert status == "excluded"
    assert reason.startswith("duplicate_fdi:")


def test_qc_image_genuine_sparse_mouth_is_ok():
    assert qc_image(["11", "12", "21", "22", "31", "41"]) == ("ok", "")


def test_qc_image_too_few_boxes_excluded():
    assert qc_image(["11", "21"]) == ("excluded", "too_few_boxes")


def test_qc_image_exactly_min_boxes_is_ok():
    codes = [f"1{t}" for t in range(1, MIN_BOXES + 1)]
    assert qc_image(codes) == ("ok", "")


def test_qc_image_empty_is_excluded_as_too_few():
    assert qc_image([]) == ("excluded", "too_few_boxes")


def test_qc_image_over_32_unique_codes_with_no_duplicate_is_excluded():
    # belt-and-suspenders case: 33 codes that are all different (impossible with real
    # FDI codes, since there are only 32, but qc_image takes any strings)
    codes = [f"x{i}" for i in range(33)]
    assert qc_image(codes) == ("excluded", "over_32_boxes")


# ---------- verify_build ----------

def write_dataset(tmp_path, images, single_class):
    """images = {stem: (sidecar_codes, written_label_codes)}"""
    (tmp_path / "labels" / "train").mkdir(parents=True)
    sidecar = {}
    for stem, (codes, written) in images.items():
        sidecar[stem + ".png"] = codes
        lines = [f"{0 if single_class else FDI2CLS[c]} 0.5 0.5 0.1 0.1" for c in written]
        (tmp_path / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines))
    (tmp_path / "fdi_codes.json").write_text(json.dumps(sidecar))


def test_verify_build_accepts_a_clean_full_mouth_32_class(tmp_path):
    write_dataset(tmp_path, {"a": (FULL, FULL)}, single_class=False)
    assert verify_build(str(tmp_path), single_class=False) == []


def test_verify_build_accepts_a_clean_full_mouth_single_class(tmp_path):
    write_dataset(tmp_path, {"a": (FULL, FULL)}, single_class=True)
    assert verify_build(str(tmp_path), single_class=True) == []


def test_verify_build_rejects_a_duplicate_tooth_even_when_flattened(tmp_path):
    # the case this whole file exists for: flattening to class 0 must not hide a
    # duplicate FDI code that slipped through
    dup = FULL[:-1] + ["11"]
    write_dataset(tmp_path, {"a": (dup, dup)}, single_class=True)
    bad = verify_build(str(tmp_path), single_class=True)
    assert bad
    assert "duplicate tooth" in bad[0]


def test_verify_build_rejects_over_32_boxes(tmp_path):
    over = FULL + ["11"] * 8
    write_dataset(tmp_path, {"a": (over, over)}, single_class=True)
    bad = verify_build(str(tmp_path), single_class=True)
    assert any("outside" in b for b in bad)


def test_verify_build_rejects_near_empty_image(tmp_path):
    few = ["11", "21"]
    write_dataset(tmp_path, {"a": (few, few)}, single_class=True)
    bad = verify_build(str(tmp_path), single_class=True)
    assert any("outside" in b for b in bad)


def test_verify_build_rejects_label_sidecar_count_mismatch(tmp_path):
    write_dataset(tmp_path, {"a": (FULL, FULL[:20])}, single_class=True)
    bad = verify_build(str(tmp_path), single_class=True)
    assert any("label lines vs" in b for b in bad)


def test_verify_build_rejects_a_label_file_with_no_sidecar_entry(tmp_path):
    write_dataset(tmp_path, {"a": (FULL, FULL)}, single_class=True)
    (tmp_path / "labels" / "train" / "stowaway.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    bad = verify_build(str(tmp_path), single_class=True)
    assert any("not in the kept set" in b for b in bad)


def test_verify_build_rejects_nonzero_class_in_single_class_mode(tmp_path):
    # sidecar says single class but a label line carries a real class id
    write_dataset(tmp_path, {"a": (["11", "12", "13", "14"], ["11", "12", "13", "14"])},
                 single_class=False)
    bad = verify_build(str(tmp_path), single_class=True)
    assert any("non-zero class" in b for b in bad)


def test_verify_build_rejects_duplicate_class_in_multiclass_mode(tmp_path):
    # written as a repeated class id even though single_class is False
    codes = ["11", "12", "13", "14"]
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "a.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n2 0.5 0.5 0.1 0.1")
    (tmp_path / "fdi_codes.json").write_text(json.dumps({"a.png": codes}))
    bad = verify_build(str(tmp_path), single_class=False)
    assert any("duplicate class" in b for b in bad)


# ---------- open_zip ----------

def test_open_zip_local_path_uses_zipfile(tmp_path):
    zip_path = tmp_path / "d.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("hello.txt", "hi")
    z = open_zip(str(zip_path))
    assert z.read("hello.txt") == b"hi"
    z.close()


def test_open_zip_http_source_goes_through_remotezip(monkeypatch):
    # no real network call: fake out RemoteZip so this only checks the branch is taken
    import remotezip

    class FakeRemoteZip:
        def __init__(self, source):
            self.source = source

    monkeypatch.setattr(remotezip, "RemoteZip", FakeRemoteZip)
    z = open_zip("http://example.com/data.zip")
    assert isinstance(z, FakeRemoteZip)
    assert z.source == "http://example.com/data.zip"


# ---------- main() end to end, against a tiny local zip ----------

def make_png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 32), color=(128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def build_zip(zip_path):
    """A tiny DENTEX-shaped zip: one clean sparse mouth, one duplicate-code image, one
    near-empty image."""
    ann = {
        "categories_1": [{"id": i, "name": str(i)} for i in (1, 2, 3, 4)],
        "categories_2": [{"id": i, "name": str(i)} for i in range(1, 9)],
        "images": [
            {"id": 1, "file_name": "a.png"},
            {"id": 2, "file_name": "b.png"},
            {"id": 3, "file_name": "c.png"},
        ],
        "annotations": [],
    }

    def add(image_id, q, t):
        ann["annotations"].append(
            {"image_id": image_id, "category_id_1": q, "category_id_2": t, "bbox": [1, 1, 10, 10]})

    for q, t in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1)]:  # 6 codes, clean
        add(1, q, t)
    for q, t in [(1, 1), (1, 2), (2, 1), (1, 1)]:  # "11" repeated -> excluded
        add(2, q, t)
    for q, t in [(1, 1), (1, 2)]:  # too few -> excluded
        add(3, q, t)

    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("training_data/quadrant_enumeration/train_quadrant_enumeration.json",
                  json.dumps(ann))
        for name in ("a.png", "b.png", "c.png"):
            z.writestr(f"training_data/quadrant_enumeration/xrays/{name}", make_png_bytes())


def test_main_builds_dataset_and_excludes_bad_images(tmp_path, monkeypatch, capsys):
    zip_path = tmp_path / "training_data.zip"
    build_zip(str(zip_path))
    out_dir = tmp_path / "out"

    monkeypatch.setattr("sys.argv", ["prepare_data.py", "--out", str(out_dir),
                                     "--source", str(zip_path), "--val-frac", "0.0"])
    import prepare_data
    prepare_data.main()

    excluded = json.load(open(out_dir / "excluded.json"))
    assert sorted(excluded["excluded_files"]) == ["b.png", "c.png"]
    assert excluded["reasons"]["c.png"] == "too_few_boxes"
    assert excluded["reasons"]["b.png"].startswith("duplicate_fdi:")

    yaml_text = (out_dir / "dentex.yaml").read_text()
    assert "nc: 32" in yaml_text

    fdi_codes = json.load(open(out_dir / "fdi_codes.json"))
    assert fdi_codes == {"a.png": ["11", "12", "21", "22", "31", "41"]}

    labels = (out_dir / "labels" / "train" / "a.txt").read_text().splitlines()
    assert len(labels) == 6

    out = capsys.readouterr().out
    assert "kept 1 clean images" in out
    assert "EXCLUDED 2 images" in out


def test_main_single_class_flattens_every_box_to_class_zero(tmp_path, monkeypatch):
    zip_path = tmp_path / "training_data.zip"
    build_zip(str(zip_path))
    out_dir = tmp_path / "out"

    monkeypatch.setattr("sys.argv", ["prepare_data.py", "--out", str(out_dir),
                                     "--source", str(zip_path), "--val-frac", "0.0",
                                     "--single-class"])
    import prepare_data
    prepare_data.main()

    yaml_text = (out_dir / "dentex.yaml").read_text()
    assert "nc: 1" in yaml_text
    assert "names: ['tooth']" in yaml_text

    labels = (out_dir / "labels" / "train" / "a.txt").read_text().splitlines()
    assert all(ln.startswith("0 ") for ln in labels)
