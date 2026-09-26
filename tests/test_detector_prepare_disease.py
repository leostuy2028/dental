# Tests for detector/prepare_disease.py
# Builds a YOLO dataset for the DISEASE classes (Impacted, Caries, Deep Caries,
# Periapical Lesion). Unlike prepare_data.py, an image here only has the ABNORMAL teeth
# boxed, so the same-code-twice rule does not apply; the only gate is "at least one
# finding, not an absurd number of them". Tested against a tiny local zip, no network.

import io
import json
import zipfile

from PIL import Image

from prepare_disease import MAX_BOXES, open_zip, qc_image


# ---------- qc_image ----------

def test_qc_image_zero_findings_is_excluded():
    assert qc_image(0) == ("excluded", "no_findings")


def test_qc_image_one_finding_is_ok():
    assert qc_image(1) == ("ok", "")


def test_qc_image_at_the_max_is_ok():
    assert qc_image(MAX_BOXES) == ("ok", "")


def test_qc_image_over_the_max_is_excluded():
    assert qc_image(MAX_BOXES + 1) == ("excluded", f"over_{MAX_BOXES}_boxes")


# ---------- open_zip ----------

def test_open_zip_local_path_uses_zipfile(tmp_path):
    zip_path = tmp_path / "d.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("hello.txt", "hi")
    z = open_zip(str(zip_path))
    assert z.read("hello.txt") == b"hi"
    z.close()


def test_open_zip_http_source_goes_through_remotezip(monkeypatch):
    import remotezip

    class FakeRemoteZip:
        def __init__(self, source):
            self.source = source

    monkeypatch.setattr(remotezip, "RemoteZip", FakeRemoteZip)
    z = open_zip("http://example.com/data.zip")
    assert isinstance(z, FakeRemoteZip)


# ---------- main() end to end, against a tiny local zip ----------

def make_png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 32), color=(128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def build_zip(zip_path):
    """One clean image (an Impacted + a Caries finding) and one image with too many
    findings to be plausible for one mouth."""
    ann = {
        "categories_1": [{"id": i, "name": str(i)} for i in (1, 2, 3, 4)],
        "categories_2": [{"id": i, "name": str(i)} for i in range(1, 9)],
        "categories_3": [{"id": 0, "name": "Impacted"}, {"id": 1, "name": "Caries"},
                        {"id": 2, "name": "Deep Caries"}, {"id": 3, "name": "Periapical Lesion"}],
        "images": [{"id": 1, "file_name": "a.png"}, {"id": 2, "file_name": "b.png"}],
        "annotations": [],
    }

    def add(image_id, c3, q, t):
        ann["annotations"].append({"image_id": image_id, "category_id_1": q,
                                   "category_id_2": t, "category_id_3": c3,
                                   "bbox": [1, 1, 10, 10]})

    add(1, 0, 1, 6)   # Impacted at FDI 16
    add(1, 1, 2, 1)   # Caries at FDI 21
    for _ in range(MAX_BOXES + 1):   # over the limit -> excluded
        add(2, 1, 1, 1)

    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(
            "training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
            json.dumps(ann))
        for name in ("a.png", "b.png"):
            z.writestr(f"training_data/quadrant-enumeration-disease/xrays/{name}",
                      make_png_bytes())


def test_main_builds_dataset_and_excludes_overloaded_image(tmp_path, monkeypatch, capsys):
    zip_path = tmp_path / "training_data.zip"
    build_zip(str(zip_path))
    out_dir = tmp_path / "out"

    monkeypatch.setattr("sys.argv", ["prepare_disease.py", "--out", str(out_dir),
                                     "--source", str(zip_path), "--val-frac", "0.0"])
    import prepare_disease
    prepare_disease.main()

    findings = json.load(open(out_dir / "findings.json"))
    assert findings == {"a.png": [{"cls": "Impacted", "fdi": "16"},
                                 {"cls": "Caries", "fdi": "21"}]}

    yaml_text = (out_dir / "dentex.yaml").read_text()
    assert "nc: 4" in yaml_text
    assert "Impacted" in yaml_text

    # b.png had too many findings and must not have been written at all
    assert not (out_dir / "labels" / "train" / "b.txt").exists()
    assert (out_dir / "labels" / "train" / "a.txt").exists()

    out = capsys.readouterr().out
    assert "kept 1 images" in out
    assert "excluded 1" in out
    assert "VERIFY OK" in out


def test_main_label_file_has_one_line_per_finding(tmp_path, monkeypatch):
    zip_path = tmp_path / "training_data.zip"
    build_zip(str(zip_path))
    out_dir = tmp_path / "out"

    monkeypatch.setattr("sys.argv", ["prepare_disease.py", "--out", str(out_dir),
                                     "--source", str(zip_path), "--val-frac", "0.0"])
    import prepare_disease
    prepare_disease.main()

    lines = (out_dir / "labels" / "train" / "a.txt").read_text().splitlines()
    assert len(lines) == 2
    classes = sorted(ln.split()[0] for ln in lines)
    assert classes == ["0", "1"]  # Impacted=0, Caries=1
