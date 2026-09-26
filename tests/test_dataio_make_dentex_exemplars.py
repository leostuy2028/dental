# Tests for dataio/make_dentex_exemplars.py
# Draws each finding's real DENTEX bounding box + label onto the image and writes a
# manifest.json. main() computes its output dir as reference/dentex_exemplars relative
# to the repo root it derives from `os.path.dirname(os.path.dirname(abspath(__file__)))`
# -- since that isn't a module constant, we point the module's own __file__ at a fake
# path inside tmp_path so "the repo" becomes tmp_path for the test.

import json
import os

from PIL import Image, ImageFont

import dataio.make_dentex_exemplars as m


def _build_source(tmp_path):
    src = tmp_path / "src"
    xr = src / "valimg" / "validation_data" / "quadrant_enumeration_disease" / "xrays"
    xr.mkdir(parents=True)

    ann = {
        "categories_1": [{"id": 0, "name": "1"}],
        "categories_2": [{"id": 0, "name": "1"}],
        "categories_3": [{"id": 0, "name": "Impacted"}],
        "images": [{"id": i, "file_name": f"{stem}.png"} for i, stem in enumerate(m.CHOSEN)],
        "annotations": [
            {"image_id": i, "category_id_1": 0, "category_id_2": 0, "category_id_3": 0, "bbox": [1, 1, 5, 5]}
            for i, stem in enumerate(m.CHOSEN)
        ],
    }
    (src / "validation_triple.json").write_text(json.dumps(ann))

    for stem in m.CHOSEN:
        # one oversized image so the MAX_PX downscale branch runs too
        size = (1200, 1200) if stem == m.CHOSEN[0] else (50, 50)
        Image.new("RGB", size, (1, 2, 3)).save(xr / f"{stem}.png")
    return src


def _point_module_at_repo(monkeypatch, repo_dir):
    fake_file = os.path.join(str(repo_dir), "dataio", "make_dentex_exemplars.py")
    monkeypatch.setattr(m, "__file__", fake_file)


def test_main_draws_boxes_and_writes_manifest(tmp_path, monkeypatch, capsys):
    src = _build_source(tmp_path)
    _point_module_at_repo(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["prog", str(src)])

    m.main()

    out_dir = tmp_path / "reference" / "dentex_exemplars"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["_source"].startswith("DENTEX")
    assert len(manifest["exemplars"]) == len(m.CHOSEN)
    for stem in m.CHOSEN:
        assert (out_dir / f"{stem}.jpg").exists()
    # the caption is built verbatim from the released label, not invented
    assert manifest["exemplars"][0]["caption"] == "Labeled findings (red boxes, FDI numbering): #11 impacted."

    # the oversized first image was downscaled to at most MAX_PX on its longest side
    resized = Image.open(out_dir / f"{m.CHOSEN[0]}.jpg")
    assert max(resized.size) <= m.MAX_PX

    printed = capsys.readouterr().out
    assert f"wrote {len(m.CHOSEN)} exemplars + manifest.json" in printed


def test_main_falls_back_to_the_default_font_when_arial_is_unavailable(tmp_path, monkeypatch):
    src = _build_source(tmp_path)
    _point_module_at_repo(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["prog", str(src)])

    real_truetype = ImageFont.truetype

    def fake_truetype(name, *a, **k):
        if name == "arial.ttf":
            raise OSError("no font")
        return real_truetype(name, *a, **k)

    monkeypatch.setattr(m.ImageFont, "truetype", fake_truetype)

    m.main()   # should not raise even though arial.ttf "isn't installed"

    out_dir = tmp_path / "reference" / "dentex_exemplars"
    assert (out_dir / "manifest.json").exists()
