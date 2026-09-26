# Tests for dataio/make_zenodo_exemplars.py
# Picks, for each target condition, the label with the largest box (or a PINNED stem for
# bone loss) and draws only that condition's boxes onto the matching image. Like
# make_dentex_exemplars.py, main() derives "the repo" from its own __file__, so we point
# the module's __file__ at tmp_path.

import json
import os

from PIL import Image

import dataio.make_zenodo_exemplars as m


def _point_module_at_repo(monkeypatch, repo_dir):
    fake_file = os.path.join(str(repo_dir), "dataio", "make_zenodo_exemplars.py")
    monkeypatch.setattr(m, "__file__", fake_file)


def _mk_label_and_image(lbl_dir, img_dir, stem, class_id, size=(100, 100)):
    (lbl_dir / f"{stem}.txt").write_text(f"{class_id} 0.5 0.5 0.2 0.2\n")
    Image.new("RGB", size, (5, 5, 5)).save(img_dir / f"{stem}.png")


def test_main_builds_one_exemplar_per_target_including_pinned_boneloss(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    lbl_dir = root / "train" / "labels"
    img_dir = root / "train" / "images"
    lbl_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    _mk_label_and_image(lbl_dir, img_dir, "lbl0", 0)
    _mk_label_and_image(lbl_dir, img_dir, "lbl1", 1)
    _mk_label_and_image(lbl_dir, img_dir, "lbl2", 2)
    _mk_label_and_image(lbl_dir, img_dir, "lbl3", 3)
    _mk_label_and_image(lbl_dir, img_dir, "57", 5)     # PIN[5] stems
    _mk_label_and_image(lbl_dir, img_dir, "591", 5)

    _point_module_at_repo(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["prog", str(root)])

    m.main()

    out_dir = tmp_path / "reference" / "zenodo_exemplars"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["_source"].startswith("Zenodo 15487430")
    names = {e["image"] for e in manifest["exemplars"]}
    assert names == {
        "zenodo_exemplars/implant.jpg", "zenodo_exemplars/crown.jpg",
        "zenodo_exemplars/filling.jpg", "zenodo_exemplars/rct.jpg",
        "zenodo_exemplars/boneloss1.jpg", "zenodo_exemplars/boneloss2.jpg",
    }
    for stem in ("implant", "crown", "filling", "rct", "boneloss1", "boneloss2"):
        assert (out_dir / f"{stem}.jpg").exists()

    printed = capsys.readouterr().out
    assert "wrote 6 exemplars + manifest.json" in printed


def test_main_skips_a_label_with_no_matching_image_and_downscales_large_ones(tmp_path, monkeypatch):
    root = tmp_path / "root"
    lbl_dir = root / "train" / "labels"
    img_dir = root / "train" / "images"
    lbl_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    # a bigger box with no matching image: must be skipped in favor of the smaller one
    (lbl_dir / "no_image.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (lbl_dir / "hasimg.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    Image.new("RGB", (1500, 1500), (9, 9, 9)).save(img_dir / "hasimg.png")

    _point_module_at_repo(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["prog", str(root)])
    monkeypatch.setattr(m, "TARGETS", [(0, "a dental implant", "implant", 1)])
    monkeypatch.setattr(m, "PIN", {})

    m.main()

    out_dir = tmp_path / "reference" / "zenodo_exemplars"
    img = Image.open(out_dir / "implant.jpg")
    # the 1500x1500 source was downscaled to MAX_PX on its longest side
    assert max(img.size) <= m.MAX_PX
