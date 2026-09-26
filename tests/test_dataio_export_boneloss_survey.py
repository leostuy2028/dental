# Tests for dataio/export_boneloss_survey.py
# Builds the blinded bone-loss survey as one portable HTML file: each radiograph is
# embedded as a base64 data URI, and the page must never reveal the key, model answers,
# or bucket (it only ever sees image_name and survey_order from the manifest).

import base64

import pandas as pd

import dataio.export_boneloss_survey as m


def test_decode_jpeg_strips_a_data_uri_prefix_if_present():
    raw = base64.b64encode(b"hi").decode()
    assert m.decode_jpeg("data:image/jpeg;base64," + raw) == b"hi"
    assert m.decode_jpeg(raw) == b"hi"


def test_main_embeds_every_image_and_stays_blind(tmp_path, monkeypatch, capsys):
    manifest_p = tmp_path / "manifest.csv"
    open_p = tmp_path / "open.parquet"
    out_p = tmp_path / "survey" / "boneloss.html"

    img_b64 = base64.b64encode(b"jpegbytes").decode()
    man = pd.DataFrame({
        "survey_order": [1, 2], "item_id": ["bl01", "bl02"],
        "image_name": ["img1.jpg", "img2.jpg"],
        # these columns exist in the real manifest but must never reach the page
        "bucket": ["DISAGREE", "KEY_POS"], "key_stance": ["none", "loss"],
    })
    man.to_csv(manifest_p, index=False)
    op = pd.DataFrame({"image_name": ["img1.jpg", "img2.jpg"], "image": [img_b64, img_b64]})
    op.to_parquet(open_p, index=False)

    monkeypatch.setattr(m, "MANIFEST", str(manifest_p))
    monkeypatch.setattr(m, "OPEN", str(open_p))
    monkeypatch.setattr(m, "OUT", str(out_p))

    m.main()

    html = out_p.read_text(encoding="utf-8")
    assert "bl01" in html and "bl02" in html
    assert "data:image/jpeg;base64," + base64.b64encode(b"jpegbytes").decode() in html
    assert "Radiograph 1 of 2" in html and "Radiograph 2 of 2" in html
    # blind: none of the manifest's hidden columns are rendered anywhere
    assert "DISAGREE" not in html and "KEY_POS" not in html
    assert m.QUESTION in html

    printed = capsys.readouterr().out
    assert "wrote" in printed and "2 images embedded, blind: no key/model/bucket" in printed
