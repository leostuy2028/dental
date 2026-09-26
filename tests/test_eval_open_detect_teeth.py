# Tests for eval_open/detect_teeth.py
# This script asks a model to list which teeth are present in an X-ray (no
# findings), one call per image, and saves the raw replies to a json file.

import base64
import json
from io import BytesIO

import pandas as pd
from PIL import Image

import eval_open.detect_teeth as dt
import eval_open.run_batched as rb


def make_b64_image(w, h, color=(200, 50, 50)):
    img = Image.new("RGB", (w, h), color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------- detect_one ----------

def test_detect_one_uses_real_image_size_in_the_prompt(monkeypatch):
    b64 = make_b64_image(37, 21)
    captured = {}

    def fake_openai(image_b64, system, user, model, exemplars):
        captured["args"] = (image_b64, system, user, model, exemplars)
        return "FAKE DETECTION JSON"

    monkeypatch.setattr(rb, "_openai", fake_openai)
    result = dt.detect_one(b64, "gpt-5-mini")

    assert result == "FAKE DETECTION JSON"
    image_b64, system, user, model, exemplars = captured["args"]
    assert image_b64 == b64
    assert system == ""
    assert user == dt.DET_PROMPT.format(w=37, h=21)
    assert model == "gpt-5-mini"
    assert exemplars is None


# ---------- main ----------

def _write_open_ended(tmp_path, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / "open_ended.parquet"
    df.to_parquet(path)
    return path


def test_main_writes_new_detections_and_skips_already_done(tmp_path, monkeypatch, capsys):
    imgs = {"img1.jpg": make_b64_image(10, 10), "img2.jpg": make_b64_image(12, 8)}
    data_path = _write_open_ended(tmp_path, {"image_name": list(imgs), "image": list(imgs.values())})
    monkeypatch.setattr(dt, "DATA", str(data_path))

    out_path = tmp_path / "out.json"
    out_path.write_text(json.dumps({"img1.jpg": "OLD"}), encoding="utf-8")

    calls = []

    def fake_openai(image_b64, system, user, model, exemplars):
        calls.append(model)
        return "NEW DETECTION"

    monkeypatch.setattr(rb, "_openai", fake_openai)
    monkeypatch.setattr(
        "sys.argv",
        ["detect_teeth.py", "--model", "test-model", "--effort", "low",
         "--workers", "1", "--n-images", "5", "--out", str(out_path)],
    )

    dt.main()

    assert rb.REASONING_EFFORT == "low"
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["img1.jpg"] == "OLD"          # already done -> untouched, not recomputed
    assert result["img2.jpg"] == "NEW DETECTION"
    assert calls == ["test-model"]              # only img2 was actually sent to the model

    printed = capsys.readouterr().out
    assert "detection: model=test-model effort=low workers=1" in printed
    assert "wrote 2 tooth maps" in printed


def test_main_creates_out_file_from_scratch_when_missing(tmp_path, monkeypatch):
    imgs = {"img1.jpg": make_b64_image(10, 10), "img2.jpg": make_b64_image(12, 8)}
    data_path = _write_open_ended(tmp_path, {"image_name": list(imgs), "image": list(imgs.values())})
    monkeypatch.setattr(dt, "DATA", str(data_path))

    out_path = tmp_path / "new_out.json"  # does not exist yet

    def fake_openai(image_b64, system, user, model, exemplars):
        return f"MAP-FOR-{model}"

    monkeypatch.setattr(rb, "_openai", fake_openai)
    monkeypatch.setattr(
        "sys.argv",
        ["detect_teeth.py", "--out", str(out_path)],
    )

    dt.main()

    assert out_path.exists()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(result) == {"img1.jpg", "img2.jpg"}
    assert result["img1.jpg"] == "MAP-FOR-gpt-5-mini"  # default --model


def test_main_keeps_first_row_per_duplicate_image_name(tmp_path, monkeypatch):
    # image_name "a" appears twice with different pixel data; drop_duplicates
    # keeps the FIRST occurrence, so the second "a" image is never sent.
    rows = {
        "image_name": ["a", "a", "b"],
        "image": [make_b64_image(5, 5), make_b64_image(9, 9), make_b64_image(7, 7)],
    }
    data_path = _write_open_ended(tmp_path, rows)
    monkeypatch.setattr(dt, "DATA", str(data_path))
    out_path = tmp_path / "out.json"

    seen_sizes = []

    def fake_openai(image_b64, system, user, model, exemplars):
        # the prompt embeds the real decoded width/height, so we can tell which
        # of the two "a" images was actually used
        seen_sizes.append(user)
        return "R"

    monkeypatch.setattr(rb, "_openai", fake_openai)
    monkeypatch.setattr("sys.argv", ["detect_teeth.py", "--workers", "1", "--out", str(out_path)])

    dt.main()

    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(result) == {"a", "b"}
    # the "a" prompt must have used the FIRST image's 5x5 size, not 9x9
    assert dt.DET_PROMPT.format(w=5, h=5) in seen_sizes


def test_main_respects_n_images_limit(tmp_path, monkeypatch):
    imgs = {"a.jpg": make_b64_image(5, 5), "b.jpg": make_b64_image(6, 6), "c.jpg": make_b64_image(7, 7)}
    data_path = _write_open_ended(tmp_path, {"image_name": list(imgs), "image": list(imgs.values())})
    monkeypatch.setattr(dt, "DATA", str(data_path))
    out_path = tmp_path / "out.json"

    monkeypatch.setattr(rb, "_openai", lambda *a, **k: "R")
    monkeypatch.setattr(
        "sys.argv",
        ["detect_teeth.py", "--n-images", "2", "--workers", "1", "--out", str(out_path)],
    )

    dt.main()

    result = json.loads(out_path.read_text(encoding="utf-8"))
    # only the first 2 of the 3 images (by original row order) were processed
    assert set(result) == {"a.jpg", "b.jpg"}
