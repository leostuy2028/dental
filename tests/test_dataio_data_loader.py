# Tests for dataio/data_loader.py
# This file has three tiny helpers: load a closed-ended parquet, load an
# open-ended parquet, and decode a base64 image string into a PIL Image.

import base64
from io import BytesIO

import pandas as pd
from PIL import Image

from dataio.data_loader import load_closed, load_open, decode_image


def test_load_closed_reads_a_parquet_file(tmp_path):
    df = pd.DataFrame({"question": ["Q1", "Q2"], "answer": ["A", "B"]})
    path = tmp_path / "closed.parquet"
    df.to_parquet(path)

    out = load_closed(str(path))

    assert list(out.columns) == ["question", "answer"]
    assert out["question"].tolist() == ["Q1", "Q2"]


def test_load_open_reads_a_parquet_file(tmp_path):
    df = pd.DataFrame({"question": ["Q1"], "answer": ["free text"]})
    path = tmp_path / "open.parquet"
    df.to_parquet(path)

    out = load_open(str(path))

    assert out["answer"].tolist() == ["free text"]


def test_decode_image_turns_base64_back_into_the_same_pixels():
    img = Image.new("RGB", (4, 4), (10, 20, 30))
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    out = decode_image(b64)

    assert out.size == (4, 4)
    assert out.convert("RGB").getpixel((0, 0)) == (10, 20, 30)
