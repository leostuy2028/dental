# Tests for utils/make_sample.py
# Builds one self-contained HTML file with a sample of MMOral-Bench questions
# (closed + open, one example per clinical dimension) to share with a
# collaborator. Images are downscaled and inlined as data URIs.
#
# The script reads argv[1]/argv[2] (sample size / output path) as MODULE-LEVEL
# code, so each test imports it fresh with a controlled sys.argv rather than
# importing it once at collection time.

import base64
import io
import sys

import pandas as pd
import pyarrow as pa
from PIL import Image


def fresh_make_sample(monkeypatch, argv=None):
    monkeypatch.setattr(sys, "argv", argv or ["make_sample.py"])
    sys.modules.pop("utils.make_sample", None)
    import utils.make_sample as make_sample
    return make_sample


def tiny_jpeg_b64(color=(0, 255, 0)):
    img = Image.new("RGB", (4, 4), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


IMG_B64 = tiny_jpeg_b64()


# ---------- module-level argv handling ----------

def test_default_n_is_2_and_default_out_is_the_documented_path(monkeypatch):
    ms = fresh_make_sample(monkeypatch, argv=["make_sample.py"])
    assert ms.N == 2
    assert ms.OUT == r"C:\Users\icyic\MMOral_Bench_Sample.html"


def test_argv_overrides_n_and_out(monkeypatch):
    ms = fresh_make_sample(monkeypatch, argv=["make_sample.py", "5", "custom.html"])
    assert ms.N == 5
    assert ms.OUT == "custom.html"


# ---------- tags / esc / render_answer ----------

def test_tags_splits_and_strips_a_comma_list(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    assert ms.tags("Teeth, Patho") == ["Teeth", "Patho"]


def test_tags_of_an_empty_string_is_empty(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    assert ms.tags("") == []
    assert ms.tags("   ") == []


def test_esc_html_escapes_the_value(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    assert ms.esc("<b>hi</b>") == "&lt;b&gt;hi&lt;/b&gt;"


def test_render_answer_turns_headers_and_bold_and_newlines_into_html(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    out = ms.render_answer("# Header\n**bold** text\nline2")
    assert out == "<b>Header</b><br><b>bold</b> text<br>line2"


# ---------- pick ----------

def test_pick_prefers_single_dimension_rows_and_marks_them_used(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    df = pd.DataFrame({"index": [1, 2, 3], "category": ["Teeth", "Patho, Teeth", "Jaw"]})
    used = set()
    chosen = ms.pick(df, "Teeth", used, 5)
    assert chosen == [0, 1]  # single-dim row (0) sorts before the 2-dim row (1)
    assert used == {0, 1}


def test_pick_respects_the_limit_n(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    df = pd.DataFrame({"index": [1, 2], "category": ["Teeth", "Teeth"]})
    chosen = ms.pick(df, "Teeth", set(), 1)
    assert chosen == [0]


def test_pick_skips_rows_already_used(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    df = pd.DataFrame({"index": [1, 2], "category": ["Teeth", "Teeth"]})
    used = {0}
    chosen = ms.pick(df, "Teeth", used, 5)
    assert chosen == [1]


def test_pick_returns_empty_when_no_row_matches(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    df = pd.DataFrame({"index": [1], "category": ["Jaw"]})
    assert ms.pick(df, "Teeth", set(), 5) == []


# ---------- img_data_uri ----------

def test_img_data_uri_returns_a_jpeg_data_uri(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    uri = ms.img_data_uri(IMG_B64, "key1")
    assert uri.startswith("data:image/jpeg;base64,")


def test_img_data_uri_caches_by_key(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    first = ms.img_data_uri(IMG_B64, "key1")
    assert "key1" in ms.IMG_CACHE
    second = ms.img_data_uri(IMG_B64, "key1")
    assert first == second


def test_img_data_uri_returns_empty_string_on_bad_input(monkeypatch, capsys):
    ms = fresh_make_sample(monkeypatch)
    result = ms.img_data_uri("not valid base64!!", "badkey")
    assert result == ""
    assert "[warn]" in capsys.readouterr().out


# ---------- closed_card / open_card ----------

def make_closed_df():
    return pd.DataFrame({
        "index": [1], "file_name": ["a.jpg"], "category": ["Teeth"], "question": ["Q1?"],
        "option1": ["opt1"], "option2": ["opt2"], "option3": ["opt3"], "option4": ["opt4"],
        "answer": ["B"],
    })


def test_closed_card_marks_the_correct_option(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    img_col = pa.array([IMG_B64])
    html = ms.closed_card(make_closed_df(), img_col, 0)
    assert 'class="correct"' in html
    assert "<b>B.</b> opt2" in html
    assert "tick" in html
    # the wrong options are listed without the "correct" class
    assert '<li><b>A.</b> opt1</li>' in html


def test_open_card_shows_the_rendered_reference_answer(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    df = pd.DataFrame({"index": [1], "image_name": ["a.jpg"], "category": ["Teeth"],
                       "question": ["Q1?"], "answer": ["**bold** ref"]})
    img_col = pa.array([IMG_B64])
    html = ms.open_card(df, img_col, 0)
    assert "<b>bold</b> ref" in html
    assert "Q1?" in html


# ---------- section ----------

def test_section_only_emits_a_heading_for_dims_with_a_match(monkeypatch):
    ms = fresh_make_sample(monkeypatch)
    df = pd.DataFrame({"index": [1], "image_name": ["a.jpg"], "category": ["Teeth"],
                       "question": ["Q1?"], "answer": ["ref"]})
    img_col = pa.array([IMG_B64])
    html = ms.section("Title", "blurb", df, img_col, ms.open_card, ["Teeth", "Patho"])
    assert "<h2>Title</h2>" in html
    assert html.count("<h3>") == 1  # only Teeth had a match; Patho was skipped


# ---------- load / main (end to end with tiny parquet files) ----------

def test_load_splits_off_the_image_column(monkeypatch, tmp_path):
    ms = fresh_make_sample(monkeypatch)
    df = pd.DataFrame({"index": [1], "file_name": ["a.jpg"], "category": ["Teeth"],
                       "question": ["Q1?"], "option1": ["1"], "option2": ["2"],
                       "option3": ["3"], "option4": ["4"], "answer": ["A"], "image": [IMG_B64]})
    path = tmp_path / "closed.parquet"
    df.to_parquet(path)

    loaded_df, img_col = ms.load(str(path))

    assert "image" not in loaded_df.columns
    assert list(loaded_df["file_name"]) == ["a.jpg"]
    assert img_col[0].as_py() == IMG_B64


def test_main_writes_an_html_file_covering_both_sections(monkeypatch, tmp_path, capsys):
    ms = fresh_make_sample(monkeypatch, argv=["make_sample.py", "1"])

    closed_df = pd.DataFrame({
        "index": [1], "file_name": ["a.jpg"], "category": ["Teeth"], "question": ["Closed Q?"],
        "option1": ["1"], "option2": ["2"], "option3": ["3"], "option4": ["4"],
        "answer": ["A"], "image": [IMG_B64],
    })
    open_df = pd.DataFrame({
        "index": [1], "image_name": ["b.jpg"], "category": ["Report"], "question": ["Open Q?"],
        "answer": ["Some reference answer"], "image": [IMG_B64],
    })
    closed_path = tmp_path / "closed.parquet"
    open_path = tmp_path / "open.parquet"
    closed_df.to_parquet(closed_path)
    open_df.to_parquet(open_path)

    out_path = tmp_path / "sample.html"
    monkeypatch.setattr(ms, "CLOSED", str(closed_path))
    monkeypatch.setattr(ms, "OPEN", str(open_path))
    monkeypatch.setattr(ms, "OUT", str(out_path))

    ms.main()

    html = out_path.read_text(encoding="utf-8")
    assert "MMOral-Bench" in html
    assert "Closed Q?" in html
    assert "Open Q?" in html
    assert "wrote" in capsys.readouterr().out
