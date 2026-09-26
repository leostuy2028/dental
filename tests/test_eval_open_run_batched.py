# Tests for eval_open/run_batched.py
# This is the core "answer all of one image's questions in a single model call"
# script. It defines the prompt builder (build_user), the raw-reply parser
# (parse_numbered), a small retry wrapper (answer_image), the three per-provider
# request builders (_openai/_anthropic/_gemini, reached through the PROVIDERS
# dict), and a big argparse main() that answers + grades a whole dataset.
#
# No test here ever calls a real model. The per-provider tests monkeypatch the
# openai/anthropic/google-genai client classes (all three packages are actually
# installed, so this exercises the REAL message-building code, not a stand-in).

import base64
import json
import os
import time
from io import BytesIO

import pandas as pd
import pytest
from PIL import Image

import eval_open.run_batched as rb


def _tiny_jpeg_b64(color=(200, 50, 50)):
    img = Image.new("RGB", (4, 4), color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------- build_user ----------

def test_build_user_plain_case_is_exact():
    primer = "PRIMER TEXT"
    qs = ["Q1?", "Q2?"]
    out = rb.build_user(primer, qs)
    expected = f"{primer.strip()}{rb.FORMAT_INSTR}\nQuestions:\n1. Q1?\n2. Q2?"
    assert out == expected


def test_build_user_overlay_wins_over_detection_text_when_both_given():
    # the code is `if overlay: ... elif detection_text: ...`, so overlay wins
    # even when a truthy detection_text is ALSO passed
    out = rb.build_user("primer", ["Q1?"], detection_text="SOME CHART", overlay=True)
    assert rb.OVERLAY_NOTE in out
    assert "SOME CHART" not in out


def test_build_user_detection_text_used_when_no_overlay():
    out = rb.build_user("primer", ["Q1?"], detection_text="SOME CHART")
    assert "SOME CHART" in out
    assert rb.OVERLAY_NOTE not in out
    assert "TOOTH CHART for THIS X-ray" in out


def test_build_user_no_detection_or_overlay_has_neither_note():
    out = rb.build_user("primer", ["Q1?"])
    assert rb.OVERLAY_NOTE not in out
    assert "TOOTH CHART" not in out


def test_build_user_concise_note_appended():
    out = rb.build_user("primer", ["Q1?"], concise=True)
    assert rb.CONCISE_NOTE in out


def test_build_user_precision_note_appended():
    out = rb.build_user("primer", ["Q1?"], precision=True)
    assert rb.PRECISION_NOTE in out


def test_build_user_concise_comes_before_precision_and_questions_come_last():
    out = rb.build_user("primer", ["Q1?"], concise=True, precision=True)
    assert out.index(rb.CONCISE_NOTE) < out.index(rb.PRECISION_NOTE)
    assert out.rstrip().endswith("Questions:\n1. Q1?")


# ---------- load_exemplars ----------

def test_load_exemplars_reads_manifest_and_b64_encodes_the_image(tmp_path):
    img_bytes = _tiny_jpeg_b64()
    img_path = tmp_path / "ex1.jpg"
    img_path.write_bytes(base64.b64decode(img_bytes))
    manifest = {"exemplars": [{"image": "ex1.jpg", "caption": "cap one"}]}
    man_path = tmp_path / "manifest.json"
    man_path.write_text(json.dumps(manifest), encoding="utf-8")

    out = rb.load_exemplars(str(man_path))

    assert out == [(img_bytes, "cap one")]


def test_load_exemplars_falls_back_to_the_basename_when_the_relative_path_is_missing(tmp_path):
    img_bytes = _tiny_jpeg_b64()
    (tmp_path / "ex1.jpg").write_bytes(base64.b64decode(img_bytes))
    # manifest points at "sub/ex1.jpg", which does not exist, but "ex1.jpg" (its
    # basename) does exist right next to the manifest -> the fallback finds it
    manifest = {"exemplars": [{"image": "sub/ex1.jpg", "caption": "cap two"}]}
    man_path = tmp_path / "manifest.json"
    man_path.write_text(json.dumps(manifest), encoding="utf-8")

    out = rb.load_exemplars(str(man_path))

    assert out == [(img_bytes, "cap two")]


def test_load_exemplars_raises_file_not_found_when_neither_path_exists(tmp_path):
    manifest = {"exemplars": [{"image": "missing.jpg", "caption": "cap"}]}
    man_path = tmp_path / "manifest.json"
    man_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        rb.load_exemplars(str(man_path))


# ---------- parse_numbered ----------

def test_parse_numbered_well_formed():
    assert rb.parse_numbered("1. foo\n2. bar", 2) == ["foo", "bar"]


def test_parse_numbered_extra_whitespace():
    assert rb.parse_numbered("  1.   foo  \n  2.   bar  ", 2) == ["foo", "bar"]


def test_parse_numbered_missing_number_becomes_empty_string():
    assert rb.parse_numbered("1. foo", 2) == ["foo", ""]


def test_parse_numbered_out_of_order_numbers_are_reassembled_by_index():
    assert rb.parse_numbered("2. bar\n1. foo", 2) == ["foo", "bar"]


def test_parse_numbered_accepts_a_closing_paren_instead_of_a_period():
    assert rb.parse_numbered("1) foo\n2) bar", 2) == ["foo", "bar"]


def test_parse_numbered_n_larger_than_whats_present():
    assert rb.parse_numbered("1. foo\n3. baz", 3) == ["foo", "", "baz"]


def test_parse_numbered_n_smaller_than_whats_present_drops_the_extra():
    assert rb.parse_numbered("1. foo\n2. bar\n3. baz", 2) == ["foo", "bar"]


def test_parse_numbered_totally_unparseable_text_gives_all_blanks():
    assert rb.parse_numbered("nonsense text here", 2) == ["", ""]


def test_parse_numbered_empty_string_gives_all_blanks():
    assert rb.parse_numbered("", 2) == ["", ""]


# ---------- answer_image ----------

def test_answer_image_succeeds_on_the_first_well_formed_reply(monkeypatch):
    def fake(image_b64, system, user, model, exemplars=None, **kw):
        return "1. foo\n2. bar"

    monkeypatch.setitem(rb.PROVIDERS, "openai", fake)
    ans, raw = rb.answer_image("img", ["Q1", "Q2"], "primer", "sys", "openai", "model")
    assert ans == ["foo", "bar"]
    assert raw == "1. foo\n2. bar"


def test_answer_image_retries_on_a_garbled_reply_then_succeeds(monkeypatch):
    calls = []

    def fake(image_b64, system, user, model, exemplars=None, **kw):
        calls.append(1)
        if len(calls) == 1:
            return "garbled, no numbers here"
        return "1. foo\n2. bar"

    monkeypatch.setitem(rb.PROVIDERS, "openai", fake)
    ans, raw = rb.answer_image("img", ["Q1", "Q2"], "primer", "sys", "openai", "model", retries=3)
    assert len(calls) == 2
    assert ans == ["foo", "bar"]


def test_answer_image_early_return_threshold_one_question_needs_only_one_answer(monkeypatch):
    # len(questions)=1 -> max(1, 0) = 1, so a single non-empty answer is "nearly all"
    def fake(image_b64, system, user, model, exemplars=None, **kw):
        return "1. only answer"

    monkeypatch.setitem(rb.PROVIDERS, "openai", fake)
    ans, raw = rb.answer_image("img", ["Q1"], "primer", "sys", "openai", "model")
    assert ans == ["only answer"]


def test_answer_image_three_questions_needs_two_parsed_to_accept_first_try(monkeypatch):
    # len(questions)=3 -> max(1, 2) = 2; a reply with only 1/3 parsed must retry
    calls = []

    def fake(image_b64, system, user, model, exemplars=None, **kw):
        calls.append(1)
        if len(calls) == 1:
            return "1. only one"
        return "1. a\n2. b\n3. c"

    monkeypatch.setitem(rb.PROVIDERS, "openai", fake)
    ans, raw = rb.answer_image("img", ["Q1", "Q2", "Q3"], "primer", "sys", "openai", "model", retries=3)
    assert len(calls) == 2
    assert ans == ["a", "b", "c"]


def test_answer_image_every_attempt_raising_falls_back_to_empty_strings(monkeypatch):
    # 'raw' is only ever undefined if EVERY attempt raised before assigning it;
    # the final line's `raw if 'raw' in dir() else ""` then uses ""
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def fake(image_b64, system, user, model, exemplars=None, **kw):
        raise RuntimeError("boom")

    monkeypatch.setitem(rb.PROVIDERS, "openai", fake)
    ans, raw = rb.answer_image("img", ["Q1", "Q2"], "primer", "sys", "openai", "model", retries=2)
    assert ans == ["", ""]
    assert raw == ""


def test_answer_image_gives_up_after_retries_exhausted_returns_last_raw(monkeypatch):
    # never raises, but never parses enough either -> after `retries` attempts it
    # returns parse_numbered() on the LAST raw text seen, even though unparsed
    def fake(image_b64, system, user, model, exemplars=None, **kw):
        return "still garbled"

    monkeypatch.setitem(rb.PROVIDERS, "openai", fake)
    ans, raw = rb.answer_image("img", ["Q1", "Q2"], "primer", "sys", "openai", "model", retries=2)
    assert ans == ["", ""]
    assert raw == "still garbled"


def test_answer_image_passes_no_image_kw_only_for_gemini(monkeypatch):
    seen_kw = {}

    def fake(image_b64, system, user, model, exemplars=None, **kw):
        seen_kw.update(kw)
        return "1. a"

    monkeypatch.setitem(rb.PROVIDERS, "gemini", fake)
    rb.answer_image("img", ["Q1"], "primer", "sys", "gemini", "model", no_image=True)
    assert seen_kw == {"no_image": True}

    seen_kw.clear()
    monkeypatch.setitem(rb.PROVIDERS, "openai", fake)
    rb.answer_image("img", ["Q1"], "primer", "sys", "openai", "model", no_image=True)
    assert seen_kw == {}  # no_image is only forwarded for the gemini provider


# ---------- _openai / _anthropic / _gemini: real request-building coverage ----------
# openai, anthropic and google-genai are all real installed dependencies here, and
# run_batched imports each of them with a LOCAL `import x` inside the function body,
# so patching the client class on the real module (not on run_batched) takes effect.

def test_openai_builds_plain_chat_request_and_returns_the_reply_text(monkeypatch):
    import openai
    captured = {}

    class FakeCompletions:
        def create(self, **kw):
            captured["kw"] = kw
            class Msg:
                content = "the reply"
            class Choice:
                message = Msg()
            class Resp:
                choices = [Choice()]
            return Resp()

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(openai, "OpenAI", lambda **kw: FakeClient())
    rb._clients.clear()

    out = rb._openai("IMG64", "sys text", "user text", "gpt-4o")

    assert out == "the reply"
    kw = captured["kw"]
    assert kw["model"] == "gpt-4o"
    assert kw["max_tokens"] == 4096 and kw["temperature"] == 0.0
    assert "reasoning_effort" not in kw and "max_completion_tokens" not in kw
    assert kw["messages"][0] == {"role": "system", "content": "sys text"}
    content = kw["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "user text"}
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,IMG64"
    assert content[1]["image_url"]["detail"] == "high"


def test_openai_empty_message_content_becomes_empty_string(monkeypatch):
    import openai

    class FakeCompletions:
        def create(self, **kw):
            class Msg:
                content = None
            class Choice:
                message = Msg()
            class Resp:
                choices = [Choice()]
            return Resp()

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(openai, "OpenAI", lambda **kw: FakeClient())
    rb._clients.clear()

    assert rb._openai("IMG64", "sys", "user", "gpt-4o") == ""


def test_openai_reasoning_model_gpt5_uses_max_completion_tokens_and_effort(monkeypatch):
    import openai
    captured = {}

    class FakeCompletions:
        def create(self, **kw):
            captured["kw"] = kw
            class Msg:
                content = "ok"
            class Choice:
                message = Msg()
            class Resp:
                choices = [Choice()]
            return Resp()

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(openai, "OpenAI", lambda **kw: FakeClient())
    rb._clients.clear()
    monkeypatch.setattr(rb, "REASONING_EFFORT", "low")

    rb._openai("IMG64", "sys", "user", "gpt-5-thing")

    kw = captured["kw"]
    assert kw["max_completion_tokens"] == rb._MAXTOK["low"]
    assert kw["reasoning_effort"] == "low"
    assert "max_tokens" not in kw and "temperature" not in kw


def test_openai_reasoning_model_o_prefix_regex_matches_o_then_digit(monkeypatch):
    import openai
    captured = {}

    class FakeCompletions:
        def create(self, **kw):
            captured["kw"] = kw
            class Msg:
                content = "ok"
            class Choice:
                message = Msg()
            class Resp:
                choices = [Choice()]
            return Resp()

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(openai, "OpenAI", lambda **kw: FakeClient())
    rb._clients.clear()

    rb._openai("IMG64", "sys", "user", "o3")
    assert "reasoning_effort" in captured["kw"]

    # "omni-model" starts with "o" but not "o<digit>", so it is NOT a reasoning model
    rb._openai("IMG64", "sys", "user", "omni-model")
    assert "reasoning_effort" not in captured["kw"]


def test_openai_exemplars_are_inserted_before_the_users_own_content(monkeypatch):
    import openai
    captured = {}

    class FakeCompletions:
        def create(self, **kw):
            captured["kw"] = kw
            class Msg:
                content = "ok"
            class Choice:
                message = Msg()
            class Resp:
                choices = [Choice()]
            return Resp()

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(openai, "OpenAI", lambda **kw: FakeClient())
    rb._clients.clear()

    rb._openai("IMG64", "sys", "user text", "gpt-4o", exemplars=[("EXB64", "cap1")])

    content = captured["kw"]["messages"][1]["content"]
    texts = [c["text"] for c in content if c["type"] == "text"]
    assert texts == [rb.EXEMPLAR_INTRO, "cap1", rb.EXEMPLAR_OUTRO, "user text"]
    image_urls = [c["image_url"]["url"] for c in content if c["type"] == "image_url"]
    assert image_urls == ["data:image/jpeg;base64,EXB64", "data:image/jpeg;base64,IMG64"]


def test_anthropic_builds_request_and_joins_only_text_blocks(monkeypatch):
    import anthropic
    captured = {}

    class Block:
        def __init__(self, type_, text=None):
            self.type = type_
            self.text = text

    class FakeMessages:
        def create(self, **kw):
            captured["kw"] = kw
            class Resp:
                content = [Block("text", "hello "), Block("image"), Block("text", "world")]
            return Resp()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: FakeClient())
    rb._clients.clear()

    out = rb._anthropic("IMG64", "sys text", "user text", "claude-x")

    assert out == "hello world"  # only the two 'text' blocks are joined, not 'image'
    kw = captured["kw"]
    assert kw["system"] == "sys text" and kw["max_tokens"] == 4096 and kw["temperature"] == 0.0
    content = kw["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "user text"}
    assert content[1]["source"]["data"] == "IMG64"
    assert content[1]["source"]["media_type"] == "image/jpeg"


def test_anthropic_with_exemplars_orders_intro_pairs_then_outro_then_user(monkeypatch):
    import anthropic
    captured = {}

    class Block:
        def __init__(self, type_, text=None):
            self.type = type_
            self.text = text

    class FakeMessages:
        def create(self, **kw):
            captured["kw"] = kw
            class Resp:
                content = [Block("text", "ok")]
            return Resp()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: FakeClient())
    rb._clients.clear()

    rb._anthropic("IMG64", "sys", "user text", "claude-x", exemplars=[("EXB64", "cap")])

    content = captured["kw"]["messages"][0]["content"]
    kinds_and_text = [(c["type"], c.get("text")) for c in content]
    assert kinds_and_text == [
        ("text", rb.EXEMPLAR_INTRO), ("image", None), ("text", "cap"),
        ("text", rb.EXEMPLAR_OUTRO), ("text", "user text"), ("image", None),
    ]


def test_gemini_builds_contents_and_thinking_config(monkeypatch):
    from google import genai
    captured = {}

    class FakeModels:
        def generate_content(self, **kw):
            captured["kw"] = kw
            class Resp:
                text = "gemini reply"
            return Resp()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(genai, "Client", lambda **kw: FakeClient())
    rb._clients.clear()
    monkeypatch.setattr(rb, "THINKING_BUDGET", 500)
    img_b64 = _tiny_jpeg_b64()

    out = rb._gemini(img_b64, "sys text", "user text", "gemini-x")

    assert out == "gemini reply"
    kw = captured["kw"]
    assert kw["model"] == "gemini-x"
    contents = kw["contents"]
    assert contents[0] == "sys text\n\n"
    assert contents[1] == "user text"
    assert contents[2].size == (4, 4)  # the decoded PIL image
    cfg = kw["config"]
    assert cfg.max_output_tokens == 4096 + 500
    assert cfg.thinking_config.thinking_budget == 500
    assert cfg.temperature == 0.0


def test_gemini_thinking_budget_zero_does_not_add_to_max_output_tokens(monkeypatch):
    from google import genai

    class FakeModels:
        def generate_content(self, **kw):
            class Resp:
                text = "ok"
            return Resp()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(genai, "Client", lambda **kw: FakeClient())
    rb._clients.clear()
    monkeypatch.setattr(rb, "THINKING_BUDGET", 0)
    captured = {}
    orig_create = FakeModels.generate_content
    def spy(self, **kw):
        captured["kw"] = kw
        return orig_create(self, **kw)
    monkeypatch.setattr(FakeModels, "generate_content", spy)

    rb._gemini(_tiny_jpeg_b64(), "sys", "user", "gemini-x")

    assert captured["kw"]["config"].max_output_tokens == 4096


def test_gemini_with_exemplars_decodes_each_exemplar_image(monkeypatch):
    from google import genai
    captured = {}

    class FakeModels:
        def generate_content(self, **kw):
            captured["kw"] = kw
            class Resp:
                text = "ok"
            return Resp()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(genai, "Client", lambda **kw: FakeClient())
    rb._clients.clear()
    ex_b64 = _tiny_jpeg_b64()

    rb._gemini(_tiny_jpeg_b64(), "sys", "user text", "gemini-x", exemplars=[(ex_b64, "cap1")])

    contents = captured["kw"]["contents"]
    assert contents[0] == "sys\n\n"
    assert contents[1] == rb.EXEMPLAR_INTRO
    assert contents[2].size == (4, 4)  # decoded exemplar image
    assert contents[3] == "cap1"
    assert contents[4] == rb.EXEMPLAR_OUTRO
    assert contents[5] == "user text"
    assert contents[6].size == (4, 4)  # the real image


def test_gemini_no_image_sends_a_note_instead_of_the_real_image(monkeypatch):
    from google import genai
    captured = {}

    class FakeModels:
        def generate_content(self, **kw):
            captured["kw"] = kw
            class Resp:
                text = "ok"
            return Resp()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(genai, "Client", lambda **kw: FakeClient())
    rb._clients.clear()

    rb._gemini(_tiny_jpeg_b64(), "sys", "user text", "gemini-x", no_image=True)

    contents = captured["kw"]["contents"]
    assert contents == ["sys\n\n", "user text", "\n(No radiograph is provided. Answer from the questions alone.)\n"]


def test_gemini_empty_text_response_becomes_empty_string(monkeypatch):
    from google import genai

    class FakeModels:
        def generate_content(self, **kw):
            class Resp:
                text = None
            return Resp()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(genai, "Client", lambda **kw: FakeClient())
    rb._clients.clear()

    assert rb._gemini(_tiny_jpeg_b64(), "sys", "user", "gemini-x") == ""


# ---------- main() ----------

def _write_dataset(path, rows):
    pd.DataFrame(rows).to_parquet(path)


def _make_two_image_two_question_rows():
    img = "PARQUET_IMG_PLACEHOLDER"  # never decoded: providers are faked away
    return {
        "index": [0, 1, 2, 3],
        "image_name": ["img1.jpg", "img1.jpg", "img2.jpg", "img2.jpg"],
        "image": [img, img, img, img],
        "question": ["Q1 for img1", "Q2 for img1", "Q1 for img2", "Q2 for img2"],
        "answer": ["gt1", "gt2", "gt3", "gt4"],
    }


def _fake_two_answer_provider(calls=None):
    def fake(image_b64, system, user, model, exemplars=None, **kw):
        if calls is not None:
            calls.append({"image_b64": image_b64, "system": system, "user": user,
                          "model": model, "exemplars": exemplars, **kw})
        return "1. ans-one\n2. ans-two"
    return fake


def _fake_grade(score=0.5):
    def fake(prompt, judge="gpt-4o", model=None):
        return score, "raw-judge-text"
    return fake


def _setup_main(tmp_path, monkeypatch, rows=None):
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "open_ended.parquet"
    _write_dataset(data_path, rows or _make_two_image_two_question_rows())
    primer_path = tmp_path / "primer.txt"
    primer_path.write_text("PRIMER", encoding="utf-8")
    monkeypatch.setattr(rb, "DATA", str(data_path))
    monkeypatch.setattr(rb, "PRIMER", str(primer_path))


def test_main_plain_run_writes_answers_and_scores_csv(tmp_path, monkeypatch, capsys):
    _setup_main(tmp_path, monkeypatch)
    monkeypatch.setitem(rb.PROVIDERS, "openai", _fake_two_answer_provider())
    monkeypatch.setattr(rb, "grade", _fake_grade(0.75))
    monkeypatch.setattr(
        "sys.argv",
        ["run_batched.py", "--model", "gpt-4o", "--provider", "openai", "--tag", "t1"],
    )

    rb.main()

    answers = pd.read_csv("results/open/batched_t1_answers.csv")
    scores = pd.read_csv("results/open/batched_t1_scores.csv")
    assert len(answers) == 4
    assert sorted(answers["image_name"].unique()) == ["img1.jpg", "img2.jpg"]
    assert set(answers["answer"]) == {"ans-one", "ans-two"}
    assert len(scores) == 4
    assert (scores["score"] == 0.75).all()


def test_main_resume_skips_images_already_in_the_answers_csv(tmp_path, monkeypatch):
    _setup_main(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setitem(rb.PROVIDERS, "openai", _fake_two_answer_provider(calls))
    monkeypatch.setattr(rb, "grade", _fake_grade(1.0))

    os.makedirs("results/open", exist_ok=True)
    pd.DataFrame({
        "index": [0, 1], "image_name": ["img1.jpg", "img1.jpg"],
        "question": ["Q1 for img1", "Q2 for img1"], "gt": ["gt1", "gt2"],
        "answer": ["already-answered-1", "already-answered-2"],
    }).to_csv("results/open/batched_t2_answers.csv", index=False)

    monkeypatch.setattr(
        "sys.argv",
        ["run_batched.py", "--model", "gpt-4o", "--provider", "openai", "--tag", "t2"],
    )
    rb.main()

    # only img2 should have gone through the (fake) provider
    assert {c["model"] for c in calls} == {"gpt-4o"}
    assert len(calls) == 1
    answers = pd.read_csv("results/open/batched_t2_answers.csv")
    assert len(answers) == 4
    assert "already-answered-1" in answers["answer"].tolist()
    assert "ans-one" in answers["answer"].tolist()


def test_main_images_file_restricts_to_the_listed_images(tmp_path, monkeypatch):
    _setup_main(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setitem(rb.PROVIDERS, "openai", _fake_two_answer_provider(calls))
    monkeypatch.setattr(rb, "grade", _fake_grade(0.0))

    images_file = tmp_path / "keep.json"
    images_file.write_text(json.dumps(["img2.jpg"]), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["run_batched.py", "--model", "gpt-4o", "--provider", "openai",
         "--tag", "t3", "--images-file", str(images_file)],
    )
    rb.main()

    answers = pd.read_csv("results/open/batched_t3_answers.csv")
    assert sorted(answers["image_name"].unique()) == ["img2.jpg"]
    assert len(calls) == 1


def test_main_detections_are_injected_into_the_prompt(tmp_path, monkeypatch):
    _setup_main(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setitem(rb.PROVIDERS, "openai", _fake_two_answer_provider(calls))
    monkeypatch.setattr(rb, "grade", _fake_grade(0.0))

    detections_file = tmp_path / "detections.json"
    detections_file.write_text(json.dumps({"img1.jpg": "TOOTH-MAP-FOR-IMG1"}), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["run_batched.py", "--model", "gpt-4o", "--provider", "openai",
         "--tag", "t4", "--detections", str(detections_file)],
    )
    rb.main()

    by_image = {}
    for c in calls:
        by_image.setdefault(c["user"], c)
    img1_call = next(c for c in calls if "Q1 for img1" in c["user"])
    img2_call = next(c for c in calls if "Q1 for img2" in c["user"])
    assert "TOOTH-MAP-FOR-IMG1" in img1_call["user"]
    assert "TOOTH-MAP-FOR-IMG1" not in img2_call["user"]  # only img1 has a detection


def test_main_overlay_dir_uses_the_overlay_file_not_the_parquet_image(tmp_path, monkeypatch):
    _setup_main(tmp_path, monkeypatch)
    overlay_dir = tmp_path / "overlays"
    overlay_dir.mkdir()
    real_bytes = base64.b64decode(_tiny_jpeg_b64())
    (overlay_dir / "img1.jpg").write_bytes(real_bytes)
    (overlay_dir / "img2.jpg").write_bytes(real_bytes)

    calls = []
    monkeypatch.setitem(rb.PROVIDERS, "openai", _fake_two_answer_provider(calls))
    monkeypatch.setattr(rb, "grade", _fake_grade(0.0))

    monkeypatch.setattr(
        "sys.argv",
        ["run_batched.py", "--model", "gpt-4o", "--provider", "openai",
         "--tag", "t5", "--overlay-dir", str(overlay_dir)],
    )
    rb.main()

    assert all(c["image_b64"] == base64.b64encode(real_bytes).decode() for c in calls)
    assert all(rb.OVERLAY_NOTE in c["user"] for c in calls)


def test_main_exemplars_manifest_is_loaded_and_passed_through(tmp_path, monkeypatch, capsys):
    _setup_main(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setitem(rb.PROVIDERS, "openai", _fake_two_answer_provider(calls))
    monkeypatch.setattr(rb, "grade", _fake_grade(0.0))

    img_bytes = _tiny_jpeg_b64()
    (tmp_path / "ex1.jpg").write_bytes(base64.b64decode(img_bytes))
    manifest = tmp_path / "exemplars.json"
    manifest.write_text(json.dumps({"exemplars": [{"image": "ex1.jpg", "caption": "cap1"}]}),
                        encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["run_batched.py", "--model", "gpt-4o", "--provider", "openai",
         "--tag", "t7", "--exemplars", str(manifest)],
    )
    rb.main()

    out = capsys.readouterr().out
    assert "loaded 1 visual exemplars" in out
    assert all(c["exemplars"] == [(img_bytes, "cap1")] for c in calls)


def test_main_images_file_as_plain_lines_when_not_valid_json(tmp_path, monkeypatch):
    _setup_main(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setitem(rb.PROVIDERS, "openai", _fake_two_answer_provider(calls))
    monkeypatch.setattr(rb, "grade", _fake_grade(0.0))

    images_file = tmp_path / "keep.txt"
    images_file.write_text("img1.jpg\n\nimg2.jpg\n", encoding="utf-8")  # not JSON, one-per-line

    monkeypatch.setattr(
        "sys.argv",
        ["run_batched.py", "--model", "gpt-4o", "--provider", "openai",
         "--tag", "t8", "--images-file", str(images_file)],
    )
    rb.main()

    answers = pd.read_csv("results/open/batched_t8_answers.csv")
    assert sorted(answers["image_name"].unique()) == ["img1.jpg", "img2.jpg"]


def test_main_score_resume_skips_already_graded_indices(tmp_path, monkeypatch):
    _setup_main(tmp_path, monkeypatch)
    monkeypatch.setitem(rb.PROVIDERS, "openai", _fake_two_answer_provider())
    graded = []

    def fake_grade(prompt, judge="gpt-4o", model=None):
        graded.append(prompt)
        return 0.25, "raw"

    monkeypatch.setattr(rb, "grade", fake_grade)

    os.makedirs("results/open", exist_ok=True)
    pd.DataFrame({"index": [0, 1], "score": [1.0, 1.0]}).to_csv(
        "results/open/batched_t9_scores.csv", index=False)

    monkeypatch.setattr(
        "sys.argv",
        ["run_batched.py", "--model", "gpt-4o", "--provider", "openai", "--tag", "t9"],
    )
    rb.main()

    scores = pd.read_csv("results/open/batched_t9_scores.csv")
    assert len(scores) == 4  # 2 pre-graded + 2 newly graded
    # only indices 2 and 3 (img2's questions) needed a real grade call
    assert len(graded) == 2
    assert sorted(scores[scores["index"].isin([0, 1])]["score"].tolist()) == [1.0, 1.0]


def test_main_no_image_with_gemini_provider_sets_the_no_image_kw(tmp_path, monkeypatch):
    _setup_main(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setitem(rb.PROVIDERS, "gemini", _fake_two_answer_provider(calls))
    monkeypatch.setattr(rb, "grade", _fake_grade(0.0))

    monkeypatch.setattr(
        "sys.argv",
        ["run_batched.py", "--model", "gemini-x", "--provider", "gemini",
         "--tag", "t6", "--no-image"],
    )
    rb.main()

    assert len(calls) == 2
    assert all(c.get("no_image") is True for c in calls)
