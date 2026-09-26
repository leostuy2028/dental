# Tests for eval_open/answerer.py
# This file asks the model under test to answer an open-ended question about an
# X-ray, using the verbatim benchmark prompt. Its real output is what gets graded,
# so nothing here is ever hand-authored. No real API calls: the openai/google-genai
# SDK client classes are swapped for tiny fakes that record what they were called
# with and return canned text.

from types import SimpleNamespace

import openai
from google import genai

import eval_open.answerer as answerer


def fake_openai_client(responses=None, error_first_n=0, error=RuntimeError("down")):
    """A stand-in for the real openai.OpenAI() client's .chat.completions.create()."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) <= error_first_n:
            raise error
        text = responses[len(calls) - 1 - error_first_n] if responses else "an answer"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.calls = calls
    return client


def fake_gemini_client(responses=None, error_first_n=0, error=RuntimeError("down")):
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        if len(calls) <= error_first_n:
            raise error
        text = responses[len(calls) - 1 - error_first_n] if responses else "an answer"
        return SimpleNamespace(text=text)

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    client.calls = calls
    return client


# ---------- _get_openai / _get_client: lazy singleton clients ----------

def test_get_openai_builds_the_client_once_and_reuses_it(monkeypatch):
    captured = {}

    class FakeClient:
        build_count = 0

        def __init__(self, **kwargs):
            FakeClient.build_count += 1
            captured["kwargs"] = kwargs

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setattr(answerer, "_openai_client", None)

    c1 = answerer._get_openai()
    c2 = answerer._get_openai()

    assert c1 is c2
    assert FakeClient.build_count == 1
    # conftest.py puts a fake key in the environment before answerer.py reads it
    assert captured["kwargs"]["api_key"] == "fake-key-for-tests"
    assert captured["kwargs"]["timeout"] == 90.0
    assert captured["kwargs"]["max_retries"] == 5


def test_get_client_builds_the_gemini_client_once_and_reuses_it(monkeypatch):
    captured = {}

    class FakeClient:
        build_count = 0

        def __init__(self, **kwargs):
            FakeClient.build_count += 1
            captured["kwargs"] = kwargs

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setattr(answerer, "_client", None)

    c1 = answerer._get_client()
    c2 = answerer._get_client()

    assert c1 is c2
    assert FakeClient.build_count == 1
    assert captured["kwargs"]["api_key"] == "fake-key-for-tests"


# ---------- answer_question_openai ----------

def test_answer_question_openai_returns_stripped_text_on_first_try(monkeypatch):
    client = fake_openai_client(responses=["  the answer  "])
    monkeypatch.setattr(answerer, "_get_openai", lambda: client)

    result = answerer.answer_question_openai("b64img", "What is this?", model="gpt-4o")

    assert result == "the answer"
    call = client.calls[0]
    assert call["model"] == "gpt-4o"
    assert call["max_tokens"] == 1024
    assert call["temperature"] == 0.0
    content = call["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": answerer.INFERENCE_PROMPT.format(q="What is this?")}
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,b64img"


def test_answer_question_openai_retries_a_transient_error_then_succeeds(monkeypatch):
    monkeypatch.setattr(answerer.time, "sleep", lambda s: None)
    client = fake_openai_client(responses=["ok answer"], error_first_n=1)
    monkeypatch.setattr(answerer, "_get_openai", lambda: client)

    result = answerer.answer_question_openai("b", "q", retries=3)

    assert result == "ok answer"
    assert len(client.calls) == 2


def test_answer_question_openai_gives_up_after_all_retries(monkeypatch):
    monkeypatch.setattr(answerer.time, "sleep", lambda s: None)
    client = fake_openai_client(error_first_n=99)
    monkeypatch.setattr(answerer, "_get_openai", lambda: client)

    result = answerer.answer_question_openai("b", "q", retries=2)

    assert result == ""
    assert len(client.calls) == 2


# ---------- answer_openai_custom ----------

def test_answer_openai_custom_without_a_system_message(monkeypatch):
    client = fake_openai_client(responses=["custom answer"])
    monkeypatch.setattr(answerer, "_get_openai", lambda: client)

    result = answerer.answer_openai_custom("b64", None, "user text", max_tokens=500, detail="low")

    assert result == "custom answer"
    call = client.calls[0]
    assert call["max_tokens"] == 500
    assert call["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "user text"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,b64", "detail": "low"}},
    ]}]


def test_answer_openai_custom_prepends_a_system_message_when_given(monkeypatch):
    client = fake_openai_client(responses=["custom answer"])
    monkeypatch.setattr(answerer, "_get_openai", lambda: client)

    answerer.answer_openai_custom("b64", "SYSTEM PROMPT", "user text")

    call = client.calls[0]
    assert call["messages"][0] == {"role": "system", "content": "SYSTEM PROMPT"}
    # default detail is 'high'
    assert call["messages"][1]["content"][1]["image_url"]["detail"] == "high"


def test_answer_openai_custom_gives_up_after_all_retries(monkeypatch):
    monkeypatch.setattr(answerer.time, "sleep", lambda s: None)
    client = fake_openai_client(error_first_n=99)
    monkeypatch.setattr(answerer, "_get_openai", lambda: client)

    result = answerer.answer_openai_custom("b", "sys", "q", retries=2)

    assert result == ""
    assert len(client.calls) == 2


# ---------- answer_question (gemini) ----------

def test_answer_question_returns_stripped_text(monkeypatch):
    client = fake_gemini_client(responses=["  gemini answer  "])
    monkeypatch.setattr(answerer, "_get_client", lambda: client)

    result = answerer.answer_question("FAKE_IMAGE", "What is this?", model="gemini-3.5-flash",
                                      thinking_budget=0, max_output_tokens=999)

    assert result == "gemini answer"
    call = client.calls[0]
    assert call["model"] == "gemini-3.5-flash"
    assert call["contents"] == [answerer.INFERENCE_PROMPT.format(q="What is this?"), "FAKE_IMAGE"]
    cfg = call["config"]
    assert cfg.temperature == 0.0
    assert cfg.max_output_tokens == 999
    assert cfg.thinking_config.thinking_budget == 0


def test_answer_question_treats_a_none_reply_as_empty_string(monkeypatch):
    client = fake_gemini_client(responses=[None])
    monkeypatch.setattr(answerer, "_get_client", lambda: client)

    result = answerer.answer_question("IMG", "q")

    assert result == ""


def test_answer_question_retries_a_transient_error_then_succeeds(monkeypatch):
    monkeypatch.setattr(answerer.time, "sleep", lambda s: None)
    client = fake_gemini_client(responses=["ok"], error_first_n=1)
    monkeypatch.setattr(answerer, "_get_client", lambda: client)

    result = answerer.answer_question("IMG", "q", retries=3)

    assert result == "ok"
    assert len(client.calls) == 2


def test_answer_question_gives_up_after_all_retries(monkeypatch):
    monkeypatch.setattr(answerer.time, "sleep", lambda s: None)
    client = fake_gemini_client(error_first_n=99)
    monkeypatch.setattr(answerer, "_get_client", lambda: client)

    result = answerer.answer_question("IMG", "q", retries=2)

    assert result == ""
    assert len(client.calls) == 2


def test_answer_question_dynamic_thinking_budget_is_passed_through(monkeypatch):
    client = fake_gemini_client(responses=["ok"])
    monkeypatch.setattr(answerer, "_get_client", lambda: client)

    answerer.answer_question("IMG", "q", thinking_budget=-1, max_output_tokens=4096)

    cfg = client.calls[0]["config"]
    assert cfg.thinking_config.thinking_budget == -1
    assert cfg.max_output_tokens == 4096
