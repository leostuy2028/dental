# Tests for eval_open/judges.py
# This file asks a text-only LLM to grade a model's answer and parses a 0..1 score
# out of its reply, retrying with rising temperature if no number can be parsed.
# No real API calls: the per-provider callers in judges._CALLERS are swapped for
# tiny fake functions. The _call_openai/_call_gemini/_call_claude request builders
# are tested separately below by faking the underlying SDK client classes, so no
# network traffic is ever made even one level deeper.

from types import SimpleNamespace

import openai
import anthropic
from google import genai

import eval_open.judges as judges


# ---------- parse_score ----------

def test_parse_score_plain_number():
    assert judges.parse_score("0.8") == 0.8


def test_parse_score_with_a_label():
    assert judges.parse_score("Correctness: 1.0") == 1.0


def test_parse_score_with_surrounding_whitespace():
    assert judges.parse_score(" .5 ") == 0.5


def test_parse_score_bare_zero_in_a_sentence():
    assert judges.parse_score("the score is 0") == 0.0


def test_parse_score_no_number_gives_none():
    assert judges.parse_score("nonsense") is None


def test_parse_score_takes_the_last_in_range_number():
    assert judges.parse_score("2.0 but 0.7") == 0.7


def test_parse_score_none_and_empty_give_none():
    assert judges.parse_score(None) is None
    assert judges.parse_score("") is None


def test_parse_score_bare_integer_one():
    assert judges.parse_score("1") == 1.0


def test_parse_score_1point5_is_not_rejected_as_out_of_range():
    # CURRENT BEHAVIOR (looks like a bug): the regex only lets "1" be followed by
    # zeros ("1.0", "1.00", ...), so for "1.5" only the "1" part matches and the
    # ".5" is invisible to the parser. An out-of-range reply like "1.5" silently
    # becomes a score of 1.0 instead of being rejected or clamped.
    assert judges.parse_score("1.5") == 1.0


def test_parse_score_last_of_several_candidates_on_one_line():
    assert judges.parse_score("Score: 0.0 and 1.0") == 1.0


# ---------- grade: basic success ----------

def test_grade_unknown_judge_raises():
    try:
        judges.grade("prompt", judge="bogus")
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "bogus" in str(e)


def test_grade_returns_score_and_raw_reply_on_first_try(monkeypatch):
    calls = []

    def fake_caller(prompt, model, temp):
        calls.append((prompt, model, temp))
        return "Score: 0.7"

    monkeypatch.setitem(judges._CALLERS, "gpt-4o", fake_caller)
    score, raw = judges.grade("PROMPT", judge="gpt-4o")
    assert score == 0.7
    assert raw == "Score: 0.7"
    # default model and temperature=0.0 (first entry of TEMP_SCHEDULE) were used
    assert calls == [("PROMPT", "gpt-4o", 0.0)]


def test_grade_uses_the_default_model_for_each_judge(monkeypatch):
    seen = {}

    def fake_caller(prompt, model, temp):
        seen["model"] = model
        return "0.4"

    monkeypatch.setitem(judges._CALLERS, "gemini", fake_caller)
    judges.grade("PROMPT", judge="gemini")
    assert seen["model"] == judges.DEFAULT_MODELS["gemini"]


def test_grade_model_override_is_passed_through(monkeypatch):
    seen = {}

    def fake_caller(prompt, model, temp):
        seen["model"] = model
        return "0.4"

    monkeypatch.setitem(judges._CALLERS, "gemini", fake_caller)
    judges.grade("PROMPT", judge="gemini", model="my-custom-model")
    assert seen["model"] == "my-custom-model"


def test_grade_caller_returning_none_becomes_empty_raw(monkeypatch):
    monkeypatch.setitem(judges._CALLERS, "gemini", lambda prompt, model, temp: None)
    score, raw = judges.grade("PROMPT", judge="gemini")
    assert score == 0.0
    assert raw == ""


# ---------- grade: unparseable replies retry across the temperature schedule ----------

def test_grade_retries_across_all_temperatures_then_gives_up(monkeypatch):
    monkeypatch.setattr(judges.time, "sleep", lambda s: None)
    seen_temps = []

    def fake_caller(prompt, model, temp):
        seen_temps.append(temp)
        return "no number in here at all"

    monkeypatch.setitem(judges._CALLERS, "claude", fake_caller)
    score, raw = judges.grade("PROMPT", judge="claude")
    assert score == 0.0
    assert raw == "no number in here at all"
    assert seen_temps == judges.TEMP_SCHEDULE


# ---------- grade: transient errors are retried with backoff ----------

def test_grade_retries_a_transient_error_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(judges.time, "sleep", lambda s: sleeps.append(s))
    calls = []

    def fake_caller(prompt, model, temp):
        calls.append(temp)
        if len(calls) == 1:
            raise TimeoutError("temporary glitch")
        return "0.6"

    monkeypatch.setitem(judges._CALLERS, "claude", fake_caller)
    score, raw = judges.grade("PROMPT", judge="claude")
    assert score == 0.6
    # both calls were at the same (first) temperature; one backoff sleep in between
    assert calls == [0.0, 0.0]
    assert sleeps == [1]


def test_grade_exhausting_all_transient_retries_moves_to_the_next_temperature(monkeypatch):
    sleeps = []
    monkeypatch.setattr(judges.time, "sleep", lambda s: sleeps.append(s))
    calls = []

    def fake_caller(prompt, model, temp):
        calls.append(temp)
        if temp == 0.0:
            raise TimeoutError("still down")
        return "0.9"

    monkeypatch.setitem(judges._CALLERS, "claude", fake_caller)
    score, raw = judges.grade("PROMPT", judge="claude")
    assert score == 0.9
    # 3 failed attempts at temp 0.0, then 1 successful attempt at temp 0.2
    assert calls == [0.0, 0.0, 0.0, 0.2]
    assert sleeps == [1, 2, 4]


# ---------- grade: non-retryable errors fail fast ----------

def test_grade_non_retryable_error_raises_runtime_error_immediately(monkeypatch):
    monkeypatch.setattr(judges.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("should not sleep")))
    calls = []

    def fake_caller(prompt, model, temp):
        calls.append(temp)
        raise Exception("Authentication failed: bad key")

    monkeypatch.setitem(judges._CALLERS, "gpt-4o", fake_caller)
    try:
        judges.grade("PROMPT", judge="gpt-4o")
        assert False, "expected a RuntimeError"
    except RuntimeError as e:
        assert "non-retryable" in str(e)
        assert "Authentication failed" in str(e)
    # only tried once: no retry, no temperature escalation
    assert calls == [0.0]


def test_grade_non_retryable_check_needs_the_exact_underscored_token(monkeypatch):
    # CURRENT BEHAVIOR (looks like a bug): the fast-fail list checks for the literal
    # substring "invalid_api_key" (an OpenAI error *code*), but a natural-language
    # message like "Invalid API Key provided" (spaces, no underscore) does not
    # contain that substring even after lowercasing, so it is treated as a
    # transient error and retried 15 times (3 attempts x 5 temperatures) instead
    # of failing fast.
    monkeypatch.setattr(judges.time, "sleep", lambda s: None)
    calls = []

    def fake_caller(prompt, model, temp):
        calls.append(temp)
        raise Exception("Invalid API Key provided")

    monkeypatch.setitem(judges._CALLERS, "gpt-4o", fake_caller)
    score, raw = judges.grade("PROMPT", judge="gpt-4o")
    assert score == 0.0
    assert len(calls) == 15


# ---------- grade: delay after a successful score ----------

def test_grade_sleeps_for_delay_seconds_after_a_successful_score(monkeypatch):
    sleeps = []
    monkeypatch.setattr(judges.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setitem(judges._CALLERS, "gemini", lambda prompt, model, temp: "0.3")
    judges.grade("PROMPT", judge="gemini", delay=2.5)
    assert sleeps == [2.5]


def test_grade_does_not_sleep_when_delay_is_zero(monkeypatch):
    sleeps = []
    monkeypatch.setattr(judges.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setitem(judges._CALLERS, "gemini", lambda prompt, model, temp: "0.3")
    judges.grade("PROMPT", judge="gemini", delay=0.0)
    assert sleeps == []


# ---------- the real per-provider request builders ----------
# These call the actual openai/anthropic/google-genai SDK classes, so instead of
# faking judges._CALLERS we fake the SDK client classes themselves and check the
# exact request judges.py builds.

def test_call_openai_builds_the_expected_request(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="0.9"))])

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        build_count = 0

        def __init__(self, **kwargs):
            FakeClient.build_count += 1
            captured["client_kwargs"] = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setattr(judges, "_clients", {})

    result = judges._call_openai("hello prompt", "gpt-4o", 0.3)

    assert result == "0.9"
    assert captured["model"] == "gpt-4o"
    assert captured["messages"] == [{"role": "user", "content": "hello prompt"}]
    assert captured["max_tokens"] == 16
    assert captured["temperature"] == 0.3
    # conftest.py puts a fake key in the environment; judges.py reads it directly
    assert captured["client_kwargs"]["api_key"] == "fake-key-for-tests"
    assert captured["client_kwargs"]["timeout"] == 60.0
    assert captured["client_kwargs"]["max_retries"] == 5

    # the client is cached: a second call must not build a new one
    judges._call_openai("again", "gpt-4o", 0.1)
    assert FakeClient.build_count == 1


def test_call_gemini_builds_the_expected_request(monkeypatch):
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="0.6")

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setattr(judges, "_clients", {})

    result = judges._call_gemini("p", "gemini-2.5-flash", 0.4)

    assert result == "0.6"
    assert captured["model"] == "gemini-2.5-flash"
    assert captured["contents"] == "p"
    cfg = captured["config"]
    assert cfg.temperature == 0.4
    assert cfg.max_output_tokens == 16
    # thinking is explicitly turned off so the 16-token budget is spent on the answer
    assert cfg.thinking_config.thinking_budget == 0
    assert captured["client_kwargs"]["api_key"] == "fake-key-for-tests"


def test_call_claude_builds_the_expected_request(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="0.2")])

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    monkeypatch.setattr(judges, "_clients", {})

    result = judges._call_claude("p", "claude-haiku-4-5-20251001", 0.1)

    assert result == "0.2"
    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["max_tokens"] == 16
    assert captured["temperature"] == 0.1
    # the system prompt forces a bare number so parse_score can find it
    assert "ONLY the correctness score" in captured["system"]
    assert captured["messages"] == [{"role": "user", "content": "p"}]
    assert captured["client_kwargs"]["api_key"] == "fake-key-for-tests"
