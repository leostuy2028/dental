# Tests for clients/claude_client.py
# Wraps the Anthropic API: builds the right request for each thinking mode,
# retries on error, and raises APICallFailed once retries run out (so the
# harness can skip the item instead of recording an error string as an answer).

import pytest

import clients.claude_client as claude_client
from clients.errors import APICallFailed


class FakeBlock:
    def __init__(self, type_, text=None):
        self.type = type_
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeBlock("text", text)]


class FakeStreamCtx:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self._response


class FakeMessages:
    """Records every call/stream kwargs. `fail_times` errors happen first,
    then it returns `reply_text` (unary or streamed, matching how call() uses it)."""

    def __init__(self, reply_text="B", fail_times=0):
        self.reply_text = reply_text
        self.fail_times = fail_times
        self.attempts = 0
        self.create_calls = []
        self.stream_calls = []

    def _maybe_fail(self):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError(f"boom {self.attempts}")

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        self._maybe_fail()
        return FakeResponse(self.reply_text)

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        self._maybe_fail()
        return FakeStreamCtx(FakeResponse(self.reply_text))


class FakeClient:
    def __init__(self, **kw):
        self.messages = FakeMessages(**kw)


def no_sleep(monkeypatch):
    monkeypatch.setattr(claude_client.time, "sleep", lambda *a, **k: None)


def use_fake_client(monkeypatch, fake):
    monkeypatch.setattr(claude_client, "get_client", lambda: fake)


# ---------- get_client ----------

def test_get_client_returns_a_singleton(monkeypatch):
    monkeypatch.setattr(claude_client, "_client", None)
    c1 = claude_client.get_client()
    c2 = claude_client.get_client()
    assert c1 is c2


# ---------- extract_answer / _extract_text ----------

def test_extract_answer_delegates_to_extract_letter():
    assert claude_client.extract_answer("The answer is B") == "B"
    assert claude_client.extract_answer("Answer: C", cot=True) == "C"


def test_extract_text_skips_a_leading_thinking_block():
    resp = FakeResponse("the real answer")
    resp.content = [FakeBlock("thinking", "reasoning..."), FakeBlock("text", "the real answer")]
    assert claude_client._extract_text(resp) == "the real answer"


def test_extract_text_returns_empty_string_when_there_is_no_text_block():
    resp = FakeResponse("unused")
    resp.content = [FakeBlock("thinking", "reasoning only")]
    assert claude_client._extract_text(resp) == ""


# ---------- call(): request shape per thinking mode ----------

def test_greedy_mode_sets_temperature_zero_and_a_small_token_cap(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="B")
    use_fake_client(monkeypatch, fake)

    pred, raw = claude_client.call("sys", [{"role": "user", "content": []}])

    assert (pred, raw) == ("B", "B")
    kwargs = fake.messages.create_calls[0]
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 64
    assert "thinking" not in kwargs


def test_greedy_cot_mode_gets_a_bigger_token_cap(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="Answer: C")
    use_fake_client(monkeypatch, fake)

    pred, raw = claude_client.call("sys", [], cot=True)

    assert pred == "C"
    assert fake.messages.create_calls[0]["max_tokens"] == 2048


def test_effort_mode_uses_adaptive_thinking_and_streams(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="A")
    use_fake_client(monkeypatch, fake)

    pred, raw = claude_client.call("sys", [], effort="high")

    assert pred == "A"
    assert fake.messages.create_calls == []  # streamed, not unary
    kwargs = fake.messages.stream_calls[0]
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["max_tokens"] == 32000
    assert "temperature" not in kwargs  # adaptive manages sampling itself


def test_thinking_budget_mode_uses_enabled_thinking_and_streams(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="D")
    use_fake_client(monkeypatch, fake)

    pred, raw = claude_client.call("sys", [], thinking_budget=1000)

    assert pred == "D"
    kwargs = fake.messages.stream_calls[0]
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1000}
    assert kwargs["temperature"] == 1.0
    assert kwargs["max_tokens"] == 1000 + 512  # +2048 instead if cot=True


def test_thinking_budget_cot_mode_adds_cot_headroom(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="Answer: D")
    use_fake_client(monkeypatch, fake)

    claude_client.call("sys", [], thinking_budget=1000, cot=True)

    assert fake.messages.stream_calls[0]["max_tokens"] == 1000 + 2048


def test_call_passes_through_the_requested_model(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="B")
    use_fake_client(monkeypatch, fake)

    claude_client.call("sys", [], model="claude-opus-4-8")

    assert fake.messages.create_calls[0]["model"] == "claude-opus-4-8"


# ---------- call(): retries ----------

def test_call_retries_after_a_transient_error_then_succeeds(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="B", fail_times=2)
    use_fake_client(monkeypatch, fake)

    pred, raw = claude_client.call("sys", [], retries=3)

    assert (pred, raw) == ("B", "B")
    assert fake.messages.attempts == 3


def test_call_raises_api_call_failed_once_retries_are_exhausted(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="B", fail_times=99)
    use_fake_client(monkeypatch, fake)

    with pytest.raises(APICallFailed, match="retries exhausted"):
        claude_client.call("sys", [], retries=2)

    assert fake.messages.attempts == 2
