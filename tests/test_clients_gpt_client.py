# Tests for clients/gpt_client.py
# Wraps the OpenAI chat API. Two very different request shapes depending on the
# model: classic chat models get temperature=0 + a pinned max_tokens (the
# benchmark-faithful config); reasoning models (gpt-5*, o*) get
# max_completion_tokens + reasoning_effort instead. Retries on error and raises
# APICallFailed once retries run out.

import pytest

import clients.gpt_client as gpt_client
from clients.errors import APICallFailed


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResp:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, reply="B", fail_times=0):
        self.reply = reply
        self.fail_times = fail_times
        self.attempts = 0
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError(f"boom {self.attempts}")
        return FakeResp(self.reply)


class FakeClient:
    def __init__(self, **kw):
        self.chat = _Chat(**kw)


class _Chat:
    def __init__(self, **kw):
        self.completions = FakeCompletions(**kw)


def no_sleep(monkeypatch):
    monkeypatch.setattr(gpt_client.time, "sleep", lambda *a, **k: None)


def use_fake_client(monkeypatch, fake):
    monkeypatch.setattr(gpt_client, "get_client", lambda: fake)


# ---------- get_client ----------

def test_get_client_returns_a_singleton(monkeypatch):
    monkeypatch.setattr(gpt_client, "_client", None)
    c1 = gpt_client.get_client()
    c2 = gpt_client.get_client()
    assert c1 is c2


# ---------- is_reasoning_model ----------

def test_gpt5_models_are_reasoning_models():
    assert gpt_client.is_reasoning_model("gpt-5.6-luna") is True


def test_o_series_models_are_reasoning_models():
    assert gpt_client.is_reasoning_model("o1") is True
    assert gpt_client.is_reasoning_model("o3-mini") is True


def test_bare_o_with_no_digit_is_not_a_reasoning_model():
    assert gpt_client.is_reasoning_model("o") is False


def test_classic_chat_models_are_not_reasoning_models():
    assert gpt_client.is_reasoning_model("gpt-4o-2024-11-20") is False
    assert gpt_client.is_reasoning_model("gpt-4.1") is False


# ---------- extract_answer ----------

def test_extract_answer_delegates_to_extract_letter():
    assert gpt_client.extract_answer("The answer is B") == "B"
    assert gpt_client.extract_answer("Answer: C", cot=True) == "C"


# ---------- call(): classic chat model request shape ----------

def test_classic_model_uses_benchmark_faithful_settings(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply="B")
    use_fake_client(monkeypatch, fake)

    raw = gpt_client.call("sys prompt", [{"type": "text", "text": "hi"}], model="gpt-4o-2024-11-20")

    assert raw == "B"
    kwargs = fake.chat.completions.calls[0]
    assert kwargs["max_tokens"] == gpt_client.BENCHMARK_MAX_TOKENS
    assert kwargs["temperature"] == gpt_client.BENCHMARK_TEMPERATURE
    assert "reasoning_effort" not in kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]


def test_call_defaults_to_the_module_model_when_none_given(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply="B")
    use_fake_client(monkeypatch, fake)

    gpt_client.call("sys", [], model=None)

    assert fake.chat.completions.calls[0]["model"] == gpt_client.MODEL


def test_no_system_prompt_means_no_system_message(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply="B")
    use_fake_client(monkeypatch, fake)

    gpt_client.call(None, [{"type": "text", "text": "hi"}], model="gpt-4o-2024-11-20")

    messages = fake.chat.completions.calls[0]["messages"]
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_service_tier_is_forwarded_when_set(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply="B")
    use_fake_client(monkeypatch, fake)

    gpt_client.call("sys", [], model="gpt-4o-2024-11-20", service_tier="flex")

    assert fake.chat.completions.calls[0]["service_tier"] == "flex"


def test_no_service_tier_means_no_service_tier_key(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply="B")
    use_fake_client(monkeypatch, fake)

    gpt_client.call("sys", [], model="gpt-4o-2024-11-20")

    assert "service_tier" not in fake.chat.completions.calls[0]


def test_empty_reply_content_becomes_empty_string(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply=None)
    use_fake_client(monkeypatch, fake)

    raw = gpt_client.call("sys", [], model="gpt-4o-2024-11-20")

    assert raw == ""


# ---------- call(): reasoning model request shape and headroom ----------

def test_reasoning_model_uses_max_completion_tokens_and_reasoning_effort(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply="B")
    use_fake_client(monkeypatch, fake)

    gpt_client.call("sys", [], model="gpt-5.6-luna", reasoning_effort="none")

    kwargs = fake.chat.completions.calls[0]
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["max_completion_tokens"] == 2000  # base, direct (no cot)


def test_reasoning_model_base_headroom_grows_with_cot(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply="B")
    use_fake_client(monkeypatch, fake)

    gpt_client.call("sys", [], model="gpt-5.6-luna", reasoning_effort="low", cot=True)

    assert fake.chat.completions.calls[0]["max_completion_tokens"] == 4000


def test_reasoning_model_high_effort_gets_8000_headroom_even_without_cot(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply="B")
    use_fake_client(monkeypatch, fake)

    gpt_client.call("sys", [], model="gpt-5.6-luna", reasoning_effort="high")

    assert fake.chat.completions.calls[0]["max_completion_tokens"] == 8000


def test_reasoning_model_xhigh_effort_gets_24000_headroom(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply="B")
    use_fake_client(monkeypatch, fake)

    gpt_client.call("sys", [], model="gpt-5.6-luna", reasoning_effort="xhigh", cot=True)

    # cot base (4000) is still smaller than the xhigh headroom (24000): max wins
    assert fake.chat.completions.calls[0]["max_completion_tokens"] == 24000


def test_o_series_model_is_treated_as_reasoning_too(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply="B")
    use_fake_client(monkeypatch, fake)

    gpt_client.call("sys", [], model="o1", reasoning_effort="medium")

    kwargs = fake.chat.completions.calls[0]
    assert "max_tokens" not in kwargs
    assert kwargs["reasoning_effort"] == "medium"


# ---------- call(): retries ----------

def test_call_retries_after_a_transient_error_then_succeeds(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply="B", fail_times=2)
    use_fake_client(monkeypatch, fake)

    raw = gpt_client.call("sys", [], model="gpt-4o-2024-11-20", retries=3)

    assert raw == "B"
    assert fake.chat.completions.attempts == 3


def test_call_raises_api_call_failed_once_retries_are_exhausted(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply="B", fail_times=99)
    use_fake_client(monkeypatch, fake)

    with pytest.raises(APICallFailed, match="retries exhausted"):
        gpt_client.call("sys", [], model="gpt-4o-2024-11-20", retries=2)

    assert fake.chat.completions.attempts == 2
