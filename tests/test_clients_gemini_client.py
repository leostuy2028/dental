# Tests for clients/gemini_client.py
# Wraps the Gemini API: builds the request (temperature=0, pinned max tokens,
# optional native thinking), retries on error, and raises APICallFailed once
# retries run out (so the harness can skip the item instead of recording an
# error string as an answer).

import pytest

import clients.gemini_client as gemini_client
from clients.errors import APICallFailed


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, reply_text="B", fail_times=0):
        self.reply_text = reply_text
        self.fail_times = fail_times
        self.attempts = 0
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError(f"boom {self.attempts}")
        return FakeResponse(self.reply_text)


class FakeClient:
    def __init__(self, **kw):
        self.models = FakeModels(**kw)


def no_sleep(monkeypatch):
    monkeypatch.setattr(gemini_client.time, "sleep", lambda *a, **k: None)


def use_fake_client(monkeypatch, fake):
    monkeypatch.setattr(gemini_client, "get_client", lambda: fake)


# ---------- get_client ----------

def test_get_client_returns_a_singleton(monkeypatch):
    monkeypatch.setattr(gemini_client, "_client", None)
    c1 = gemini_client.get_client()
    c2 = gemini_client.get_client()
    assert c1 is c2


# ---------- extract_answer ----------

def test_extract_answer_delegates_to_extract_letter():
    assert gemini_client.extract_answer("The answer is B") == "B"
    assert gemini_client.extract_answer("Answer: C", cot=True) == "C"


# ---------- call(): request shape ----------

def test_default_call_uses_greedy_temperature_and_pinned_max_tokens(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="B")
    use_fake_client(monkeypatch, fake)

    pred, raw = gemini_client.call(["hello"])

    assert (pred, raw) == ("B", "B")
    kwargs = fake.models.calls[0]
    assert kwargs["model"] == gemini_client.MODEL
    assert kwargs["contents"] == ["hello"]
    assert kwargs["config"].temperature == 0.0
    assert kwargs["config"].max_output_tokens == gemini_client.MAX_OUTPUT_TOKENS
    assert kwargs["config"].thinking_config is None


def test_thinking_budget_is_set_on_the_config(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="A")
    use_fake_client(monkeypatch, fake)

    gemini_client.call(["hello"], thinking_budget=256)

    cfg = fake.models.calls[0]["config"]
    assert cfg.thinking_config.thinking_budget == 256


def test_a_none_response_text_becomes_an_empty_raw_string(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text=None)
    use_fake_client(monkeypatch, fake)

    pred, raw = gemini_client.call(["hello"])

    assert raw == ""
    assert pred is None


def test_success_path_sleeps_when_delay_seconds_is_set(monkeypatch):
    # DELAY_SECONDS is 0 by default (no pacing needed for Gemini), so the success-path
    # sleep is normally skipped; force it on to cover that branch.
    monkeypatch.setattr(gemini_client, "DELAY_SECONDS", 5)
    slept = []
    monkeypatch.setattr(gemini_client.time, "sleep", lambda s: slept.append(s))
    fake = FakeClient(reply_text="B")
    use_fake_client(monkeypatch, fake)

    gemini_client.call(["hello"])

    assert slept == [5]


def test_cot_flag_is_forwarded_to_the_extractor(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="Answer: D")
    use_fake_client(monkeypatch, fake)

    pred, raw = gemini_client.call(["hello"], cot=True)

    assert pred == "D"


# ---------- call(): retries ----------

def test_call_retries_after_a_transient_error_then_succeeds(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="C", fail_times=2)
    use_fake_client(monkeypatch, fake)

    pred, raw = gemini_client.call(["hello"], retries=3)

    assert (pred, raw) == ("C", "C")
    assert fake.models.attempts == 3


def test_call_raises_api_call_failed_once_retries_are_exhausted(monkeypatch):
    no_sleep(monkeypatch)
    fake = FakeClient(reply_text="C", fail_times=99)
    use_fake_client(monkeypatch, fake)

    with pytest.raises(APICallFailed, match="retries exhausted"):
        gemini_client.call(["hello"], retries=2)

    assert fake.models.attempts == 2
