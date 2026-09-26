# Tests for clients/errors.py
# This file defines one exception, APICallFailed, raised by a model client when
# all its retries are exhausted.

import pytest

from clients.errors import APICallFailed


def test_api_call_failed_is_a_runtime_error():
    assert issubclass(APICallFailed, RuntimeError)


def test_api_call_failed_carries_its_message():
    with pytest.raises(APICallFailed) as exc_info:
        raise APICallFailed("claude opus: 3 retries exhausted: boom")
    assert str(exc_info.value) == "claude opus: 3 retries exhausted: boom"


def test_api_call_failed_can_be_caught_as_runtime_error():
    try:
        raise APICallFailed("failure")
    except RuntimeError as e:
        assert isinstance(e, APICallFailed)
