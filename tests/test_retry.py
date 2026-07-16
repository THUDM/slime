"""CPU unit tests for slime.utils.retry."""

import pytest

from slime.utils import retry
from slime.utils.retry import retry_with_backoff

NUM_GPUS = 0
pytestmark = pytest.mark.unit


class Transient(Exception):
    """Retryable stand-in for a transient backend error."""


class Permanent(Exception):
    """Non-retryable stand-in for a permanent backend error."""


def _retry_transient(exc: Exception) -> bool:
    return isinstance(exc, Transient)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _s: None)


def test_recovers_after_retryable_failures():
    sentinel = object()
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Transient("temporarily unavailable")
        return sentinel

    result = retry_with_backoff(thunk, should_retry=_retry_transient, what="test", max_retries=3)

    assert result is sentinel
    assert calls["n"] == 3


def test_exhaustion_reraises_last_error():
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        raise Transient("still unavailable")

    with pytest.raises(Transient):
        retry_with_backoff(thunk, should_retry=_retry_transient, what="test", max_retries=3)

    assert calls["n"] == 3


@pytest.mark.parametrize("exc_type", [Permanent, RuntimeError])
def test_non_retryable_error_propagates_immediately(exc_type):
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        raise exc_type("not retryable")

    with pytest.raises(exc_type):
        retry_with_backoff(thunk, should_retry=_retry_transient, what="test", max_retries=3)

    assert calls["n"] == 1


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_control_flow_exceptions_are_never_intercepted(exc_type):
    calls = {"n": 0}
    seen_by_predicate: list[Exception] = []

    def retry_everything(exc: Exception) -> bool:
        seen_by_predicate.append(exc)
        return True

    def thunk():
        calls["n"] += 1
        raise exc_type()

    with pytest.raises(exc_type):
        retry_with_backoff(thunk, should_retry=retry_everything, what="test", max_retries=3)
    assert calls["n"] == 1
    assert seen_by_predicate == []


def test_success_on_first_attempt_calls_thunk_once():
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        return "ok"

    assert retry_with_backoff(thunk, should_retry=_retry_transient, what="test") == "ok"
    assert calls["n"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
