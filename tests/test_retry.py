"""CPU unit tests for slime.utils.retry.

These tests import ONLY ``slime.utils.retry`` (stdlib-only), so they run under
the CPU-only ``cpu-unittest`` CI job — no GPU, no Ray, no sglang. They exercise
the generic retry mechanism with a caller-supplied ``should_retry`` predicate.
"""

import pytest

from slime.utils import retry
from slime.utils.retry import retry_with_backoff

pytestmark = pytest.mark.unit


# Stand-ins for the Ray exception hierarchy, to drive an isinstance-based
# predicate exactly like the rollout-bringup call site does.
class Transient(Exception):
    """Retryable (stands in for ray.exceptions.ActorUnavailableError)."""


class Permanent(Exception):
    """Not retryable (stands in for ray.exceptions.ActorDiedError)."""


def _retry_transient(exc: Exception) -> bool:
    return isinstance(exc, Transient)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make backoff instantaneous so the retry tests run fast."""
    monkeypatch.setattr(retry.time, "sleep", lambda _s: None)


def test_recovers_after_retryable_failures():
    """A thunk that fails retryably N times then succeeds is retried to success."""
    sentinel = object()
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        if calls["n"] < 3:  # fail on attempts 1 and 2, succeed on 3
            raise Transient("temporarily unavailable")
        return sentinel

    result = retry_with_backoff(thunk, should_retry=_retry_transient, what="test", max_retries=3)

    assert result is sentinel
    assert calls["n"] == 3


def test_exhaustion_reraises_last_error():
    """A thunk that always fails retryably re-raises after exactly max_retries attempts."""
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        raise Transient("still unavailable")

    with pytest.raises(Transient):
        retry_with_backoff(thunk, should_retry=_retry_transient, what="test", max_retries=3)

    assert calls["n"] == 3  # no swallow: exhausted then re-raised


@pytest.mark.parametrize("exc_type", [Permanent, RuntimeError])
def test_non_retryable_error_propagates_immediately(exc_type):
    """An error the predicate rejects propagates on the FIRST attempt — never masked.

    Covers a permanent backend error (the ActorDiedError stand-in) and a
    generic RuntimeError (e.g. CUDA OOM).
    """
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        raise exc_type("not retryable")

    with pytest.raises(exc_type):
        retry_with_backoff(thunk, should_retry=_retry_transient, what="test", max_retries=3)

    assert calls["n"] == 1


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_control_flow_exceptions_are_never_intercepted(exc_type):
    """KeyboardInterrupt/SystemExit propagate at once, without consulting the predicate.

    The handler catches ``Exception``, so control-flow exceptions cannot be
    retried no matter what ``should_retry`` returns — pinned here with a
    predicate that retries everything it is shown.
    """
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
