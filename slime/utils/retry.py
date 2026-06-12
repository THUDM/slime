"""Generic bounded retry with exponential jittered backoff.

A small synchronous counterpart to the async ``should_retry`` + loop idiom used
elsewhere in slime (``slime/rollout/openai_workflow/client.py`` for the proxy
client, ``slime/rollout/rm_hub`` for the reward model). The caller supplies a
``should_retry`` predicate so the retry policy stays decoupled from *what*
counts as retryable — this module is not tied to any backend (HTTP, Ray, ...).

Backoff/jitter mirror the existing idiom in ``slime/rollout/rm_hub/__init__.py``.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    thunk: Callable[[], T],
    *,
    should_retry: Callable[[Exception], bool],
    what: str,
    max_retries: int = 3,
    base_backoff: float = 2.0,
    max_backoff: float = 30.0,
) -> T:
    """Run ``thunk`` with bounded retries, gated by ``should_retry``.

    ``thunk`` must be safe to re-invoke on a retried failure — wrap an
    idempotent operation (e.g. a ``ray.get`` on already-submitted object refs),
    not a non-idempotent side effect.

    ``should_retry(exc)`` decides whether a raised ``Exception`` is retryable;
    anything it rejects propagates immediately (so genuine errors are never
    masked), and the last retryable error is re-raised once ``max_retries``
    attempts — the first call included — are exhausted (so a persistent
    failure fails fast rather than hanging). ``BaseException``s that are not
    ``Exception``s (``KeyboardInterrupt``, ``SystemExit``) are never
    intercepted. Backoff is exponential with an additive 0-1s jitter: the
    deterministic floor preserves a minimum recovery window, the jitter
    de-synchronizes concurrent callers.
    """
    for attempt in range(max_retries):
        try:
            return thunk()
        except Exception as e:
            if not should_retry(e) or attempt + 1 >= max_retries:
                raise
            backoff = min(base_backoff * (2**attempt), max_backoff) + random.random()
            logger.warning(
                f"{what}: {type(e).__name__}, retrying in {backoff:.1f}s ({attempt + 1}/{max_retries}): {e}"
            )
            time.sleep(backoff)
    raise AssertionError("retry_with_backoff exhausted without return or raise")  # unreachable
