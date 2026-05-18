"""Regression tests for the Semaphore / HTTP-client event-loop rebind.

Without the rebind, re-entering the rollout actor on a different asyncio
loop (e.g. rollout -> eval) causes either ``RuntimeError: <...> is bound
to a different event loop`` or a silent hang. These tests exercise that
transition by running two separate ``asyncio.run(...)`` calls (each
spins up a fresh loop) and asserting that:

  1. The cached primitive is rebuilt on the second call
     (identity check: ``is not``).
  2. Pre-patch behavior — the primitive stays pinned to the first loop
     — is what the patched code avoids; this is implicit in the test
     (if the cache weren't rebuilt, the second-loop user would crash
     or hang).

CPU-only; no Ray, no GPU.
"""

import asyncio
import types

import pytest


# ---------------------------------------------------------------------------
# GenerateState.semaphore
# ---------------------------------------------------------------------------


def _make_state():
    """Build a GenerateState with the minimum args it reads during init,
    skipping the tokenizer / processor / sampling-params setup that would
    require a real HF checkpoint."""
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.misc import SingletonMeta

    # Reset singleton so each test starts fresh.
    SingletonMeta._instances.pop(GenerateState, None)

    args = types.SimpleNamespace(
        sglang_server_concurrency=2,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
    )

    state = GenerateState.__new__(GenerateState)
    state.args = args
    state._semaphore_value = (
        args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
    )
    state._semaphore = None
    state._semaphore_loop_ref = None
    return state


def test_semaphore_rebinds_across_fresh_event_loops():
    """Running `touch()` in two separate asyncio.run() calls spins up two
    different loops; the second call must see a fresh Semaphore bound to
    the new loop."""
    state = _make_state()

    captured = []

    async def touch():
        # Acquire/release to force the Semaphore to bind to this loop.
        async with state.semaphore:
            pass
        captured.append(state.semaphore)

    asyncio.run(touch())
    first_sem = captured[-1]

    asyncio.run(touch())
    second_sem = captured[-1]

    assert first_sem is not second_sem, (
        "Semaphore was not rebuilt on second asyncio.run(); the event-loop "
        "rebind is not firing."
    )


def test_semaphore_reused_within_same_loop():
    """Within a single asyncio.run(), repeated property access returns the
    same Semaphore — otherwise concurrent callers would see different
    objects and the concurrency cap would not hold."""
    state = _make_state()

    async def run():
        sem_a = state.semaphore
        async with sem_a:
            pass
        sem_b = state.semaphore
        return sem_a, sem_b

    a, b = asyncio.run(run())
    assert a is b


# ---------------------------------------------------------------------------
# _get_http_client
# ---------------------------------------------------------------------------


def test_http_client_rebinds_across_fresh_event_loops():
    """Analog of the semaphore test for ``_get_http_client``. The client
    is module-global; we reset it per test."""
    import slime.utils.http_utils as http_utils

    http_utils._http_client = None
    http_utils._http_client_loop_ref = None
    http_utils._client_concurrency = 4  # any positive

    captured = []

    async def touch():
        client = http_utils._get_http_client()
        captured.append(client)

    asyncio.run(touch())
    first_client = captured[-1]

    asyncio.run(touch())
    second_client = captured[-1]

    assert first_client is not second_client, (
        "httpx.AsyncClient was not rebuilt on second asyncio.run(); the "
        "event-loop rebind is not firing."
    )


def test_http_client_reused_within_same_loop():
    import slime.utils.http_utils as http_utils

    http_utils._http_client = None
    http_utils._http_client_loop_ref = None
    http_utils._client_concurrency = 4

    async def run():
        a = http_utils._get_http_client()
        b = http_utils._get_http_client()
        return a, b

    a, b = asyncio.run(run())
    assert a is b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
