"""CPU-only lifecycle tests for the remote reward-model HTTP session."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

# ``rm_hub`` imports the math scorers eagerly. The lifecycle tests do not use
# LaTeX parsing, and the lightweight CPU test environment may omit pylatexenc.
if "pylatexenc" not in sys.modules:
    try:
        import pylatexenc  # noqa: F401
    except ModuleNotFoundError:

        class _LatexNodes2Text:
            def latex_to_text(self, value):
                return value

        pylatexenc = types.ModuleType("pylatexenc")
        pylatexenc.latex2text = types.SimpleNamespace(LatexNodes2Text=_LatexNodes2Text)
        sys.modules["pylatexenc"] = pylatexenc

from slime.rollout import rm_hub


def _close_loop_session(loop: asyncio.AbstractEventLoop) -> None:
    """Close this loop's test session without ever awaiting it elsewhere."""
    if loop.is_closed():
        return
    cleanup = getattr(rm_hub, "cleanup_remote_rm_session", None)
    if cleanup is not None:
        loop.run_until_complete(cleanup())
    else:
        # RED-phase compatibility with the pre-fix single-session cache.
        session = rm_hub._shared_session
        if session is not None and not session.closed and session._loop is loop:
            loop.run_until_complete(session.close())


@pytest.mark.unit
def test_remote_rm_session_is_reused_within_its_event_loop():
    loop = asyncio.new_event_loop()
    try:

        async def get_twice():
            first = rm_hub._get_shared_session()
            second = rm_hub._get_shared_session()
            return first, second, asyncio.get_running_loop()

        first, second, running_loop = loop.run_until_complete(get_twice())

        assert first is second
        assert first._loop is running_loop
    finally:
        _close_loop_session(loop)
        loop.close()


@pytest.mark.unit
def test_remote_rm_sessions_are_isolated_between_live_event_loops():
    first_loop = asyncio.new_event_loop()
    second_loop = asyncio.new_event_loop()
    try:
        first_session = first_loop.run_until_complete(_get_session())
        second_session = second_loop.run_until_complete(_get_session())

        assert second_session is not first_session
        assert first_loop.run_until_complete(_get_session()) is first_session
        assert second_loop.run_until_complete(_get_session()) is second_session
    finally:
        _close_loop_session(first_loop)
        _close_loop_session(second_loop)
        first_loop.close()
        second_loop.close()


@pytest.mark.unit
def test_closed_session_or_loop_is_never_reused():
    first_loop = asyncio.new_event_loop()
    try:
        first_session = first_loop.run_until_complete(_get_session())
        first_loop.run_until_complete(first_session.close())
    finally:
        _close_loop_session(first_loop)
        first_loop.close()

    second_loop = asyncio.new_event_loop()
    try:
        second_session = second_loop.run_until_complete(_get_session())
        assert second_session is not first_session
        assert second_session._loop is second_loop
    finally:
        _close_loop_session(second_loop)
        second_loop.close()


@pytest.mark.unit
def test_cleanup_closes_and_removes_only_the_current_loops_session():
    first_loop = asyncio.new_event_loop()
    second_loop = asyncio.new_event_loop()
    try:
        first_session = first_loop.run_until_complete(_get_session())
        second_session = second_loop.run_until_complete(_get_session())

        first_loop.run_until_complete(rm_hub.cleanup_remote_rm_session())

        assert first_session.closed
        assert not second_session.closed
        assert first_loop.run_until_complete(_get_session()) is not first_session
        assert second_loop.run_until_complete(_get_session()) is second_session
    finally:
        _close_loop_session(first_loop)
        _close_loop_session(second_loop)
        first_loop.close()
        second_loop.close()


async def _get_session():
    return rm_hub._get_shared_session()
