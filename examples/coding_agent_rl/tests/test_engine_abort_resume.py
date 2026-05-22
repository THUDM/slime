"""Tests for §6 ENGINE generate_with_abort_resume (P2).

Covers SPEC §7.1 entry `test_engine_abort_resume.py`:
  * accumulator extends across abort -> resume cycles
  * MAX_RESUME_ATTEMPTS cap returns whatever was accumulated
  * on_partial callback bumps caller-side counters
  * budget exhaustion returns finish='length'
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import middleware as mw  # noqa: E402


def _mk_meta(ids: list[int], finish: str = "stop") -> dict:
    return {
        "meta_info": {
            "output_token_logprobs": [[0.0, i, ""] for i in ids],
            "finish_reason": {"type": finish},
        }
    }


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if not asyncio.iscoroutine(coro) \
        else asyncio.new_event_loop().run_until_complete(coro)


def test_abort_then_resume_concatenates_output() -> None:
    """One abort midway -> caller sees pre+post accumulated as single output."""
    seq = [_mk_meta([1, 2, 3], "abort"), _mk_meta([4, 5], "stop")]
    coordinator = mw.AbortCoordinator()
    partials: list[tuple[list[int], str]] = []

    async def fake_post(_url, _input, _sp):
        return seq.pop(0)

    async def on_partial(ids, why):
        partials.append((ids, why))

    async def drive():
        # No real wait_for_resume: we instantly clear after the first call.
        # Since coordinator starts cleared, we don't need to set/clear here.
        out, finish, _ = await mw.generate_with_abort_resume(
            sglang_url="http://x", input_ids=[100, 101],
            sampling_params={"max_new_tokens": 100},
            abort=coordinator, on_partial=on_partial,
        )
        return out, finish

    # Monkey-patch _post_generate
    orig = mw._post_generate
    mw._post_generate = fake_post
    try:
        loop = asyncio.new_event_loop()
        try:
            out, finish = loop.run_until_complete(drive())
        finally:
            loop.close()
    finally:
        mw._post_generate = orig

    assert out == [1, 2, 3, 4, 5], f"got {out}"
    assert finish == "stop"
    assert len(partials) == 1
    assert partials[0][0] == [1, 2, 3]
    assert partials[0][1] == "abort"


def test_max_attempts_caps_loop() -> None:
    """If sglang keeps returning abort, loop caps at MAX_RESUME_ATTEMPTS."""
    coordinator = mw.AbortCoordinator()
    call_count = [0]

    async def fake_post(_url, _input, _sp):
        call_count[0] += 1
        return _mk_meta([call_count[0]], "abort")

    async def drive():
        return await mw.generate_with_abort_resume(
            sglang_url="http://x", input_ids=[100],
            sampling_params={"max_new_tokens": 1000},
            abort=coordinator,
        )

    orig = mw._post_generate
    orig_max = mw.MAX_RESUME_ATTEMPTS
    mw._post_generate = fake_post
    mw.MAX_RESUME_ATTEMPTS = 3
    try:
        loop = asyncio.new_event_loop()
        try:
            out, finish, _ = loop.run_until_complete(drive())
        finally:
            loop.close()
    finally:
        mw._post_generate = orig
        mw.MAX_RESUME_ATTEMPTS = orig_max

    assert call_count[0] == 3
    assert finish == "abort"
    assert out == [1, 2, 3]


def test_budget_exhaustion_returns_length() -> None:
    """When accumulated >= max_new_tokens, return finish='length'."""
    coordinator = mw.AbortCoordinator()
    seq = [_mk_meta([10, 11, 12, 13, 14], "abort")]  # accumulated=5, budget=5

    async def fake_post(_url, _input, _sp):
        return seq.pop(0)

    async def drive():
        return await mw.generate_with_abort_resume(
            sglang_url="http://x", input_ids=[1],
            sampling_params={"max_new_tokens": 5},
            abort=coordinator,
        )

    orig = mw._post_generate
    mw._post_generate = fake_post
    try:
        loop = asyncio.new_event_loop()
        try:
            out, finish, _ = loop.run_until_complete(drive())
        finally:
            loop.close()
    finally:
        mw._post_generate = orig

    # First call returns 5 ids with finish=abort. Loop wakes up, budget = 5 - 5 = 0, returns 'length'.
    assert out == [10, 11, 12, 13, 14]
    assert finish == "length"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
