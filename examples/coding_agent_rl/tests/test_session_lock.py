"""Tests for §8 HANDLER P6 invariant: Session.lock is the only sync primitive.

Covers SPEC §7.1 entry `test_session_lock.py`. Verifies that:
  * concurrent requests on the same session_id serialize through s.lock
  * different session_ids don't block each other
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import middleware as mw  # noqa: E402


def test_session_lock_serializes_same_sid() -> None:
    """Two coroutines acquiring the same Session.lock cannot overlap."""
    s = mw.Session()
    timeline: list[str] = []

    async def task(label: str):
        async with s.lock:
            timeline.append(f"{label}-enter")
            await asyncio.sleep(0.01)
            timeline.append(f"{label}-exit")

    async def drive():
        await asyncio.gather(task("A"), task("B"))

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(drive())
    finally:
        loop.close()

    # Either A or B finishes fully before the other enters.
    enter_idx = [i for i, t in enumerate(timeline) if t.endswith("-enter")]
    exit_idx = [i for i, t in enumerate(timeline) if t.endswith("-exit")]
    assert enter_idx[1] > exit_idx[0], (
        f"sessions interleaved (lock broken): {timeline}"
    )


def test_different_sids_are_independent() -> None:
    """Each Session has its own lock; concurrent on different sids may interleave."""
    s1 = mw.Session()
    s2 = mw.Session()
    timeline: list[str] = []

    async def task(s, label):
        async with s.lock:
            timeline.append(f"{label}-enter")
            await asyncio.sleep(0.01)
            timeline.append(f"{label}-exit")

    async def drive():
        await asyncio.gather(task(s1, "A"), task(s2, "B"))

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(drive())
    finally:
        loop.close()

    # Both should enter before either exits (interleave).
    enter_idx = [i for i, t in enumerate(timeline) if t.endswith("-enter")]
    exit_idx = [i for i, t in enumerate(timeline) if t.endswith("-exit")]
    assert min(exit_idx) > max(enter_idx), (
        f"locks unexpectedly serialized different sids: {timeline}"
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
