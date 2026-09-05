"""Deterministic wave planning for rollout-engine operations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar


T = TypeVar("T")
R = TypeVar("R")


def build_engine_group_waves(
    engine_groups: Sequence[T],
    max_inflight_engine_groups: int,
) -> tuple[tuple[tuple[int, T], ...], ...]:
    """Partition engine groups into stable, bounded waves.

    Each item is paired with its original index so callers can coordinate the
    same wave across trainer ranks. ``0`` preserves the existing all-at-once
    behavior. A limit larger than the number of groups is also one wave.
    """
    if max_inflight_engine_groups < 0:
        raise ValueError("max_inflight_engine_groups must be non-negative")
    if not engine_groups:
        return ()

    wave_size = max_inflight_engine_groups or len(engine_groups)
    indexed_groups = tuple(enumerate(engine_groups))
    return tuple(indexed_groups[start : start + wave_size] for start in range(0, len(indexed_groups), wave_size))


def run_engine_group_waves(
    engine_groups: Sequence[T],
    max_inflight_engine_groups: int,
    submit: Callable[[int, T], R],
    wait: Callable[[list[R]], object],
) -> None:
    """Submit each wave concurrently and wait before admitting the next one."""
    for wave in build_engine_group_waves(engine_groups, max_inflight_engine_groups):
        pending = [submit(index, engine_group) for index, engine_group in wave]
        wait(pending)
