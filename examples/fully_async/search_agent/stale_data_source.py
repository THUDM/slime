"""Staleness-aware DataSource for fully-async rollout.

Inherits ``RolloutDataSourceWithBuffer`` and adds staleness counting: when the
number of groups consumed since the last ``reset_staleness()`` exceeds
``staleness_threshold * rollout_batch_size``, ``get_samples()`` blocks until
training syncs weights and calls ``reset_staleness()``.

Only ``get_samples()`` is overridden; all other methods (``add_samples``,
``save``, ``load``, ``__len__``, etc.) are inherited unchanged.

Usage::

    --data-source-path examples.fully_async.search_agent.stale_data_source.StalenessDataSource
    --staleness-threshold 1
"""

from __future__ import annotations

import logging
import threading

from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.types import Sample

__all__ = ["StalenessDataSource"]

logger = logging.getLogger(__name__)


class StalenessDataSource(RolloutDataSourceWithBuffer):
    """DataSource that pauses generation when staleness exceeds threshold."""

    def __init__(self, args):
        super().__init__(args)
        self._threshold = getattr(args, "staleness_threshold", 1) * args.rollout_batch_size
        self._counter = 0
        self._lock = threading.Lock()
        self._resume = threading.Event()
        self._resume.set()

        logger.info(
            "StalenessDataSource: threshold=%d groups (staleness_threshold=%d × rollout_batch_size=%d)",
            self._threshold,
            getattr(args, "staleness_threshold", 1),
            args.rollout_batch_size,
        )

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Return *num_samples* groups, blocking if staleness limit reached."""
        self._resume.wait()
        samples = super().get_samples(num_samples)

        with self._lock:
            self._counter += len(samples)
            if self._counter >= self._threshold:
                self._resume.clear()
                logger.info(
                    "StalenessDataSource: paused (counter=%d >= threshold=%d)",
                    self._counter,
                    self._threshold,
                )

        return samples

    def reset_staleness(self) -> None:
        """Reset staleness counter and resume generation (called after weight sync)."""
        with self._lock:
            old = self._counter
            self._counter = 0
        self._resume.set()
        logger.info("StalenessDataSource: reset (was %d), resumed", old)
