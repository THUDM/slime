"""Lightweight helper for inspecting engine apply results from delta-weight sync.

Kept in its own module so it can be unit-tested without pulling in torch, ray,
or any other GPU/distributed dependency.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_apply_results(results: list) -> None:
    """Log and raise on any failed delta-weight apply reported by a receiver engine.

    The SGLang receiver wraps its apply logic in a try/except and always returns
    ``(success: bool, message: str)``.  ``_finalize_sync`` previously called
    ``ray.get()`` and discarded these return values, so a failed apply was
    silent: the sender snapshot advanced past what receivers actually hold, and
    subsequent diffs were computed against a stale baseline (issue #2104).

    This helper must be called immediately after ``ray.get()`` to surface
    failures before the sync is declared complete.

    Args:
        results: the list returned by ``ray.get(object_refs)`` — one entry per
            engine, expected to be ``(success, message)`` tuples.  Entries that
            are not 2-tuples are treated as successful (forward-compatible with
            engines that do not yet return structured results).

    Raises:
        RuntimeError: if one or more engines reported ``success=False``.
    """
    failures = [
        (idx, result[1])
        for idx, result in enumerate(results)
        if isinstance(result, tuple) and len(result) == 2 and not result[0]
    ]
    if not failures:
        return
    for idx, msg in failures:
        logger.error("Engine[%d] failed to apply delta weights: %s", idx, msg)
    raise RuntimeError(
        f"Delta weight apply failed on {len(failures)}/{len(results)} engine(s). "
        "The sender snapshot has advanced past what receivers actually hold; "
        "subsequent diffs will be computed against a stale baseline. "
        "See per-engine error messages logged above."
    )
