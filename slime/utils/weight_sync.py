"""Cache refresh policy shared by the weight transports."""


def initial_weight_version(args, loaded_start_rollout_id: int | None) -> int:
    """Keep restored weights aligned with the next rollout and any live version.

    An explicit start ID also identifies fresh model-only initialization, whose
    loader can return iteration zero even though the first rollout is zero.
    """
    start_rollout_id = getattr(args, "start_rollout_id", None)
    if start_rollout_id is None:
        start_rollout_id = loaded_start_rollout_id
    return max(getattr(args, "update_weight_start_version", 0), start_rollout_id or 0)


def should_flush_cache(flush_cache_interval: int, weight_version: int, initial_weight_version: int = 0) -> bool:
    """Refresh at startup and every N subsequent syncs; nonpositive preserves KV.

    The first publication flushes even when resuming with a nonzero version.
    The subsequent schedule keeps its phase relative to version 1.
    A full refresh aborts requests; other updates pause them in place.
    """
    return weight_version == initial_weight_version + 1 or (
        flush_cache_interval > 0 and (weight_version - 1) % flush_cache_interval == 0
    )
