"""FP8 weight reuse within one Megatron forward/backward schedule.

The scope must end before optimizer updates or model-weight swaps. This is
deliberately not a persistent tensor-version cache: optimizers/loaders can write
through ``.data`` without incrementing the Parameter's version counter.
"""

from collections.abc import Callable, Hashable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class _Entry:
    # Retain detached source views so even Parameter.data replacement cannot
    # free/recycle the old storage address while its cache entry remains live.
    weights: tuple[torch.Tensor, ...]
    signature: tuple
    value: tuple[torch.Tensor, torch.Tensor]


@dataclass
class _WeightCache:
    entries: dict[Hashable, _Entry] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0


# Megatron actors execute schedules serially within a process. A process-local
# scope also reaches Python checkpoint recomputation on autograd worker threads.
_active_cache: _WeightCache | None = None


@contextmanager
def deepgemm_weight_cache_scope(enabled: bool = False) -> Iterator[_WeightCache | None]:
    """Create an isolated cache, released on success or failure.

    Nested scopes never reuse each other's weights (e.g. actor/ref evaluation).
    Direct kernel calls outside a scope retain their original uncached behavior.
    """
    global _active_cache
    previous = _active_cache
    cache = _WeightCache() if enabled else None
    _active_cache = cache
    try:
        yield cache
    finally:
        if cache is not None:
            cache.entries.clear()
        _active_cache = previous


def is_deepgemm_weight_cache_enabled() -> bool:
    return _active_cache is not None


def get_quantized_weight(
    key: Hashable,
    weights: Sequence[torch.Tensor],
    quantize: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse an immutable, independently owned quantized weight/scale pair.

    Callers must include the quantization format and grouping in ``key`` and
    must not return buffers that will be overwritten by activation workspaces.
    Version/layout checks additionally invalidate ordinary in-place mutations
    within a schedule. Inference tensors lack version counters, so bypass them.
    """
    cache = _active_cache
    if cache is None:
        return quantize()
    weights = tuple(weights)
    try:
        signature = tuple(
            (w._version, w.data_ptr(), w.device, w.dtype, tuple(w.shape), tuple(w.stride())) for w in weights
        )
    except RuntimeError:
        return quantize()
    entry = cache.entries.get(key)
    if entry is not None and entry.signature == signature:
        cache.hits += 1
        return entry.value
    # Drop a stale allocation before building its replacement.
    cache.entries.pop(key, None)
    del entry
    value = quantize()
    cache.entries[key] = _Entry(tuple(w.detach() for w in weights), signature, value)
    cache.misses += 1
    return value
