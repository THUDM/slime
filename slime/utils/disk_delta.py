from __future__ import annotations

import os
import zlib

import numpy as np

# The delta phases (diff, zstd, checksum) are memory-bandwidth bound and release the GIL,
# so a thread pool over tensors recovers the bandwidth one thread leaves idle.
NUM_WORKERS = min(32, (os.cpu_count() or 8))

# Trainer-side (publish) helpers for disk-level delta weight sync. The receive side —
# materializing the host-local checkpoint and applying published deltas in place — lives in
# the engine behind its /pull_weights endpoint (sglang.srt.weight_sync.disk_delta), so it
# runs on every host of a multi-node engine while slime only talks to one endpoint.


def overwrite_encode(new: np.ndarray, changed_mask: np.ndarray) -> np.ndarray:
    """The 'overwrite' delta: changed-position count (u4), positions (u4 each), then new values.
    Idempotent to apply, unlike xor (an involution); the trainer picks the encoding per the docs."""
    pos = np.flatnonzero(changed_mask).astype("<u4")
    return np.concatenate([np.array([pos.size], "<u4").view(np.uint8), pos.view(np.uint8), new[changed_mask]])


class _Adler32:
    """adler32 behind the incremental .update / .hexdigest interface the hash objects expose."""

    def __init__(self):
        self._value = 1

    def update(self, data) -> None:
        self._value = zlib.adler32(data, self._value)

    def hexdigest(self) -> str:
        return f"{self._value:08x}"


def _new_hasher(algorithm: str):
    if algorithm == "xxh3-128":
        import xxhash

        return xxhash.xxh3_128()
    if algorithm == "blake3":
        import blake3

        return blake3.blake3()
    if algorithm == "adler32":
        return _Adler32()
    raise KeyError(f"unknown checksum algorithm {algorithm!r}")


def checksum(algorithm: str, buf) -> str:
    hasher = _new_hasher(algorithm)
    hasher.update(buf)
    return hasher.hexdigest()
