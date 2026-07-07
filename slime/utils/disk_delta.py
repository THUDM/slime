from __future__ import annotations

import glob
import json
import os
import struct
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


def _tensor_locations(ckpt_dir: str) -> dict[str, tuple[str, int, int]]:
    """Map each tensor name to (file, byte offset, nbytes) by reading every safetensors header."""
    locations: dict[str, tuple[str, int, int]] = {}
    for path in glob.glob(os.path.join(ckpt_dir, "*.safetensors")):
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            locations[name] = (path, 8 + header_len + begin, end - begin)
    return locations


def make_tensor_reader(ckpt_dir: str):
    """Index the headers once, then return ``read(name) -> uint8 bytes`` that seeks straight to the
    tensor — for reading many tensors without rescanning every header. KeyError if absent."""
    locations = _tensor_locations(ckpt_dir)

    def read(name: str) -> np.ndarray:
        path, offset, nbytes = locations[name]
        with open(path, "rb") as f:
            f.seek(offset)
            return np.frombuffer(f.read(nbytes), dtype=np.uint8)

    return read
