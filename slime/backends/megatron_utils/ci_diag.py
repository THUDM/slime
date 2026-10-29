"""Opt-in diagnostics for host/pinned-memory failures (enabled with SLIME_CI_DIAG=1)."""

import logging
import os
import resource

import torch

logger = logging.getLogger(__name__)

_pinned = {"calls": 0, "bytes": 0}
_patched = False


def enabled() -> bool:
    return os.environ.get("SLIME_CI_DIAG", "0") == "1"


def _read(path: str, limit: int = 4000) -> str:
    try:
        with open(path) as f:
            return f.read(limit).strip()
    except OSError as e:
        return f"<unavailable: {e}>"


def _meminfo() -> str:
    keep = ("MemTotal", "MemAvailable", "Shmem", "Mlocked", "Unevictable", "HugePages_Total")
    lines = [ln.split() for ln in _read("/proc/meminfo", 100000).splitlines()]
    return " ".join(f"{k[0].rstrip(':')}={k[1]}kB" for k in lines if k and k[0].rstrip(":") in keep)


def log_environment(tag: str) -> None:
    """Log GPU/driver, cgroup, memlock and pinned-memory state of this process."""
    if not enabled():
        return
    parts = [f"[ci-diag] {tag} rank={os.environ.get('RANK', '?')}"]
    try:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        free, total = torch.cuda.mem_get_info()
        parts.append(f"gpu={p.name} total_GiB={total / 2**30:.1f} free_GiB={free / 2**30:.1f}")
    except Exception as e:  # diagnostics must never fail the run
        parts.append(f"gpu=<error {e}>")
    parts.append("driver=" + _read("/proc/driver/nvidia/version", 200).splitlines()[0])
    parts.append(f"torch={torch.__version__} cuda={torch.version.cuda}")
    parts.append(f"memlock={resource.getrlimit(resource.RLIMIT_MEMLOCK)}")
    parts.append(
        f"cgroup.memory.max={_read('/sys/fs/cgroup/memory.max')} current={_read('/sys/fs/cgroup/memory.current')}"
    )
    parts.append(f"meminfo: {_meminfo()}")
    parts.append(f"pin_memory calls={_pinned['calls']} bytes_GiB={_pinned['bytes'] / 2**30:.2f}")
    try:
        s = torch.cuda.memory.host_memory_stats()
        parts.append(
            f"host_alloc_GiB={s.get('allocated_bytes.all.current', 0) / 2**30:.2f} "
            f"peak_GiB={s.get('allocated_bytes.all.peak', 0) / 2**30:.2f} "
            f"reserved_GiB={s.get('reserved_bytes.all.current', 0) / 2**30:.2f}"
        )
    except Exception as e:
        parts.append(f"host_memory_stats=<error {e}>")
    parts.append("env CUDA_LAUNCH_BLOCKING=" + os.environ.get("CUDA_LAUNCH_BLOCKING", "unset"))
    logger.warning(" | ".join(parts))
    print(" | ".join(parts), flush=True)


def install_pin_memory_probe() -> None:
    """Wrap Tensor.pin_memory to count pinned bytes and log the failing tensor."""
    global _patched
    if not enabled() or _patched:
        return
    _patched = True
    orig = torch.Tensor.pin_memory

    def pin_memory(self, *args, **kwargs):
        try:
            out = orig(self, *args, **kwargs)
        except Exception:
            msg = (
                f"[ci-diag] pin_memory FAILED rank={os.environ.get('RANK', '?')} shape={tuple(self.shape)} "
                f"dtype={self.dtype} device={self.device} nbytes={self.numel() * self.element_size()}"
            )
            print(msg, flush=True)
            log_environment("at-pin_memory-failure")
            raise
        _pinned["calls"] += 1
        _pinned["bytes"] += self.numel() * self.element_size()
        return out

    torch.Tensor.pin_memory = pin_memory
