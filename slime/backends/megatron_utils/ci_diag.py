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


def _cgroup() -> str:
    """Memory limit/usage under cgroup v2 or v1, plus the cgroup path of this process."""
    files = {
        "v2.max": "/sys/fs/cgroup/memory.max",
        "v2.current": "/sys/fs/cgroup/memory.current",
        "v1.limit": "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "v1.usage": "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "v1.max_usage": "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
        "v1.failcnt": "/sys/fs/cgroup/memory/memory.failcnt",
    }
    out = [f"{k}={_read(v, 64)}" for k, v in files.items() if os.path.exists(v)]
    out.append("proc_cgroup=" + _read("/proc/self/cgroup", 300).replace("\n", ";"))
    return " ".join(out) if out else "<no cgroup memory files>"


def _numa() -> str:
    """Per-process NUMA/affinity info and per-node free memory."""
    keep = ("Cpus_allowed_list", "Mems_allowed_list", "VmLck", "VmPin", "VmRSS")
    status = [ln.split(":", 1) for ln in _read("/proc/self/status", 20000).splitlines() if ":" in ln]
    out = [f"{k}={' '.join(v.split())}" for k, v in status if k in keep]
    node_dir = "/sys/devices/system/node"
    try:
        for node in sorted(d for d in os.listdir(node_dir) if d.startswith("node")):
            free = [ln for ln in _read(f"{node_dir}/{node}/meminfo", 4000).splitlines() if "MemFree" in ln]
            if free:
                out.append(f"{node}.MemFree={free[0].split()[-2]}kB")
    except OSError as e:
        out.append(f"numa=<unavailable: {e}>")
    return " ".join(out)


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
    parts.append("cgroup: " + _cgroup())
    parts.append("numa: " + _numa())
    parts.append(f"meminfo: {_meminfo()}")
    parts.append(f"pin_memory calls={_pinned['calls']} bytes_GiB={_pinned['bytes'] / 2**30:.2f}")
    try:
        # Key names differ across torch versions, so report every non-zero byte counter as-is.
        s = torch.cuda.memory.host_memory_stats()
        nonzero = {k: round(v / 2**30, 2) for k, v in s.items() if "bytes" in k and v}
        parts.append(f"host_memory_stats_GiB={nonzero}")
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
