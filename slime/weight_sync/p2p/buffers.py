"""Registered storage that can be reused across completed transactions."""

import threading

from .transport import create_transport


def resolve_device(device):
    import torch

    device = torch.device(device or "cuda")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


class BufferPool:
    """Own registered buffers for one device and one P2P engine.

    Pass the same pool to successive publishers or receivers, then ``close`` it
    after the last transaction. Call ``trim`` between transactions to release
    idle GPU storage. Active buffers stay alive.
    Stable keys keep large buffers attached to the source that needs them.
    """

    def __init__(self, host, *, device=None, transport=None):
        self.device = resolve_device(device)
        self.transport = transport or create_transport(host, device=self.device)
        self._lock = threading.Lock()
        self._free = []
        self._leased = {}
        self._keys = {}
        self._closed = False

    def acquire(self, size, *, key=None):
        import torch

        with self._lock:
            if self._closed:
                raise RuntimeError("P2P buffer pool is closed")
            candidates = [
                (tensor.numel(), index)
                for index, tensor in enumerate(self._free)
                if self._keys[id(tensor)] == key and tensor.numel() >= size
            ]
            if candidates:
                _, index = min(candidates)
                tensor = self._free.pop(index)
            else:
                tensor = torch.empty(max(1, size), dtype=torch.uint8, device=self.device)
                self.transport.register(tensor)
                self._keys[id(tensor)] = key
            self._leased[id(tensor)] = tensor
            return tensor

    def release(self, tensor, *, cache=True):
        """Release a lease; ``cache=False`` also unregisters its storage."""
        with self._lock:
            if self._leased.get(id(tensor)) is not tensor:
                raise ValueError("Buffer is not leased from this P2P pool")
            if not cache:
                self.transport.unregister(tensor)
                del self._keys[id(tensor)]
            del self._leased[id(tensor)]
            if cache:
                self._free.append(tensor)

    def _trim(self):
        if self._free and hasattr(self.transport, "unregister_many"):
            self.transport.unregister_many(self._free)
            for tensor in self._free:
                del self._keys[id(tensor)]
            self._free.clear()
        else:
            while self._free:
                self.transport.unregister(self._free[-1])
                del self._keys[id(self._free.pop())]
        if not self._leased and hasattr(self.transport, "reset"):
            self.transport.reset()

    def trim(self):
        """Unregister and release idle buffers, keeping the pool reusable.

        Active transfers, including buffers quarantined after failure, retain
        their leases and cannot be freed here. Once no leases remain, reset
        cached remote registrations before another version reuses GPU addresses.
        """
        with self._lock:
            self._trim()

    def close(self):
        with self._lock:
            if self._leased:
                raise RuntimeError("Cannot close a P2P pool with active buffers")
            self._trim()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def select_pool(host, device, transport, pool):
    if pool is None:
        return BufferPool(host, device=device, transport=transport)
    if device is not None and resolve_device(device) != pool.device:
        raise ValueError("P2P pool and stream must use the same device")
    if transport is not None and transport is not pool.transport:
        raise ValueError("P2P pool and stream must use the same transport")
    return pool
