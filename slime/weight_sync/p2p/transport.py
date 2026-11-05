"""Data-plane contract used by the P2P stream and registered buffer pool."""

from typing import Protocol

import torch


class Transport(Protocol):
    @property
    def session(self) -> str: ...

    def register(self, tensor: torch.Tensor) -> None: ...

    def unregister(self, tensor: torch.Tensor) -> None: ...

    def read(self, session: str, tensor: torch.Tensor, address: int, size: int) -> None:
        """Read into registered storage; return only when the bytes are ready.

        On failure the stream retains the destination because DMA may still
        be in flight. Implementations must raise rather than return an error
        code. Stream users synchronize CUDA before publishing/reusing buffers.
        """
        ...


def create_transport(host: str, *, device=None) -> Transport:
    # The current provider is optional and imported only when a pool is opened.
    # Publisher/Receiver also accept an explicit transport for other providers.
    from .mooncake import MooncakeTransport

    return MooncakeTransport(host, device=device)
