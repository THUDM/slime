"""Acknowledged P2P tensor streams independent of training and serving runtimes."""

from .buffers import BufferPool
from .stream import Publisher, Receiver

__all__ = ["BufferPool", "Publisher", "Receiver"]
