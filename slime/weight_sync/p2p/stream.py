"""A bounded, acknowledged stream of named tensors.

TCP carries JSON descriptors and acknowledgements only. Registered GPU buffers
carry the payload through RDMA. A source slot cannot be reused until every
consumer acknowledges its read. Receivers keep one buffer per source and drain
ready sources independently, so a slow source cannot block the other sources'
training collectives. The application controls when a complete version becomes
visible to inference.
"""

from __future__ import annotations

import json
import math
import queue
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field

from .buffers import select_pool

_MAX_MESSAGE = 16 * 1024**2
_ALIGN = 64
# Unacknowledged remote reads can outlive a failed control connection. These
# buffers are deliberately retained until process exit, never reused.
_QUARANTINED = []


def _send(stream, value):
    payload = json.dumps(value, separators=(",", ":")).encode() + b"\n"
    if len(payload) > _MAX_MESSAGE:
        raise ValueError("P2P control message is too large")
    stream.write(payload)
    stream.flush()


def _recv(stream):
    payload = stream.readline(_MAX_MESSAGE + 1)
    if not payload:
        raise ConnectionError("P2P peer disconnected")
    if len(payload) > _MAX_MESSAGE or not payload.endswith(b"\n"):
        raise ValueError("Invalid P2P control frame")
    message = json.loads(payload)
    if "error" in message:
        raise RuntimeError(message["error"])
    return message


def tensor_layout(named_tensors):
    entries, size = [], 0
    for name, tensor in named_tensors:
        nbytes = tensor.numel() * tensor.element_size()
        entries.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "offset": size,
                "nbytes": nbytes,
            }
        )
        size += (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    return entries, size


@dataclass
class _Slot:
    tensor: object = None
    sequence: int = -1
    pending: set = field(default_factory=set)
    message: dict | None = None


class Publisher:
    """Publish on the caller's CUDA stream; clients run on background threads.

    ``capacity`` is the normal bucket size. An indivisible model tensor may use
    a larger slot. Two source slots allow conversion to overlap remote reads.
    ``consumers`` is an exact set, not a count: duplicate/stale clients cannot
    acknowledge another rank's read.
    """

    def __init__(
        self,
        host,
        *,
        version,
        consumers,
        capacity=64 * 1024**2,
        device=None,
        timeout=120,
        transport=None,
        pool=None,
    ):
        if capacity <= 0 or timeout <= 0 or not consumers or len(set(consumers)) != len(consumers):
            raise ValueError("Positive capacity/timeout and distinct consumers are required")
        self.version, self.consumers = str(version), frozenset(consumers)
        self._owns_pool = pool is None
        self.pool = select_pool(host, device, transport, pool)
        self.capacity, self.device, self.timeout = capacity, self.pool.device, timeout
        self.transport = self.pool.transport
        self._token = secrets.token_hex(24)
        self._condition = threading.Condition()
        self._slots = [_Slot(), _Slot()]
        self._sequence = 0
        self._eof = None
        self._finished = set()
        self._connected = set()
        self._sockets = set()
        self._threads = []
        self._error = None
        self._closed = False
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, 0))
        self._listener.listen(max(16, len(consumers)))
        self._listener.settimeout(0.2)
        self.endpoint = {
            "host": host,
            "port": self._listener.getsockname()[1],
            "token": self._token,
            "version": self.version,
            "protocol": 1,
        }
        self._accept_thread = threading.Thread(target=self._accept, daemon=True)
        self._accept_thread.start()

    def _check(self):
        if self._error is not None:
            raise RuntimeError(f"P2P publication failed: {self._error}")
        if self._closed:
            raise RuntimeError("P2P publisher is closed")

    def _wait(self, predicate):
        deadline = time.monotonic() + self.timeout
        while not predicate():
            self._check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for P2P peers")
            self._condition.wait(min(0.2, remaining))
        self._check()

    def _accept(self):
        while not self._closed:
            try:
                connection, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            connection.settimeout(self.timeout)
            with self._condition:
                if self._closed:
                    connection.close()
                    return
                self._sockets.add(connection)
            thread = threading.Thread(target=self._serve, args=(connection,), daemon=True)
            self._threads.append(thread)
            thread.start()

    def _serve(self, connection):
        authenticated = False
        try:
            with connection, connection.makefile("rwb") as stream:
                hello = _recv(stream)
                consumer = hello.get("consumer")
                with self._condition:
                    if (
                        hello.get("token") != self._token
                        or hello.get("version") != self.version
                        or hello.get("protocol") != 1
                        or consumer not in self.consumers
                        or consumer in self._connected
                    ):
                        _send(
                            stream,
                            {"error": "Unknown, duplicate, or stale P2P consumer"},
                        )
                        return
                    authenticated = True
                    self._connected.add(consumer)
                sequence = 0
                while True:
                    with self._condition:
                        slot = self._slots[sequence % 2]
                        self._wait(
                            lambda slot=slot, sequence=sequence: slot.sequence == sequence or self._eof == sequence
                        )
                        eof = self._eof == sequence
                        message = {"sequence": sequence, "eof": True} if eof else slot.message
                    _send(stream, message)
                    ack = _recv(stream)
                    if ack != {"ack": sequence}:
                        raise ValueError(f"Invalid acknowledgement for sequence {sequence}: {ack}")
                    with self._condition:
                        if eof:
                            self._finished.add(consumer)
                        else:
                            slot.pending.remove(consumer)
                        self._condition.notify_all()
                    if eof:
                        return
                    sequence += 1
        except Exception as exc:
            if authenticated and not self._closed:
                self.abort(exc)
        finally:
            with self._condition:
                self._sockets.discard(connection)

    def publish(self, named_tensors):
        """Publish tensors in capacity-sized batches without splitting a tensor."""
        if self._eof is not None:
            raise RuntimeError("Cannot publish after finishing a P2P stream")
        batch, size = [], 0
        for name, tensor in named_tensors:
            nbytes = tensor.numel() * tensor.element_size()
            padded = (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
            if batch and size + padded > self.capacity:
                self._publish_batch(batch)
                batch, size = [], 0
            batch.append((name, tensor))
            size += padded
        if batch:
            self._publish_batch(batch)

    def _publish_batch(self, named_tensors):
        import torch

        named_tensors = list(named_tensors)
        if not named_tensors:
            return
        entries, size = tensor_layout(named_tensors)
        slot = self._slots[self._sequence % 2]
        with self._condition:
            self._wait(lambda: not slot.pending)
        if slot.tensor is None or slot.tensor.numel() < size:
            if slot.tensor is not None:
                self.pool.release(slot.tensor)
            slot.tensor = None
            slot.tensor = self.pool.acquire(max(self.capacity, size), key=("publisher", self._sequence % 2))
        for (_, tensor), entry in zip(named_tensors, entries, strict=True):
            target = slot.tensor.narrow(0, entry["offset"], entry["nbytes"])
            target.view(tensor.dtype).view(tensor.shape).copy_(tensor)
        if slot.tensor.is_cuda:
            torch.cuda.current_stream(slot.tensor.device).synchronize()
        with self._condition:
            self._check()
            slot.sequence = self._sequence
            slot.pending = set(self.consumers)
            slot.message = {
                "sequence": self._sequence,
                "session": self.transport.session,
                "address": slot.tensor.data_ptr(),
                "size": size,
                "capacity": slot.tensor.numel(),
                "tensors": entries,
            }
            self._sequence += 1
            self._condition.notify_all()

    def finish(self):
        with self._condition:
            self._eof = self._sequence
            self._condition.notify_all()
            self._wait(lambda: self._finished == self.consumers)

    def abort(self, error):
        with self._condition:
            if self._error is None:
                self._error = str(error)
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            sockets = list(self._sockets)
        self._listener.close()
        for connection in sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._accept_thread.join()
        for thread in self._threads:
            thread.join(timeout=self.timeout)
        # After an unacknowledged read, remote DMA may still target a source
        # buffer. Keep those allocations registered and alive on failure.
        for slot in self._slots:
            if slot.tensor is not None and not slot.pending:
                self.pool.release(slot.tensor)
                slot.tensor = None
        if any(slot.pending for slot in self._slots):
            _QUARANTINED.append(self)
        elif self._owns_pool:
            self.pool.close()


class Receiver:
    """Pull sources concurrently and yield ready tensor batches.

    One registered receive buffer per source bounds memory and lets unrelated
    sources make progress. Each yielded batch remains valid until iteration
    advances. Callers retaining tensors beyond that point must copy them.
    Keep endpoint order stable when reusing a pool across versions.
    """

    def __init__(
        self,
        endpoints,
        *,
        version,
        consumer,
        host,
        device=None,
        timeout=120,
        transport=None,
        pool=None,
    ):
        self.endpoints, self.version, self.consumer = (
            list(endpoints),
            str(version),
            consumer,
        )
        self._owns_pool = pool is None
        self.pool = select_pool(host, device, transport, pool)
        self.device, self.timeout = self.pool.device, timeout
        self.transport = self.pool.transport
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._threads = []
        self._connections = []
        self._buffers = []
        self._unsafe_buffers = []
        self._lock = threading.Lock()

    def _read_source(self, index, endpoint):
        import torch

        buffer = None
        try:
            device = torch.device(self.device)
            if device.type == "cuda":
                torch.cuda.set_device(device)
            if endpoint.get("version") != self.version or endpoint.get("protocol") != 1:
                raise ValueError("P2P source version/protocol mismatch")
            connection = socket.create_connection((endpoint["host"], endpoint["port"]), self.timeout)
            with self._lock:
                if self._stop.is_set():
                    connection.close()
                    return
                self._connections.append(connection)
            with connection, connection.makefile("rwb") as stream:
                _send(stream, {**endpoint, "consumer": self.consumer})
                sequence = 0
                while not self._stop.is_set():
                    message = _recv(stream)
                    if message.get("sequence") != sequence:
                        raise ValueError("P2P source sequence mismatch")
                    if message.get("eof"):
                        _send(stream, {"ack": sequence})
                        break
                    size = message["size"]
                    capacity = message.get("capacity", size)
                    if not isinstance(size, int) or size < 0 or not isinstance(capacity, int) or capacity < size:
                        raise ValueError("Invalid P2P buffer size")
                    tensors = []
                    tensor = None
                    if buffer is None or buffer.numel() < size:
                        if buffer is not None:
                            # This is a local RDMA read destination; no peer
                            # caches its rkey. The previous batch is consumed,
                            # so discard its smaller buffer before growing.
                            self.pool.release(buffer, cache=False)
                        buffer = None
                        # Buckets vary slightly in size. Register once at the
                        # sender's capacity instead of repeatedly growing an MR
                        # for each larger bucket on every receiver and HCA.
                        buffer = self.pool.acquire(max(1, capacity), key=("receiver", index))
                    for entry in message["tensors"]:
                        dtype = getattr(torch, entry["dtype"])
                        nbytes = math.prod(entry["shape"]) * dtype.itemsize
                        offset = entry["offset"]
                        if nbytes != entry["nbytes"] or offset < 0 or offset + nbytes > size:
                            raise ValueError("Invalid P2P tensor bounds")
                        tensor = buffer.narrow(0, offset, nbytes).view(dtype).view(entry["shape"])
                        tensors.append((entry["name"], tensor))
                    if size:
                        try:
                            self.transport.read(message["session"], buffer, message["address"], size)
                        except Exception:
                            # An error/timeout from a native transfer API does
                            # not prove DMA has stopped. Keep its destination
                            # registered and never reuse it after a failed read.
                            with self._lock:
                                self._unsafe_buffers.append(buffer)
                            buffer = None
                            raise
                    _send(stream, {"ack": sequence})
                    consumed = threading.Event()
                    self._queue.put((tensors, consumed))
                    while not consumed.wait(0.1):
                        if self._stop.is_set():
                            return
                    sequence += 1
            self._queue.put(None)
        except Exception as exc:
            self._queue.put(exc)
        finally:
            if buffer is not None:
                with self._lock:
                    self._buffers.append(buffer)

    def __enter__(self):
        for index, endpoint in enumerate(self.endpoints):
            thread = threading.Thread(target=self._read_source, args=(index, endpoint), daemon=True)
            self._threads.append(thread)
            thread.start()
        return self

    def __iter__(self):
        import torch

        remaining = len(self.endpoints)
        while remaining:
            try:
                item = self._queue.get(timeout=self.timeout)
            except queue.Empty as exc:
                raise TimeoutError("Timed out receiving P2P weights") from exc
            if item is None:
                remaining -= 1
                continue
            if isinstance(item, Exception):
                raise item
            tensors, consumed = item
            try:
                yield tensors
            finally:
                # The model loader may enqueue asynchronous device copies.
                # Complete those reads before a worker overwrites its buffer.
                if tensors and tensors[0][1].is_cuda:
                    torch.cuda.current_stream(tensors[0][1].device).synchronize()
                consumed.set()

    def __exit__(self, *exc):
        self._stop.set()
        with self._lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for thread in self._threads:
            thread.join(timeout=self.timeout)
        active = any(thread.is_alive() for thread in self._threads)
        if active or self._unsafe_buffers:
            _QUARANTINED.append(self)
            if active:
                raise TimeoutError("P2P read still active; receive buffers retained")
        try:
            if self.device.type == "cuda":
                import torch

                # A loader can stop early while retaining its iterator, so
                # __iter__'s finally block may not have run. Complete device
                # reads before releasing storage even on an exceptional exit.
                torch.cuda.synchronize(self.device)
        except Exception:
            _QUARANTINED.append(self)
            raise
        for buffer in self._buffers:
            self.pool.release(buffer)
        self._buffers.clear()
        if self._owns_pool and not self._unsafe_buffers:
            self.pool.close()
