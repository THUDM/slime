import concurrent.futures
import ctypes
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime.weight_sync.p2p import BufferPool, Publisher, Receiver
from slime.weight_sync.p2p.stream import _recv, _send

NUM_GPUS = 0


class MemoryTransport:
    session = "in-process"

    def __init__(self):
        self.registered = set()

    def register(self, tensor):
        assert tensor.data_ptr() not in self.registered
        self.registered.add(tensor.data_ptr())

    def unregister(self, tensor):
        self.registered.remove(tensor.data_ptr())

    def read(self, session, tensor, address, size):
        ctypes.memmove(tensor.data_ptr(), address, size)


def test_default_cuda_device_is_resolved_in_the_calling_thread(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    receiver = Receiver([], version="7", consumer="a", host="127.0.0.1", transport=MemoryTransport())
    assert receiver.device == torch.device("cuda:7")


def receive(endpoint, consumer, *, version="7", slow=False):
    result = {}
    transport = MemoryTransport()
    with Receiver(
        [endpoint],
        version=version,
        consumer=consumer,
        host="127.0.0.1",
        device="cpu",
        transport=transport,
        timeout=2,
    ) as receiver:
        for batch in receiver:
            if slow:
                time.sleep(0.002)
            result.update((name, tensor.clone()) for name, tensor in batch)
    assert not transport.registered
    return result


def test_multiple_consumers_and_repeated_slot_reuse():
    transport = MemoryTransport()
    publisher = Publisher(
        "127.0.0.1",
        version="7",
        consumers=["a", "b"],
        capacity=64,
        device="cpu",
        transport=transport,
        timeout=2,
    )
    expected = {}
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            readers = [pool.submit(receive, publisher.endpoint, name, slow=(name == "b")) for name in ["a", "b"]]
            for i in range(30):
                # Noncontiguous, mixed dtypes, scalar and oversize tensors.
                batch = [
                    (f"matrix-{i}", torch.arange(60).reshape(6, 10).T + i),
                    (f"scalar-{i}", torch.tensor(float(i), dtype=torch.bfloat16)),
                ]
                expected.update(batch)
                publisher.publish(batch)
            publisher.finish()
            for reader in readers:
                actual = reader.result()
                assert actual.keys() == expected.keys()
                for name in expected:
                    assert torch.equal(actual[name], expected[name])
    finally:
        publisher.close()
    assert not transport.registered


def test_wrong_version_is_rejected_before_read():
    publisher = Publisher(
        "127.0.0.1",
        version="7",
        consumers=["a"],
        device="cpu",
        transport=MemoryTransport(),
    )
    try:
        with pytest.raises(ValueError, match="version"):
            receive(publisher.endpoint, "a", version="6")
    finally:
        publisher.close()


def test_large_group_is_split_but_individual_tensors_remain_complete():
    transport = MemoryTransport()
    publisher = Publisher("127.0.0.1", version="7", consumers=["a"], capacity=64, transport=transport, device="cpu")
    expected = [
        (name, torch.full((size,), index, dtype=torch.uint8))
        for index, (name, size) in enumerate([("a", 40), ("b", 40), ("oversized", 128), ("c", 40)])
    ]

    def read():
        batches = []
        with Receiver(
            [publisher.endpoint],
            version="7",
            consumer="a",
            host="127.0.0.1",
            device="cpu",
            transport=MemoryTransport(),
        ) as receiver:
            for batch in receiver:
                batches.append([(name, tensor.clone()) for name, tensor in batch])
        return batches

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = executor.submit(read)
            publisher.publish(expected)
            publisher.finish()
            batches = result.result()
        assert [len(batch) for batch in batches] == [1, 1, 1, 1]
        for batch, (name, tensor) in zip(batches, expected, strict=True):
            assert batch[0][0] == name and torch.equal(batch[0][1], tensor)
    finally:
        publisher.close()


def test_abort_unblocks_waiting_consumer():
    publisher = Publisher(
        "127.0.0.1",
        version="7",
        consumers=["a"],
        device="cpu",
        transport=MemoryTransport(),
        timeout=2,
    )
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(receive, publisher.endpoint, "a")
            deadline = time.monotonic() + 2
            while not publisher._connected:
                assert time.monotonic() < deadline
                time.sleep(0.001)
            publisher.abort("conversion failed")
            with pytest.raises(ConnectionError):
                result.result(timeout=2)
    finally:
        publisher.close()


def test_independent_sources_do_not_head_of_line_block():
    publishers = [
        Publisher(
            "127.0.0.1",
            version="7",
            consumers=["a"],
            capacity=64,
            device="cpu",
            transport=MemoryTransport(),
            timeout=2,
        )
        for _ in range(2)
    ]
    received = threading.Event()

    def read():
        with Receiver(
            [p.endpoint for p in publishers],
            version="7",
            consumer="a",
            host="127.0.0.1",
            device="cpu",
            transport=MemoryTransport(),
            timeout=2,
        ) as receiver:
            result = []
            for batch in receiver:
                result.extend(name for name, _ in batch)
                received.set()
            return result

    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            reader = pool.submit(read)
            publishers[1].publish([("second", torch.ones(1))])
            assert received.wait(1)
            publishers[0].publish([("first", torch.ones(1))])
            finishes = [pool.submit(p.finish) for p in publishers]
            assert reader.result() == ["second", "first"]
            for result in finishes:
                result.result()
    finally:
        for publisher in publishers:
            publisher.close()


def test_unknown_consumer_cannot_join_or_acknowledge():
    import socket

    publisher = Publisher("127.0.0.1", version="7", consumers=["a"], device="cpu", transport=MemoryTransport())
    try:
        with socket.create_connection(("127.0.0.1", publisher.endpoint["port"])) as connection:
            with connection.makefile("rwb") as stream:
                _send(stream, {**publisher.endpoint, "consumer": "intruder"})
                with pytest.raises(RuntimeError, match="Unknown"):
                    _recv(stream)
        assert not publisher._connected
        assert publisher._error is None
    finally:
        publisher.close()


def test_source_cannot_overwrite_an_unacknowledged_slot():
    publisher = Publisher(
        "127.0.0.1", version="7", consumers=["a"], capacity=64, device="cpu", transport=MemoryTransport(), timeout=0.1
    )
    try:
        publisher.publish([("a", torch.full((4,), 1))])
        publisher.publish([("b", torch.full((4,), 2))])
        before = publisher._slots[0].tensor.clone()
        with pytest.raises(TimeoutError):
            publisher.publish([("c", torch.full((4,), 3))])
        assert torch.equal(publisher._slots[0].tensor, before)
    finally:
        publisher.close()


def test_published_empty_tensors_preserve_shapes():
    publisher = Publisher(
        "127.0.0.1", version="7", consumers=["a"], capacity=64, device="cpu", transport=MemoryTransport(), timeout=2
    )
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            reader = pool.submit(receive, publisher.endpoint, "a")
            publisher.publish([("empty", torch.empty(3, 0, dtype=torch.bfloat16))])
            publisher.finish()
            tensor = reader.result()["empty"]
            assert tensor.shape == (3, 0) and tensor.dtype == torch.bfloat16
    finally:
        publisher.close()


def test_varying_buckets_reuse_one_receive_registration():
    class RecordingTransport(MemoryTransport):
        def __init__(self):
            super().__init__()
            self.sizes = []

        def register(self, tensor):
            self.sizes.append(tensor.numel())
            super().register(tensor)

    transport = RecordingTransport()
    publisher = Publisher(
        "127.0.0.1", version="7", consumers=["a"], capacity=4096, device="cpu", transport=MemoryTransport(), timeout=2
    )

    def read():
        with Receiver(
            [publisher.endpoint], version="7", consumer="a", host="127.0.0.1", device="cpu", transport=transport
        ) as receiver:
            for batch in receiver:
                name, tensor = batch[0]
                assert torch.equal(tensor, torch.full((int(name),), 7, dtype=torch.uint8))

    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            reader = pool.submit(read)
            for size in [1, 65, 300, 1000, 2049, 4096]:
                publisher.publish([(str(size), torch.full((size,), 7, dtype=torch.uint8))])
            publisher.finish()
            reader.result()
        assert transport.sizes == [4096]
        assert not transport.registered
    finally:
        publisher.close()


@pytest.mark.parametrize("trim_between_updates", [False, True])
def test_repeated_updates_with_retained_or_released_buffers(trim_between_updates):
    class RecordingTransport(MemoryTransport):
        def __init__(self):
            super().__init__()
            self.registrations = 0

        def register(self, tensor):
            self.registrations += 1
            super().register(tensor)

    source_transport, receiver_transport = RecordingTransport(), RecordingTransport()
    with BufferPool("127.0.0.1", device="cpu", transport=source_transport) as source_pool:
        with BufferPool("127.0.0.1", device="cpu", transport=receiver_transport) as receiver_pool:
            for version in range(3):
                publisher = Publisher("127.0.0.1", version=version, consumers=["a"], capacity=64, pool=source_pool)

                def read(publisher=publisher, version=version):
                    with Receiver(
                        [publisher.endpoint], version=version, consumer="a", host="127.0.0.1", pool=receiver_pool
                    ) as receiver:
                        received = []
                        for batch in receiver:
                            name, tensor = batch[0]
                            assert torch.equal(tensor, torch.full((8,), version * 10 + int(name)))
                            received.append(name)
                        assert received == [str(i) for i in range(4)]

                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = executor.submit(read)
                        for step in range(4):
                            publisher.publish([(str(step), torch.full((8,), version * 10 + step))])
                        publisher.finish()
                        result.result()
                finally:
                    publisher.close()
                if trim_between_updates:
                    source_pool.trim()
                    receiver_pool.trim()
                    assert not source_transport.registered and not receiver_transport.registered
                    assert not source_pool._free and not receiver_pool._free
            assert source_transport.registrations == (6 if trim_between_updates else 2)
            assert receiver_transport.registrations == (3 if trim_between_updates else 1)
    assert not source_transport.registered and not receiver_transport.registered


def test_pool_cannot_free_active_memory_or_release_a_buffer_twice():
    transport = MemoryTransport()
    pool = BufferPool("127.0.0.1", device="cpu", transport=transport)
    tensor = pool.acquire(64)
    with pytest.raises(RuntimeError, match="active buffers"):
        pool.close()
    assert tensor.data_ptr() in transport.registered
    pool.release(tensor)
    with pytest.raises(ValueError, match="not leased"):
        pool.release(tensor)
    pool.close()
    assert not transport.registered


def test_trim_releases_idle_storage_but_preserves_active_transfer():
    import weakref

    class ResettingTransport(MemoryTransport):
        resets = 0

        def unregister_many(self, tensors):
            for tensor in tensors:
                self.unregister(tensor)

        def reset(self):
            assert not self.registered
            self.resets += 1

    transport = ResettingTransport()
    with BufferPool("127.0.0.1", device="cpu", transport=transport) as pool:
        active = pool.acquire(64)
        idle = pool.acquire(128)
        released = weakref.ref(idle)
        pool.release(idle)
        del idle
        pool.trim()
        assert released() is None
        assert transport.registered == {active.data_ptr()}
        assert transport.resets == 0
        assert not pool._free
        pool.release(active)
        pool.trim()
        assert not transport.registered and not pool._keys
        assert transport.resets == 1


def test_uncached_release_frees_storage_while_other_leases_remain():
    import weakref

    transport = MemoryTransport()
    with BufferPool("127.0.0.1", device="cpu", transport=transport) as pool:
        active = pool.acquire(64)
        old = pool.acquire(128)
        old_ref = weakref.ref(old)
        pool.release(old, cache=False)
        del old
        assert old_ref() is None and not pool._free
        assert transport.registered == {active.data_ptr()}
        pool.release(active)


def test_growing_sources_keep_their_buffers_when_connection_order_changes():
    transport = MemoryTransport()
    with BufferPool("127.0.0.1", device="cpu", transport=transport) as pool:
        for version in range(4):
            # One source grows after its first bucket. Changing connection
            # order must not give its large MR to the small source and force
            # another large allocation on each version.
            keys = ["growing", "small"] if version % 2 else ["small", "growing"]
            buffers = {key: pool.acquire(64, key=key) for key in keys}
            pool.release(buffers["growing"])
            buffers["growing"] = pool.acquire(4096, key="growing")
            for tensor in buffers.values():
                pool.release(tensor)
            assert len(transport.registered) == 3


def test_native_read_failure_retains_destination_until_process_exit():
    from slime.weight_sync.p2p.stream import _QUARANTINED

    class FailedRead(MemoryTransport):
        def read(self, *args):
            raise RuntimeError("native transfer timed out")

    sender = Publisher("127.0.0.1", version=1, consumers=["a"], device="cpu", transport=MemoryTransport(), capacity=64)
    transport = FailedRead()
    receiver = Receiver(
        [sender.endpoint], version=1, consumer="a", host="127.0.0.1", device="cpu", transport=transport
    )
    try:
        sender.publish([("weight", torch.ones(4))])
        with pytest.raises(RuntimeError, match="native transfer"):
            with receiver:
                list(receiver)
        receiver.pool.trim()
        assert receiver in _QUARANTINED
        assert len(transport.registered) == 1
        assert receiver._unsafe_buffers[0].data_ptr() in transport.registered
        with pytest.raises(RuntimeError, match="active buffers"):
            receiver.pool.close()
    finally:
        sender.close()


@pytest.mark.parametrize("cuda_failure", [False, True])
def test_early_loader_exit_does_not_release_buffers_before_cuda_finishes(monkeypatch, cuda_failure):
    from slime.weight_sync.p2p.stream import _QUARANTINED

    sender = Publisher("127.0.0.1", version=1, consumers=["a"], device="cpu", transport=MemoryTransport(), capacity=64)
    transport = MemoryTransport()
    pool = BufferPool("127.0.0.1", device="cpu", transport=transport)
    receiver = Receiver([sender.endpoint], version=1, consumer="a", host="127.0.0.1", pool=pool)
    synchronized = []

    def synchronize(device):
        assert device == torch.device("cuda:0")
        assert pool._leased and transport.registered
        synchronized.append(True)
        if cuda_failure:
            raise RuntimeError("CUDA work failed")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    iterator = None
    try:
        sender.publish([("weight", torch.ones(4))])
        receiver.__enter__()
        iterator = iter(receiver)
        next(iterator)
        # Keep the iterator alive, as a loader retaining its input can do.
        # Emulate pending CUDA work after the CPU transport has delivered data.
        receiver.device = torch.device("cuda:0")
        if cuda_failure:
            with pytest.raises(RuntimeError, match="CUDA work failed"):
                receiver.__exit__(None, None, None)
            pool.trim()
            assert receiver in _QUARANTINED
            assert pool._leased and transport.registered
        else:
            receiver.__exit__(None, None, None)
            pool.close()
            assert not transport.registered
        assert synchronized == [True]
    finally:
        if iterator is not None:
            iterator.close()
        sender.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
