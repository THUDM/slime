"""Check RDMA payloads after colocated rollout KV memory is remapped."""

import multiprocessing as mp
import subprocess
import sys

import torch
from torch_memory_saver import configure_subprocess

NUM_GPUS = 2


def _receive(pipe):
    assert pipe.poll(60), "Timed out waiting for the RDMA peer"
    return pipe.recv()


def _worker(rank, pipes):
    from mooncake.engine import TransferEngine
    from torch_memory_saver import torch_memory_saver

    torch.cuda.set_device(rank)
    pipe = pipes[rank]
    pipes[1 - rank].close()
    with torch_memory_saver.region(tag="kv_cache"):
        tensor = torch.empty(64 * 1024**2, dtype=torch.uint8, device="cuda")
    engine = TransferEngine()
    assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "rdma", "") == 0
    session = f"127.0.0.1:{engine.get_rpc_port()}"

    for cycle in range(5):
        if cycle:
            torch_memory_saver.pause("kv_cache")
            # Keep the former physical pages occupied while restoring the same VA.
            ballast = torch.full((2 * 1024**3,), 219, dtype=torch.uint8, device="cuda")
            torch_memory_saver.resume("kv_cache")
        tensor.fill_(cycle + 1 if rank == 0 else 0)
        torch.cuda.synchronize()
        assert engine.batch_register_memory([tensor.data_ptr()], [tensor.nbytes]) == 0
        pipe.send((session, tensor.data_ptr()))
        peer_session, peer_pointer = _receive(pipe)
        if rank == 0:
            assert (
                engine.batch_transfer_sync_write(peer_session, [tensor.data_ptr()], [peer_pointer], [tensor.nbytes])
                == 0
            )
            pipe.send("sent")
            assert _receive(pipe) == "checked"
        else:
            assert _receive(pipe) == "sent"
            torch.cuda.synchronize()
            assert torch.all(tensor == cycle + 1).item(), f"Corrupt RDMA payload after remap {cycle}"
            pipe.send("checked")
        assert engine.batch_unregister_memory([tensor.data_ptr()]) == 0
        pipe.send("released")
        assert _receive(pipe) == "released"
        if cycle:
            del ballast
            torch.cuda.empty_cache()
    pipe.close()
    print(f"Rank {rank}: all five RDMA transfers passed", flush=True)


def execute():
    with configure_subprocess():
        subprocess.run([sys.executable, __file__, "--workers"], check=True, timeout=180)


if __name__ == "__main__":
    if "--workers" in sys.argv:
        assert torch.cuda.device_count() >= NUM_GPUS
        pipes = mp.get_context("spawn").Pipe()
        torch.multiprocessing.spawn(_worker, args=(pipes,), nprocs=NUM_GPUS)
    else:
        execute()
