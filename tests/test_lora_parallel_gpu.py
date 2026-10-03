"""Real Megatron/Transformer Engine TP backward and torch_dist checkpoint parity.

Run in the training image: python tests/test_lora_parallel_gpu.py
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

NUM_GPUS = 2


def _worker(rank, rendezvous, checkpoint_root):
    # Never fall back to CPU substitutes in a real Megatron parity test.
    os.environ["SLIME_LORA_MULTI_GPU_TEST"] = "1"
    from megatron.core import dist_checkpointing, mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from test_lora_parallel import _check_network

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    mpu.initialize_model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(137)
    metadata = {"dp_cp_group": mpu.get_data_parallel_group(with_context_parallel=True)}
    try:
        for implementation in ["local", "te"]:
            for sequence_parallel in [False, True]:
                modules = _check_network(
                    sequence_parallel, fused_norm=implementation == "te", implementation=implementation
                )
                checkpoint = Path(checkpoint_root) / f"{implementation}-sp{sequence_parallel}"
                if rank == 0:
                    checkpoint.mkdir()
                dist.barrier()
                state = {}
                expected = {}
                for i, module in enumerate(modules):
                    prefix = f"projection.{i}."
                    state.update(module.sharded_state_dict(prefix=prefix, metadata=metadata))
                    for name in ("lora_A", "lora_B"):
                        expected[prefix + name] = getattr(module, name).detach().clone()
                dist_checkpointing.save(state, str(checkpoint))
                with torch.no_grad():
                    for module in modules:
                        module.lora_A.zero_()
                        module.lora_B.zero_()
                to_load = {}
                for i, module in enumerate(modules):
                    to_load.update(module.sharded_state_dict(prefix=f"projection.{i}.", metadata=metadata))
                restored = dist_checkpointing.load(to_load, str(checkpoint))
                for name, tensor in expected.items():
                    torch.testing.assert_close(restored[name], tensor, rtol=0, atol=0)
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.skipif(
    torch.cuda.device_count() < NUM_GPUS, reason="requires two CUDA GPUs and the Megatron training image"
)
def test_megatron_lora_backward_and_checkpoint(tmp_path):
    torch.multiprocessing.spawn(
        _worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)), nprocs=NUM_GPUS, join=True
    )


if __name__ == "__main__":
    if torch.cuda.device_count() < NUM_GPUS:
        raise SystemExit("LoRA GPU validation requires two visible CUDA GPUs; no validation was performed.")
    raise SystemExit(pytest.main([__file__]))
