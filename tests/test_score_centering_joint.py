"""Dense-oracle and NCCL checks for shared sampled/head scores and entropy."""

import itertools
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from slime.utils.ppo_utils import calculate_joint_log_probs_and_entropy

NUM_GPUS = 2


def check_case(device, dtype, rows, chunk_size, temperature, entropy_grad, group=None, entropy_only=False):
    torch.manual_seed(103)
    rank = 0 if group is None else dist.get_rank(group)
    world = 1 if group is None else dist.get_world_size(group)
    full = (torch.randn(rows, 32, device=device) * 3).to(dtype)
    # Sampled tokens include head members, out-of-head tokens and TP boundaries.
    targets = (torch.arange(rows, device=device) * 7) % 32
    heads = torch.stack((targets, targets, (targets + 16) % 32, (targets + 1) % 32), dim=-1)
    shard = full[:, rank * (32 // world) : (rank + 1) * (32 // world)].detach().requires_grad_()
    before = shard.detach().clone()
    sampled, head, entropy = calculate_joint_log_probs_and_entropy(
        shard, targets, heads, group, chunk_size, temperature, entropy_grad
    )
    reference = full.float().detach().requires_grad_()
    logp = (reference / temperature).log_softmax(-1)
    expected = (logp.gather(-1, targets[:, None]).squeeze(-1), logp.gather(-1, heads), -(logp.exp() * logp).sum(-1))
    for actual, want in zip((sampled, head, entropy), expected, strict=True):
        torch.testing.assert_close(actual, want, atol=8e-6, rtol=3e-6)
    assert entropy.requires_grad == entropy_grad
    weights = [torch.randn_like(value) for value in expected]
    if entropy_only:
        loss = (entropy * weights[2]).sum()
        ref_loss = (expected[2] * weights[2]).sum()
    else:
        loss = (sampled * weights[0]).sum() + (head * weights[1]).sum()
        ref_loss = (expected[0] * weights[0]).sum() + (expected[1] * weights[1]).sum()
        if entropy_grad:
            loss = loss + (entropy * weights[2]).sum()
            ref_loss = ref_loss + (expected[2] * weights[2]).sum()
    gradient = torch.autograd.grad(loss, shard)[0]
    dense_gradient = torch.autograd.grad(ref_loss, reference)[0][:, rank * (32 // world) : (rank + 1) * (32 // world)]
    tol = 0.016 if dtype == torch.bfloat16 else 8e-6
    torch.testing.assert_close(gradient.float(), dense_gradient, atol=tol, rtol=tol)
    torch.testing.assert_close(shard, before, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows,chunk_size", [(0, 3), (7, -1), (7, 3)])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.3])
@pytest.mark.parametrize("entropy_grad", [False, True])
def test_joint_dense_oracle(device, dtype, rows, chunk_size, temperature, entropy_grad):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    check_case(device, dtype, rows, chunk_size, temperature, entropy_grad)


def _nccl_worker(rank, port):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        for dtype, rows, chunk, temperature, entropy_grad in itertools.product(
            [torch.float32, torch.bfloat16], [0, 7], [-1, 3], [0.7, 1.0], [False, True]
        ):
            check_case(f"cuda:{rank}", dtype, rows, chunk, temperature, entropy_grad, dist.group.WORLD)
        check_case(f"cuda:{rank}", torch.float32, 7, 3, 0.7, True, dist.group.WORLD, entropy_only=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two GPUs required")
def test_joint_tensor_parallel_nccl():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_nccl_worker, args=(port,), nprocs=2, join=True)
