"""CUDA correctness check for the vendored verl linear cross entropy kernel."""

import os
import socket
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from slime.backends.megatron_utils.triton_log_probs import linear_cross_entropy


NUM_GPUS = 2


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)


def _linear_cross_entropy_case(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    temperature: float,
    dlogprob_scale: float,
    dentropy_scale: float,
) -> None:
    torch.manual_seed(1234)
    tokens, hidden_size, local_vocab = 257, 256, 1025
    hidden = torch.randn(tokens, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, local_vocab * world_size, (tokens,), device=device)
    labels[0] = rank * local_vocab
    if world_size > 1:
        labels[1] = ((rank + 1) % world_size) * local_vocab
    if dist.is_initialized():
        with torch.no_grad():
            dist.broadcast(hidden, 0)
            dist.broadcast(labels, 0)

    torch.manual_seed(4321 + rank)
    weight = torch.randn(local_vocab, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight_parts = [torch.empty_like(weight) for _ in range(world_size)]
    if dist.is_initialized():
        dist.all_gather(weight_parts, weight)
    else:
        weight_parts[0].copy_(weight)
    full_weight = torch.cat(weight_parts).detach().requires_grad_(True)
    reference_hidden = hidden.detach().requires_grad_(True)
    logits = (reference_hidden @ full_weight.T).float() / temperature
    reference_log_probs = -torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    probabilities = logits.softmax(dim=-1)
    reference_entropy = torch.logsumexp(logits, dim=-1) - (probabilities * logits).sum(dim=-1)
    torch.manual_seed(5678)
    dlogprobs = torch.randn(tokens, device=device, dtype=torch.float32) * dlogprob_scale
    dentropy = torch.randn(tokens, device=device, dtype=torch.float32) * dentropy_scale
    reference_log_probs, reference_entropy = torch.stack((reference_log_probs, reference_entropy), dim=-1).unbind(-1)
    reference_loss = (reference_log_probs * dlogprobs + reference_entropy * dentropy).sum()
    reference_loss.backward()

    log_probs, entropy = linear_cross_entropy(
        hidden, weight, labels, temperature, "none", None if world_size == 1 else dist.group.WORLD
    )
    log_probs, entropy = torch.stack((log_probs, entropy), dim=-1).unbind(-1)
    loss = (log_probs * dlogprobs + entropy * dentropy).sum()
    loss.backward()
    if dist.is_initialized():
        dist.all_reduce(hidden.grad)

    torch.testing.assert_close(log_probs, reference_log_probs, atol=5e-4, rtol=1e-5)
    torch.testing.assert_close(entropy, reference_entropy, atol=5e-4, rtol=1e-5)
    assert _relative_l2(hidden.grad, reference_hidden.grad) <= 0.01
    expected_weight_grad = full_weight.grad[rank * local_vocab : (rank + 1) * local_vocab]
    assert _relative_l2(weight.grad, expected_weight_grad) <= 0.01
    if dist.is_initialized():
        dist.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("dlogprob_scale,dentropy_scale", [(1.0, 0.0), (0.0, 0.01), (1.0, 0.01)])
def test_linear_cross_entropy_matches_bf16_output_boundary(
    temperature: float, dlogprob_scale: float, dentropy_scale: float
) -> None:
    torch.cuda.set_device(0)
    _linear_cross_entropy_case(
        rank=0,
        world_size=1,
        device=torch.device("cuda", 0),
        temperature=temperature,
        dlogprob_scale=dlogprob_scale,
        dentropy_scale=dentropy_scale,
    )


def _tp2_linear_cross_entropy_worker(rank: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    try:
        for temperature in (1.0, 0.7):
            for dlogprob_scale, dentropy_scale in ((1.0, 0.0), (0.0, 0.01), (1.0, 0.01)):
                _linear_cross_entropy_case(
                    rank=rank,
                    world_size=2,
                    device=torch.device("cuda", rank),
                    temperature=temperature,
                    dlogprob_scale=dlogprob_scale,
                    dentropy_scale=dentropy_scale,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_linear_cross_entropy_matches_bf16_output_boundary_tp2() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("TP=2 parity requires two CUDA devices")
    if not dist.is_nccl_available():
        pytest.skip("NCCL is required")

    import torch.multiprocessing as mp

    mp.spawn(_tp2_linear_cross_entropy_worker, args=(_free_port(),), nprocs=2, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("output", ["log_probs", "entropy"])
def test_linear_cross_entropy_backward_with_one_output_unused(output: str) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        pytest.skip("single-rank materialized zero-gradient check")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.manual_seed(1234)
    hidden = torch.randn(33, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(257, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 257, (33,), device=device)

    log_probs, entropy = linear_cross_entropy(hidden, weight, labels, 1.0, "none", None)
    loss = log_probs.mean() if output == "log_probs" else entropy.mean()
    loss.backward()

    assert hidden.grad is not None
    assert weight.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(weight.grad).all()


class _CopyToTensorParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor) -> torch.Tensor:
        return hidden

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        grad = grad.contiguous()
        dist.all_reduce(grad)
        return grad


class _GatherFromSequenceParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor) -> torch.Tensor:
        ctx.rank = dist.get_rank()
        ctx.local_tokens = hidden.size(0)
        parts = [torch.empty_like(hidden) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, hidden)
        return torch.cat(parts, dim=0)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        # ponytail: all_reduce+slice is the small equivalent of reduce_scatter for this 2-rank CI fixture.
        grad = grad.contiguous()
        dist.all_reduce(grad)
        return grad[ctx.rank * ctx.local_tokens : (ctx.rank + 1) * ctx.local_tokens].contiguous()


def _adapter_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    upstream: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_parts = [torch.empty_like(weight) for _ in range(dist.get_world_size())]
    dist.all_gather(weight_parts, weight)
    full_weight = torch.cat(weight_parts).detach().requires_grad_(True)
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    logits = (ref_hidden @ full_weight.T).float()
    log_probs = -torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    probs = logits.softmax(dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - (probs * logits).sum(dim=-1)
    ref_out = torch.stack((log_probs, entropy), dim=-1)
    (ref_out * upstream).sum().backward()
    vocab = weight.size(0)
    return ref_out, ref_hidden.grad, full_weight.grad[rank * vocab : (rank + 1) * vocab]


def _adapter_case(rank: int, sequence_parallel: bool) -> None:
    from slime.backends.megatron_utils.triton_log_probs import megatron as adapter

    device = torch.device("cuda", rank)
    adapter.mpu.get_tensor_model_parallel_group = lambda: dist.group.WORLD
    adapter.tensor_parallel = SimpleNamespace(
        copy_to_tensor_model_parallel_region=_CopyToTensorParallel.apply,
        gather_from_sequence_parallel_region=_GatherFromSequenceParallel.apply,
    )

    tokens, hidden_size, local_vocab = 64, 128, 257
    torch.manual_seed(1234)
    full_hidden = torch.randn(tokens, hidden_size, device=device, dtype=torch.bfloat16)
    labels = torch.randint(0, local_vocab * 2, (tokens,), device=device)
    labels[0] = 0
    labels[1] = local_vocab
    dist.broadcast(full_hidden, 0)
    dist.broadcast(labels, 0)
    torch.manual_seed(4321 + rank)
    weight = torch.randn(local_vocab, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)

    if sequence_parallel:
        local_tokens = tokens // 2
        hidden = full_hidden[rank * local_tokens : (rank + 1) * local_tokens].detach().clone().requires_grad_(True)
    else:
        hidden = full_hidden.detach().clone().requires_grad_(True)

    model = SimpleNamespace(
        config=Namespace(sequence_parallel=sequence_parallel),
        share_embeddings_and_output_weights=False,
        output_layer=SimpleNamespace(weight=weight, bias=None),
        _slime_triton_log_probs_args=Namespace(rollout_temperature=1.0),
    )
    out = adapter._fused_postprocess(model, hidden, labels).squeeze(0)
    upstream = torch.randn(tokens, 2, device=device)
    dist.broadcast(upstream, 0)
    logprob_upstream, entropy_upstream = upstream.unbind(-1)
    assert not logprob_upstream.is_contiguous()
    (out[:, 0] * logprob_upstream + out[:, 1] * entropy_upstream).sum().backward()

    ref_out, ref_hidden_grad, ref_weight_grad = _adapter_reference(full_hidden, weight, labels, upstream, rank)
    expected_hidden_grad = (
        ref_hidden_grad[rank * (tokens // 2) : (rank + 1) * (tokens // 2)] if sequence_parallel else ref_hidden_grad
    )
    torch.testing.assert_close(out, ref_out, atol=5e-4, rtol=1e-5)
    assert _relative_l2(hidden.grad, expected_hidden_grad) <= 0.01
    assert _relative_l2(weight.grad, ref_weight_grad) <= 0.01


def _adapter_tp2_worker(rank: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    try:
        _adapter_case(rank, sequence_parallel=False)
        _adapter_case(rank, sequence_parallel=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_megatron_adapter_hidden_and_weight_grads_tp2_sp_modes() -> None:
    pytest.importorskip("megatron.core")
    if torch.cuda.device_count() < 2:
        pytest.skip("TP=2 adapter parity requires two CUDA devices")
    if not dist.is_nccl_available():
        pytest.skip("NCCL is required")

    import torch.multiprocessing as mp

    mp.spawn(_adapter_tp2_worker, args=(_free_port(),), nprocs=2, join=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
