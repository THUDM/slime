"""CUDA parity test for fused PPO log-probability and entropy calculation."""

from __future__ import annotations

import os
import socket

import pytest
import torch

from slime.utils.ppo_utils import calculate_log_probs_and_entropy


NUM_GPUS = 2

# Megatron's JIT fused CE can differ from the same Python-level expression by
# one fp32 ulp in the unmasked path.
FORWARD_ATOL = 1e-7
FORWARD_RTOL = 0.0
# Entropy values are O(1) in this parity fixture; allow a small difference from
# the memory-saving CUDA reduction without relaxing log-prob parity.
ENTROPY_FORWARD_ATOL = 1e-4
BACKWARD_ATOL = 1e-8
BACKWARD_RTOL = 0.0
# The combined logits gradient includes the entropy branch when entropy has grad;
# log-prob-only gradients still use BACKWARD_ATOL.
ENTROPY_BACKWARD_ATOL = 1e-6

PARITY_SCENARIOS = [
    (-1, False, False, False),
    (-1, False, True, False),
    (-1, False, True, True),
    (-1, True, False, False),
    (-1, True, True, False),
    (-1, True, True, True),
    (2, False, False, False),
    (2, False, True, False),
    (2, False, True, True),
    (2, True, False, False),
    (2, True, True, False),
    (2, True, True, True),
]


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _full_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 0.5, -1.0],
            [4.0, 1.0, 0.5, 2.0, 3.0, 0.0],
            [-1.0, 3.0, 2.0, 0.0, 1.0, 5.0],
            [0.2, -0.4, 1.7, -2.0, 3.3, 0.0],
        ],
        dtype=torch.float32,
    )


def _keep_mask() -> torch.Tensor:
    return torch.tensor(
        [
            [False, True, False, True, False, False],  # target 5 is absent.
            [False, True, False, False, True, False],  # target 0 is absent.
            [True, False, True, False, False, True],  # target 3 is absent.
            [False, True, False, True, False, False],  # target 2 is absent.
        ],
        dtype=torch.bool,
    )


def _weighted_loss(
    log_probs: torch.Tensor,
    entropy: torch.Tensor | None,
    *,
    logprob_weights: torch.Tensor,
    entropy_weights: torch.Tensor | None,
) -> torch.Tensor:
    loss = (log_probs.squeeze(-1) * logprob_weights).sum()
    if entropy is not None and entropy_weights is not None:
        loss = loss + (entropy * entropy_weights).sum()
    return loss


def _legacy_compute_log_probs(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    process_group,
    keep_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    if keep_mask is not None:
        keep_mask = keep_mask.clone()
        vocab_local = keep_mask.size(-1)
        vocab_start = process_group.rank() * vocab_local
        local_tokens = tokens - vocab_start
        on_shard = (local_tokens >= 0) & (local_tokens < vocab_local)
        rows = torch.nonzero(on_shard, as_tuple=False).squeeze(-1)
        if rows.numel() > 0:
            keep_mask[rows, local_tokens[rows]] = True
        logits = logits.masked_fill(~keep_mask, float("-inf"))

    return -fused_vocab_parallel_cross_entropy(logits.unsqueeze(1), tokens.unsqueeze(1), process_group)


class _LegacyVocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group) -> torch.Tensor:
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        torch.distributed.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = (softmax_logits * vocab_parallel_logits).sum(dim=-1, keepdim=True)
        torch.distributed.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        grad_input = softmax_logits * (sum_softmax_times_logits - vocab_parallel_logits)
        grad_input = grad_input * grad_output.unsqueeze(dim=-1)
        return grad_input, None


def _legacy_compute_entropy_from_logits(logits: torch.Tensor, process_group) -> torch.Tensor:
    return _LegacyVocabParallelEntropy.apply(logits, process_group)


def _assert_legacy_parity(
    *,
    process_group,
    device: torch.device,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    keep_mask: torch.Tensor | None,
    chunk_size: int,
    with_entropy: bool,
    entropy_has_grad: bool,
) -> None:
    log_probs, entropy = calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=process_group,
        with_entropy=with_entropy,
        chunk_size=chunk_size,
        log_prob_keep_mask=keep_mask,
        with_entropy_grad=entropy_has_grad,
    )

    legacy_logits = logits.detach().clone().requires_grad_()
    legacy_log_probs = _legacy_compute_log_probs(legacy_logits.clone(), tokens, process_group, keep_mask=keep_mask)

    torch.testing.assert_close(log_probs, legacy_log_probs, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    if with_entropy:
        legacy_entropy = _legacy_compute_entropy_from_logits(legacy_logits.clone(), process_group)
        torch.testing.assert_close(entropy, legacy_entropy, rtol=FORWARD_RTOL, atol=ENTROPY_FORWARD_ATOL)
        assert entropy.requires_grad == entropy_has_grad
    else:
        legacy_entropy = None
        assert entropy is None

    logprob_weights = torch.tensor([0.25, -0.5, 1.5, -0.75], dtype=torch.float32, device=device)
    entropy_weights = torch.tensor([0.55, -0.2, 1.8, 0.4], dtype=torch.float32, device=device)
    if not entropy_has_grad:
        entropy_weights = None
    loss = _weighted_loss(
        log_probs,
        entropy,
        logprob_weights=logprob_weights,
        entropy_weights=entropy_weights,
    )
    legacy_loss = _weighted_loss(
        legacy_log_probs,
        legacy_entropy,
        logprob_weights=logprob_weights,
        entropy_weights=entropy_weights,
    )
    loss.backward()
    legacy_loss.backward()

    backward_atol = ENTROPY_BACKWARD_ATOL if with_entropy and entropy_has_grad else BACKWARD_ATOL
    torch.testing.assert_close(
        logits.grad,
        legacy_logits.grad,
        rtol=BACKWARD_RTOL,
        atol=backward_atol,
    )


@pytest.fixture(scope="module")
def nccl_process_group():
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("megatron.core.fusions.fused_cross_entropy")
    if not dist.is_nccl_available():
        pytest.skip("NCCL is required")

    created_process_group = False
    if dist.is_initialized():
        process_group = dist.group.WORLD
        if dist.get_backend(process_group) != "nccl":
            pytest.skip("legacy Megatron CUDA parity needs an NCCL process group")
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_free_port())
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        created_process_group = True
        process_group = dist.group.WORLD

    yield process_group

    if created_process_group:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "with_entropy,entropy_has_grad",
    [
        pytest.param(False, False, id="without_entropy"),
        pytest.param(True, False, id="entropy_forward_only"),
        pytest.param(True, True, id="entropy_backward"),
    ],
)
@pytest.mark.parametrize("with_mask", [False, True], ids=["unmasked", "masked"])
@pytest.mark.parametrize("chunk_size", [-1, 2], ids=["no_chunks", "chunks"])
def test_calculate_log_probs_and_entropy_matches_legacy_megatron_cuda(
    nccl_process_group,
    chunk_size: int,
    with_mask: bool,
    with_entropy: bool,
    entropy_has_grad: bool,
):
    process_group = nccl_process_group
    torch.cuda.set_device(0)
    device = torch.device("cuda", torch.cuda.current_device())
    logits = _full_logits().to(device=device).requires_grad_()
    tokens = torch.tensor([5, 0, 3, 2], dtype=torch.long, device=device)
    keep_mask = _keep_mask().to(device=device) if with_mask else None

    _assert_legacy_parity(
        process_group=process_group,
        device=device,
        logits=logits,
        tokens=tokens,
        keep_mask=keep_mask,
        chunk_size=chunk_size,
        with_entropy=with_entropy,
        entropy_has_grad=entropy_has_grad,
    )


def _tp2_worker(rank: int, world_size: int, master_port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        process_group = dist.group.WORLD
        device = torch.device("cuda", rank)
        full_logits = _full_logits().to(device=device)
        full_keep_mask = _keep_mask().to(device=device)
        tokens = torch.tensor([5, 0, 3, 2], dtype=torch.long, device=device)

        vocab_per_rank = full_logits.size(-1) // world_size
        vocab_start = rank * vocab_per_rank
        vocab_end = vocab_start + vocab_per_rank
        for chunk_size, with_mask, with_entropy, entropy_has_grad in PARITY_SCENARIOS:
            logits = full_logits[:, vocab_start:vocab_end].detach().clone().requires_grad_()
            keep_mask = full_keep_mask[:, vocab_start:vocab_end] if with_mask else None
            _assert_legacy_parity(
                process_group=process_group,
                device=device,
                logits=logits,
                tokens=tokens,
                keep_mask=keep_mask,
                chunk_size=chunk_size,
                with_entropy=with_entropy,
                entropy_has_grad=entropy_has_grad,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_calculate_log_probs_and_entropy_matches_legacy_megatron_cuda_tp2():
    pytest.importorskip("megatron.core.fusions.fused_cross_entropy")
    if torch.cuda.device_count() < 2:
        pytest.skip("TP=2 parity requires two CUDA devices")

    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_nccl_available():
        pytest.skip("NCCL is required")

    world_size = 2
    mp.spawn(
        _tp2_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
