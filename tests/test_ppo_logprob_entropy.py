"""CPU tests for fused PPO log-probability and entropy calculation."""

from __future__ import annotations

import os
import socket

import pytest
import torch

from slime.utils.ppo_utils import calculate_log_probs_and_entropy


NUM_GPUS = 0

STRICT_ATOL = 1e-8
STRICT_RTOL = 0.0


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _unfused_reference_logprob_entropy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    keep_mask: torch.Tensor | None,
    *,
    with_entropy: bool,
    num_partitions: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference for the pre-fused behavior, preserving its reduction order."""
    logprob_logits = logits
    if keep_mask is not None:
        logprob_logits = logits.masked_fill(~keep_mask, float("-inf"))
        # Match replay behavior: the sampled token must stay finite even
        # when an engine-side top-p mask omitted it.
        rows = torch.arange(tokens.numel(), device=logits.device)
        logprob_logits[rows, tokens] = logits[rows, tokens]

    log_probs = _reference_log_probs_with_partition_order(logprob_logits, tokens, num_partitions=num_partitions)
    entropy = None
    if with_entropy:
        entropy = _reference_entropy_with_partition_order(logits, num_partitions=num_partitions)
    return log_probs, entropy


def _sum_in_partition_order(chunks: list[torch.Tensor]) -> torch.Tensor:
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


def _reference_log_probs_with_partition_order(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    num_partitions: int,
) -> torch.Tensor:
    rows = torch.arange(tokens.numel(), device=logits.device)
    chunks = list(logits.chunk(num_partitions, dim=-1))
    vocab_per_partition = chunks[0].size(-1)

    logits_max = torch.stack([chunk.max(dim=-1, keepdim=True).values for chunk in chunks], dim=0).max(dim=0).values
    normalized_chunks = [chunk - logits_max for chunk in chunks]
    exp_chunks = [chunk.exp() for chunk in normalized_chunks]
    sum_exp_logits = _sum_in_partition_order([chunk.sum(dim=-1, keepdim=True) for chunk in exp_chunks])

    predicted_logits = logits.new_zeros((tokens.numel(), 1))
    for partition, normalized_chunk in enumerate(normalized_chunks):
        vocab_start = partition * vocab_per_partition
        local_tokens = tokens - vocab_start
        on_partition = (local_tokens >= 0) & (local_tokens < vocab_per_partition)
        local_tokens = local_tokens.clamp(0, vocab_per_partition - 1)
        partition_predicted_logits = normalized_chunk[rows, local_tokens].unsqueeze(-1)
        partition_predicted_logits = partition_predicted_logits.masked_fill(~on_partition.unsqueeze(-1), 0.0)
        predicted_logits = predicted_logits + partition_predicted_logits

    return predicted_logits - sum_exp_logits.log()


def _reference_entropy_with_partition_order(logits: torch.Tensor, *, num_partitions: int) -> torch.Tensor:
    chunks = list(logits.chunk(num_partitions, dim=-1))
    logits_max = torch.stack([chunk.max(dim=-1, keepdim=True).values for chunk in chunks], dim=0).max(dim=0).values
    normalized_chunks = [chunk - logits_max for chunk in chunks]
    exp_chunks = [chunk.exp() for chunk in normalized_chunks]
    sum_exp_logits = _sum_in_partition_order([chunk.sum(dim=-1, keepdim=True) for chunk in exp_chunks])
    softmax_chunks = [chunk / sum_exp_logits for chunk in exp_chunks]
    sum_softmax_times_logits = _sum_in_partition_order(
        [(softmax * chunk).sum(dim=-1, keepdim=True) for softmax, chunk in zip(softmax_chunks, chunks, strict=True)]
    )
    return (logits_max + sum_exp_logits.log() - sum_softmax_times_logits).squeeze(dim=-1)


def _reference_grad_with_partition_order(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    keep_mask: torch.Tensor | None,
    *,
    with_entropy: bool,
    logprob_weights: torch.Tensor,
    entropy_weights: torch.Tensor,
    num_partitions: int = 1,
) -> torch.Tensor:
    logprob_logits = logits
    if keep_mask is not None:
        logprob_logits = logits.masked_fill(~keep_mask, float("-inf"))
        rows = torch.arange(tokens.numel(), device=logits.device)
        logprob_logits[rows, tokens] = logits[rows, tokens]

    logprob_softmax_chunks = _reference_softmax_chunks_with_partition_order(
        logprob_logits,
        num_partitions=num_partitions,
    )
    grad_chunks = []
    vocab_per_partition = logprob_softmax_chunks[0].size(-1)
    for partition, softmax_chunk in enumerate(logprob_softmax_chunks):
        vocab_start = partition * vocab_per_partition
        local_tokens = tokens - vocab_start
        on_partition = (local_tokens >= 0) & (local_tokens < vocab_per_partition)
        local_tokens = local_tokens.clamp(0, vocab_per_partition - 1)

        grad_chunk = -softmax_chunk
        rows = torch.arange(tokens.numel(), device=logits.device)
        grad_2d = grad_chunk.view(-1, vocab_per_partition)
        grad_2d[rows, local_tokens] += on_partition.to(dtype=grad_2d.dtype)
        grad_chunk = grad_chunk * logprob_weights.reshape(-1, 1)
        grad_chunks.append(grad_chunk)

    grad = torch.cat(grad_chunks, dim=-1)

    if with_entropy:
        entropy_softmax_chunks = _reference_softmax_chunks_with_partition_order(
            logits,
            num_partitions=num_partitions,
        )
        logits_chunks = list(logits.chunk(num_partitions, dim=-1))
        sum_softmax_times_logits = _sum_in_partition_order(
            [
                (softmax * logits_chunk).sum(dim=-1, keepdim=True)
                for softmax, logits_chunk in zip(entropy_softmax_chunks, logits_chunks, strict=True)
            ]
        )
        entropy_grad = torch.cat(
            [
                softmax * (sum_softmax_times_logits - logits_chunk) * entropy_weights.reshape(-1, 1)
                for softmax, logits_chunk in zip(entropy_softmax_chunks, logits_chunks, strict=True)
            ],
            dim=-1,
        )
        grad = grad + entropy_grad

    return grad


def _reference_softmax_chunks_with_partition_order(
    logits: torch.Tensor,
    *,
    num_partitions: int,
) -> list[torch.Tensor]:
    chunks = list(logits.chunk(num_partitions, dim=-1))
    logits_max = torch.stack([chunk.max(dim=-1, keepdim=True).values for chunk in chunks], dim=0).max(dim=0).values
    normalized_chunks = [chunk - logits_max for chunk in chunks]
    exp_chunks = [chunk.exp() for chunk in normalized_chunks]
    sum_exp_logits = _sum_in_partition_order([chunk.sum(dim=-1, keepdim=True) for chunk in exp_chunks])
    return [chunk / sum_exp_logits for chunk in exp_chunks]


def _single_rank_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 1.0, 0.5, 2.0],
            [-1.0, 3.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )


def _single_rank_keep_mask() -> torch.Tensor:
    return torch.tensor(
        [
            [False, True, True, False],  # target 3 is deliberately absent.
            [False, True, False, True],  # target 0 is deliberately absent.
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )


def _weighted_loss(
    log_probs: torch.Tensor,
    entropy: torch.Tensor | None,
    *,
    logprob_weights: torch.Tensor,
    entropy_weights: torch.Tensor,
) -> torch.Tensor:
    loss = (log_probs.squeeze(-1) * logprob_weights).sum()
    if entropy is not None:
        loss = loss + (entropy * entropy_weights).sum()
    return loss


@pytest.mark.parametrize("chunk_size", [-1, 1, 2, 8])
@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("with_entropy", [False, True])
def test_calculate_log_probs_and_entropy_matches_unfused_reference_single_rank(
    chunk_size: int,
    with_mask: bool,
    with_entropy: bool,
):
    logits = _single_rank_logits().requires_grad_()
    tokens = torch.tensor([3, 0, 1], dtype=torch.long)
    keep_mask = _single_rank_keep_mask() if with_mask else None

    log_probs, entropy = calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        with_entropy=with_entropy,
        chunk_size=chunk_size,
        log_prob_keep_mask=keep_mask,
    )

    ref_logits = logits.detach().clone().requires_grad_()
    expected_log_probs, expected_entropy = _unfused_reference_logprob_entropy(
        ref_logits,
        tokens,
        keep_mask,
        with_entropy=with_entropy,
    )

    torch.testing.assert_close(log_probs, expected_log_probs, rtol=STRICT_RTOL, atol=STRICT_ATOL)
    if with_entropy:
        torch.testing.assert_close(entropy, expected_entropy, rtol=STRICT_RTOL, atol=STRICT_ATOL)
    else:
        assert entropy is None
        assert expected_entropy is None

    logprob_weights = torch.tensor([0.25, -0.5, 1.5], dtype=torch.float32)
    entropy_weights = torch.tensor([0.55, -0.2, 1.8], dtype=torch.float32)
    loss = _weighted_loss(
        log_probs,
        entropy,
        logprob_weights=logprob_weights,
        entropy_weights=entropy_weights,
    )
    loss.backward()
    expected_grad = _reference_grad_with_partition_order(
        ref_logits,
        tokens,
        keep_mask,
        with_entropy=with_entropy,
        logprob_weights=logprob_weights,
        entropy_weights=entropy_weights,
    )

    torch.testing.assert_close(logits.grad, expected_grad, rtol=STRICT_RTOL, atol=STRICT_ATOL)


@pytest.mark.parametrize("with_entropy", [False, True])
def test_calculate_log_probs_and_entropy_handles_empty_input(with_entropy: bool):
    logits = torch.empty((0, 4), dtype=torch.float32, requires_grad=True)
    tokens = torch.empty((0,), dtype=torch.long)
    keep_mask = torch.empty((0, 4), dtype=torch.bool)

    log_probs, entropy = calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        with_entropy=with_entropy,
        chunk_size=2,
        log_prob_keep_mask=keep_mask,
    )

    assert log_probs.shape == (0,)
    if with_entropy:
        assert entropy is not None
        assert entropy.shape == (0,)
    else:
        assert entropy is None


def _distributed_full_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 0.5, -1.0],
            [4.0, 1.0, 0.5, 2.0, 3.0, 0.0],
            [-1.0, 3.0, 2.0, 0.0, 1.0, 5.0],
            [0.2, -0.4, 1.7, -2.0, 3.3, 0.0],
        ],
        dtype=torch.float32,
    )


def _distributed_keep_mask() -> torch.Tensor:
    return torch.tensor(
        [
            [False, True, False, True, False, False],  # target 5 is absent.
            [False, True, False, False, True, False],  # target 0 is absent.
            [True, False, True, False, False, True],  # target 3 is absent.
            [False, True, False, True, False, False],  # target 2 is absent.
        ],
        dtype=torch.bool,
    )


def _distributed_vocab_worker(
    rank: int,
    world_size: int,
    with_mask: bool,
    with_entropy: bool,
    chunk_size: int,
    master_port: int,
) -> None:
    import torch.distributed as dist

    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        full_logits = _distributed_full_logits()
        tokens = torch.tensor([5, 0, 3, 2], dtype=torch.long)
        full_keep_mask = _distributed_keep_mask() if with_mask else None

        vocab_per_rank = full_logits.size(-1) // world_size
        vocab_start = rank * vocab_per_rank
        vocab_end = vocab_start + vocab_per_rank
        local_logits = full_logits[:, vocab_start:vocab_end].detach().clone().requires_grad_()
        local_keep_mask = None
        if full_keep_mask is not None:
            local_keep_mask = full_keep_mask[:, vocab_start:vocab_end]

        log_probs, entropy = calculate_log_probs_and_entropy(
            local_logits,
            tokens,
            tp_group=None,
            with_entropy=with_entropy,
            chunk_size=chunk_size,
            log_prob_keep_mask=local_keep_mask,
        )

        ref_logits = full_logits.detach().clone().requires_grad_()
        expected_log_probs, expected_entropy = _unfused_reference_logprob_entropy(
            ref_logits,
            tokens,
            full_keep_mask,
            with_entropy=with_entropy,
            num_partitions=world_size,
        )

        torch.testing.assert_close(log_probs, expected_log_probs, rtol=STRICT_RTOL, atol=STRICT_ATOL)
        if with_entropy:
            torch.testing.assert_close(entropy, expected_entropy, rtol=STRICT_RTOL, atol=STRICT_ATOL)
        else:
            assert entropy is None
            assert expected_entropy is None

        logprob_weights = torch.tensor([0.25, -0.5, 1.5, -0.75], dtype=torch.float32)
        entropy_weights = torch.tensor([0.55, -0.2, 1.8, 0.4], dtype=torch.float32)
        loss = _weighted_loss(
            log_probs,
            entropy,
            logprob_weights=logprob_weights,
            entropy_weights=entropy_weights,
        )
        loss.backward()
        expected_grad = _reference_grad_with_partition_order(
            ref_logits,
            tokens,
            full_keep_mask,
            with_entropy=with_entropy,
            logprob_weights=logprob_weights,
            entropy_weights=entropy_weights,
            num_partitions=world_size,
        )

        torch.testing.assert_close(
            local_logits.grad,
            expected_grad[:, vocab_start:vocab_end],
            rtol=STRICT_RTOL,
            atol=STRICT_ATOL,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "with_mask,with_entropy,chunk_size",
    [
        pytest.param(False, True, -1, id="unmasked_entropy_no_chunks"),
        pytest.param(True, True, 2, id="masked_entropy_chunks"),
        pytest.param(True, False, -1, id="masked_logprob_only_no_chunks"),
        pytest.param(False, False, 2, id="unmasked_logprob_only_chunks"),
    ],
)
def test_calculate_log_probs_and_entropy_matches_unfused_reference_vocab_parallel(
    with_mask: bool,
    with_entropy: bool,
    chunk_size: int,
):
    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _distributed_vocab_worker,
        args=(world_size, with_mask, with_entropy, chunk_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
