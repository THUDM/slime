from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def _cpu_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    tensor = torch.as_tensor(value, dtype=dtype) if dtype is not None else torch.as_tensor(value)
    return tensor.detach().cpu().contiguous()


def _pad_routed_experts(experts: torch.Tensor, pad: int, num_experts: int) -> torch.Tensor:
    if pad == 0:
        return experts
    _, num_layers, topk = experts.shape
    pad_experts = (
        torch.arange(
            pad * num_layers * topk,
            device=experts.device,
            dtype=experts.dtype,
        ).reshape((pad, num_layers, topk))
        % num_experts
    )
    return torch.cat([experts, pad_experts], dim=0)


def _slice_routed_experts_with_cp_rank(
    experts: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
    num_experts: int,
) -> torch.Tensor:
    """Apply the THD mirrored CP slice without consulting process-global MPU."""
    if cp_size == 1:
        return experts

    token_len = len(experts)
    chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    experts = _pad_routed_experts(experts, 2 * cp_size * chunk_size - token_len, num_experts)
    start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    start_2 = chunk_size * (2 * cp_size - cp_rank - 1)
    end_2 = chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([experts[start_1:end_1], experts[start_2:end_2]], dim=0)


def prepare_routed_experts_for_routing_replay_rank(
    rollout_routed_experts: Sequence[torch.Tensor],
    tokens: Sequence[torch.Tensor],
    *,
    num_experts: int,
    data_pad_size_multiplier: int,
    sequence_parallel: bool,
    allgather_cp: bool,
    cp_rank: int,
    cp_size: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Align routes for an explicit logical CP/TP rank."""
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank {cp_rank} is outside [0, {cp_size})")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank {tp_rank} is outside [0, {tp_size})")
    assert len(rollout_routed_experts) == len(tokens)
    for experts, token_ids in zip(rollout_routed_experts, tokens, strict=False):
        assert experts.shape[0] == token_ids.shape[0] - 1, f"{experts.shape}, {token_ids.shape}"

    padded_experts = [_pad_routed_experts(experts, 1, num_experts) for experts in rollout_routed_experts]
    pad_size = tp_size * data_pad_size_multiplier

    if allgather_cp:
        routed_experts = torch.cat(padded_experts, dim=0)
        global_pad_size = cp_size * pad_size
        pad = (global_pad_size - routed_experts.size(0) % global_pad_size) % global_pad_size
        routed_experts = _pad_routed_experts(routed_experts, pad, num_experts)
        routed_experts = routed_experts.chunk(cp_size, dim=0)[cp_rank]
    else:
        routed_experts = [
            _slice_routed_experts_with_cp_rank(
                experts,
                cp_rank=cp_rank,
                cp_size=cp_size,
                num_experts=num_experts,
            )
            for experts in padded_experts
        ]
        routed_experts = torch.cat(routed_experts, dim=0)
        pad = (pad_size - routed_experts.size(0) % pad_size) % pad_size
        routed_experts = _pad_routed_experts(routed_experts, pad, num_experts)

    if sequence_parallel:
        seqlen = routed_experts.size(0)
        assert seqlen % tp_size == 0
        start = seqlen // tp_size * tp_rank
        end = seqlen // tp_size * (tp_rank + 1)
        routed_experts = routed_experts[start:end]

    return routed_experts


def prepare_routing_replay_shard(
    *,
    tokens: list[Any],
    routed_experts: list[Any],
    micro_batch_indices: list[list[int]],
    cp_rank: int,
    cp_size: int,
    tp_rank: int,
    tp_size: int,
    num_experts: int,
    data_pad_size_multiplier: int,
    sequence_parallel: bool,
    allgather_cp: bool,
) -> list[torch.Tensor]:
    """Build compact per-microbatch route tensors for one logical CP/TP rank."""
    tokens = [_cpu_tensor(value, dtype=torch.long) for value in tokens]
    routed_experts = [_cpu_tensor(value) for value in routed_experts]
    return [
        prepare_routed_experts_for_routing_replay_rank(
            [routed_experts[i] for i in indices],
            [tokens[i] for i in indices],
            num_experts=num_experts,
            data_pad_size_multiplier=data_pad_size_multiplier,
            sequence_parallel=sequence_parallel,
            allgather_cp=allgather_cp,
            cp_rank=cp_rank,
            cp_size=cp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        .detach()
        .cpu()
        # A TP slice along dim 0 can be logically contiguous while still
        # retaining the entire pre-slice CP storage. Ray serializes that
        # backing storage, so force an owning, compact allocation.
        .clone(memory_format=torch.contiguous_format)
        for indices in micro_batch_indices
    ]
