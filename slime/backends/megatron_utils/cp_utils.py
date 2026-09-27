from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu

from slime.utils.tensor_store import TensorRef

_RoutedExpertsInput = torch.Tensor | TensorRef


def get_logits_and_tokens_offset_with_cp(
    total_length: int,
    response_length: int,
):
    """
    All offsets start from the begining of the prompt.
    """
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size > 1

    prompt_length = total_length - response_length
    chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)

    # the offset of 2 chunks
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = (
        (2 * cp_size - cp_rank - 1) * chunk_size,
        (2 * cp_size - cp_rank) * chunk_size,
    )

    # the offset of 2 logits, note that the logits need a "-1".
    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1), min(chunk_1[1], total_length - 1))

    # when the sequence is empty, make an empty slice to continue the gradient flow.
    if logits_0[0] < logits_0[1]:
        token_0 = (logits_0[0] + 1, logits_0[1] + 1)
    else:
        logits_0 = (0, 0)
        token_0 = (0, 0)

    if logits_1[0] < logits_1[1]:
        token_1 = (logits_1[0] + 1, logits_1[1] + 1)
    else:
        logits_1 = (0, 0)
        token_1 = (0, 0)

    return chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1)


def get_sum_of_sample_mean(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    sample_denoms: list[torch.Tensor] | torch.Tensor | None = None,
    calculate_per_token_loss: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Calculate correct sample mean for CP.

    The default (``sample_denoms=None``) is the legacy per-sample mean: each
    sample's denominator is its own ``loss_mask.sum()``. Callers that want a
    per-rollout token-weighted mean pass pre-computed per-sample denominators
    (already as GPU tensors — see actor side) where every sample in the same
    rollout group carries the same value (the sum of that rollout's mask
    totals across every sibling sample in the step). Pre-computing at the
    step level rather than per-mb is required — otherwise a rollout whose
    samples land in different micro-batches would get a partial denominator
    on each side.
    """
    if sample_denoms is None:
        sample_denoms = [m.sum() for m in loss_masks]

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum() / torch.clamp_min(denom, 1)
                    for x_i, loss_mask_i, denom in zip(
                        x.split(response_lengths, dim=0),
                        loss_masks,
                        sample_denoms,
                        strict=False,
                    )
                ]
            )

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum()
                    for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
                ]
            )

    else:
        cp_chunk_lengths: list[int] = []
        chunked_loss_masks: list[torch.Tensor] = []

        for total_length, response_length, loss_mask in zip(total_lengths, response_lengths, loss_masks, strict=False):
            prompt_length = total_length - response_length
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(total_length, response_length)
            loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            chunked_loss_mask = torch.cat([loss_mask_0, loss_mask_1], dim=0)
            chunked_loss_masks.append(chunked_loss_mask)
            cp_chunk_lengths.append(chunked_loss_mask.size(0))

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum() / torch.clamp_min(denom, 1)
                    for x_i, chunked_loss_mask, denom in zip(
                        x.split(cp_chunk_lengths, dim=0),
                        chunked_loss_masks,
                        sample_denoms,
                        strict=False,
                    )
                ]
            )

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum()
                    for x_i, chunked_loss_mask in zip(
                        x.split(cp_chunk_lengths, dim=0),
                        chunked_loss_masks,
                        strict=False,
                    )
                ]
            )

    return sum_of_sample_mean if not calculate_per_token_loss else sum_of_token


def all_gather_with_cp(tensor: torch.Tensor, total_length: int, response_length: int) -> torch.Tensor:
    """
    Gather tensors across all ranks in the context parallel group.
    The first dimension of the output tensor will be the `response_length`.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_size = mpu.get_context_parallel_world_size()

    if cp_size == 1:
        return tensor

    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length)

    prompt_length = total_length - response_length

    chunk_0 = tensor[: logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0] :]
    assert chunk_1.shape[0] == logits_offset[1][1] - logits_offset[1][0]

    def zero(len: int) -> torch.Tensor:
        return torch.zeros(
            [len] + list(tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=True,
        )

    # logprob should be within the range of [prompt_length - 1, total_length - 1]
    if chunk_0.shape[0] == 0 and chunk_1.shape[0] == 0:
        # all empty
        full_tensor = zero(response_length)
    elif chunk_0.shape[0] != 0 and chunk_1.shape[0] == 0:
        # only first chunk
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[0][1])
        full_tensor = torch.cat([left, chunk_0, right], dim=0)
    elif chunk_0.shape[0] == 0 and chunk_1.shape[0] != 0:
        # only second chunk
        left = zero(logits_offset[1][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_1, right], dim=0)
    else:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        mid = zero(logits_offset[1][0] - logits_offset[0][1])
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)

    assert full_tensor.shape[0] == response_length, f"Expected {response_length}, got {full_tensor.shape}"
    full_tensor = dist.nn.all_reduce(full_tensor, group=cp_group)
    return full_tensor


def slice_with_cp(
    tokens: torch.Tensor,
    pad_value: tuple[int, float, Callable],
) -> torch.Tensor:
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()

    def pad_tokens(tokens, pad):
        if isinstance(pad_value, Callable):
            pad_func = pad_value
            tokens = pad_func(tokens, pad)
        else:
            # pad on the first dimension
            pad_tuple = (0, 0) * (tokens.dim() - 1) + (0, pad)
            tokens = F.pad(tokens, pad_tuple, value=pad_value)
        return tokens

    if cp_size == 1:
        return tokens

    token_len = len(tokens)
    chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)

    # pad
    pad = 2 * cp_size * chunk_size - token_len
    tokens = pad_tokens(tokens, pad)

    # get 2 chunk for thd cp
    start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])


def slice_log_prob_with_cp(
    log_prob: list[float] | torch.Tensor,
    total_length: int,
    response_length: int,
) -> list[float] | torch.Tensor:
    assert len(log_prob) == response_length, (
        f"log_prob length mismatch: len(log_prob)={len(log_prob)}, "
        f"response_length={response_length}, total_length={total_length}"
    )

    cp_size = mpu.get_context_parallel_world_size()

    if cp_size == 1:
        return log_prob

    prompt_length = total_length - response_length
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length)

    chunk_1 = log_prob[logits_offset[0][0] - (prompt_length - 1) : logits_offset[0][1] - (prompt_length - 1)]
    chunk_2 = log_prob[logits_offset[1][0] - (prompt_length - 1) : logits_offset[1][1] - (prompt_length - 1)]

    if isinstance(log_prob, list):
        return chunk_1 + chunk_2
    else:
        return torch.cat([chunk_1, chunk_2], dim=0)


def _make_routed_experts_pad(
    rows: int,
    num_layers: int,
    topk: int,
    *,
    num_experts: int,
    device: torch.device,
    dtype: torch.dtype,
    row_offset: int = 0,
) -> torch.Tensor:
    if rows == 0:
        return torch.empty((0, num_layers, topk), device=device, dtype=dtype)

    numel_per_row = num_layers * topk
    pad_experts = (
        torch.arange(
            rows * numel_per_row,
            device=device,
            dtype=torch.int64,
        ).add_(row_offset * numel_per_row)
        % num_experts
    )
    return pad_experts.to(dtype=dtype).reshape((rows, num_layers, topk))


def _new_routed_experts_buffer(
    rows: int,
    num_layers: int,
    topk: int,
    reference: _RoutedExpertsInput,
) -> torch.Tensor:
    kwargs = {
        "device": (torch.device("cpu") if isinstance(reference, TensorRef) else reference.device),
        "dtype": (reference.torch_dtype if isinstance(reference, TensorRef) else reference.dtype),
    }
    if kwargs["device"].type == "cpu":
        return torch.empty((rows, num_layers, topk), pin_memory=True, **kwargs)
    return torch.empty((rows, num_layers, topk), **kwargs)


def _fill_routed_experts_pad(
    dst: torch.Tensor,
    dst_start: int,
    rows: int,
    *,
    num_experts: int,
    row_offset: int = 0,
) -> None:
    if rows == 0:
        return
    _, num_layers, topk = dst.shape
    dst[dst_start : dst_start + rows].copy_(
        _make_routed_experts_pad(
            rows,
            num_layers,
            topk,
            num_experts=num_experts,
            device=dst.device,
            dtype=dst.dtype,
            row_offset=row_offset,
        )
    )


def _fill_padded_sample_range(
    dst: torch.Tensor,
    dst_start: int,
    experts: _RoutedExpertsInput,
    sample_start: int,
    sample_end: int,
    *,
    num_experts: int,
) -> None:
    """Copy a range from ``experts + sample-tail-pad + cp-pad`` into ``dst``."""
    if sample_start >= sample_end:
        return

    actual_len = experts.shape[0]
    token_len = actual_len + 1

    # Real routed-expert rows.
    real_start = max(sample_start, 0)
    real_end = min(sample_end, actual_len)
    if real_start < real_end:
        out_start = dst_start + real_start - sample_start
        dst[out_start : out_start + real_end - real_start].copy_(experts[real_start:real_end])

    # The per-sample row added before CP slicing.
    tail_start = max(sample_start, actual_len)
    tail_end = min(sample_end, token_len)
    if tail_start < tail_end:
        out_start = dst_start + tail_start - sample_start
        _fill_routed_experts_pad(
            dst,
            out_start,
            tail_end - tail_start,
            num_experts=num_experts,
            row_offset=tail_start - actual_len,
        )

    # Extra rows added by CP padding. This is a separate pad call in the old
    # implementation, so its deterministic expert pattern starts again at 0.
    cp_pad_start = max(sample_start, token_len)
    cp_pad_end = sample_end
    if cp_pad_start < cp_pad_end:
        out_start = dst_start + cp_pad_start - sample_start
        _fill_routed_experts_pad(
            dst,
            out_start,
            cp_pad_end - cp_pad_start,
            num_experts=num_experts,
            row_offset=cp_pad_start - token_len,
        )


def _fill_routed_experts_allgather_range(
    dst: torch.Tensor,
    range_start: int,
    range_end: int,
    rollout_routed_experts: Sequence[_RoutedExpertsInput],
    *,
    total_rows: int,
    num_experts: int,
) -> None:
    pos = 0
    for experts in rollout_routed_experts:
        sample_len = experts.shape[0] + 1
        overlap_start = max(range_start, pos)
        overlap_end = min(range_end, pos + sample_len)
        if overlap_start < overlap_end:
            _fill_padded_sample_range(
                dst,
                overlap_start - range_start,
                experts,
                overlap_start - pos,
                overlap_end - pos,
                num_experts=num_experts,
            )
        pos += sample_len
        if pos >= range_end:
            return

    overlap_start = max(range_start, total_rows)
    overlap_end = range_end
    if overlap_start < overlap_end:
        _fill_routed_experts_pad(
            dst,
            overlap_start - range_start,
            overlap_end - overlap_start,
            num_experts=num_experts,
            row_offset=overlap_start - total_rows,
        )


def _fill_routed_experts_cp_range(
    dst: torch.Tensor,
    range_start: int,
    range_end: int,
    sample_specs: Sequence[tuple[_RoutedExpertsInput, int, list[tuple[int, int]]]],
    *,
    total_rows: int,
    num_experts: int,
) -> None:
    pos = 0
    for experts, sample_out_len, chunks in sample_specs:
        overlap_start = max(range_start, pos)
        overlap_end = min(range_end, pos + sample_out_len)
        if overlap_start < overlap_end:
            rel_start = overlap_start - pos
            rel_end = overlap_end - pos
            chunk_out_start = 0
            for chunk_src_start, chunk_src_end in chunks:
                chunk_len = chunk_src_end - chunk_src_start
                chunk_overlap_start = max(rel_start, chunk_out_start)
                chunk_overlap_end = min(rel_end, chunk_out_start + chunk_len)
                if chunk_overlap_start < chunk_overlap_end:
                    dst_offset = overlap_start - range_start + chunk_overlap_start - rel_start
                    src_start = chunk_src_start + chunk_overlap_start - chunk_out_start
                    src_end = chunk_src_start + chunk_overlap_end - chunk_out_start
                    _fill_padded_sample_range(
                        dst,
                        dst_offset,
                        experts,
                        src_start,
                        src_end,
                        num_experts=num_experts,
                    )
                chunk_out_start += chunk_len
        pos += sample_out_len
        if pos >= range_end:
            return

    overlap_start = max(range_start, total_rows)
    overlap_end = range_end
    if overlap_start < overlap_end:
        _fill_routed_experts_pad(
            dst,
            overlap_start - range_start,
            overlap_end - overlap_start,
            num_experts=num_experts,
            row_offset=overlap_start - total_rows,
        )


def prepare_routed_experts_for_routing_replay(
    rollout_routed_experts: Sequence[_RoutedExpertsInput],
    tokens: Sequence[torch.Tensor],
    *,
    num_experts: int,
    data_pad_size_multiplier: int,
    sequence_parallel: bool,
    allgather_cp: bool,
) -> torch.Tensor:
    """Align routes with training tokens, reading only this rank's lazy rows."""
    assert len(rollout_routed_experts) == len(tokens)
    assert len(rollout_routed_experts) > 0
    _, num_layers, topk = rollout_routed_experts[0].shape
    for experts, token_ids in zip(rollout_routed_experts, tokens, strict=False):
        assert experts.shape[0] == token_ids.shape[0] - 1, f"{experts.shape}, {token_ids.shape}"
        assert experts.shape[1:] == (
            num_layers,
            topk,
        ), f"{experts.shape}, expected (*, {num_layers}, {topk})"

    pad_size = mpu.get_tensor_model_parallel_world_size() * data_pad_size_multiplier
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    range_start = 0

    if allgather_cp:
        total_rows = sum(experts.shape[0] + 1 for experts in rollout_routed_experts)
        global_pad_size = cp_size * pad_size
        pad = (global_pad_size - total_rows % global_pad_size) % global_pad_size
        final_rows = total_rows + pad
        cp_chunk_size = final_rows // cp_size
        range_start = cp_rank * cp_chunk_size
        range_end = range_start + cp_chunk_size
    else:
        sample_specs: list[tuple[_RoutedExpertsInput, int, list[tuple[int, int]]]] = []
        total_rows = 0
        for experts in rollout_routed_experts:
            token_len = experts.shape[0] + 1
            if cp_size == 1:
                chunks = [(0, token_len)]
            else:
                chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
                chunks = [
                    (chunk_size * cp_rank, chunk_size * (cp_rank + 1)),
                    (
                        chunk_size * (2 * cp_size - cp_rank - 1),
                        chunk_size * (2 * cp_size - cp_rank),
                    ),
                ]
            sample_out_len = sum(end - start for start, end in chunks)
            sample_specs.append((experts, sample_out_len, chunks))
            total_rows += sample_out_len
        pad = (pad_size - total_rows % pad_size) % pad_size
        final_rows = total_rows + pad
        range_end = final_rows

    if sequence_parallel:
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        seqlen = range_end - range_start
        assert seqlen % tp_size == 0
        tp_chunk_size = seqlen // tp_size
        range_start += tp_chunk_size * tp_rank
        range_end = range_start + tp_chunk_size

    routed_experts = _new_routed_experts_buffer(
        range_end - range_start,
        num_layers,
        topk,
        rollout_routed_experts[0],
    )
    if allgather_cp:
        _fill_routed_experts_allgather_range(
            routed_experts,
            range_start,
            range_end,
            rollout_routed_experts,
            total_rows=total_rows,
            num_experts=num_experts,
        )
    else:
        _fill_routed_experts_cp_range(
            routed_experts,
            range_start,
            range_end,
            sample_specs,
            total_rows=total_rows,
            num_experts=num_experts,
        )
    return routed_experts
