"""Runtime fallback for Megatron linear cross entropy.

Slime v0.3.0 can run on images where Megatron's Blackwell linear CE module is
present but its CUDA/Cutlass entry points were not built.  Megatron then fails
late at the first long-context SFT forward.  This patch keeps the linear-CE
code path enabled and provides a chunked PyTorch implementation only when the
native entry points are absent.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
_SHAPE_MISMATCH_LOGGED = False
_TIMING_COUNTS = {"forward": 0, "backward": 0}
_TIMING_CONFIG_LOGGED = False


def _env_value(name: str, default: str) -> str:
    return os.environ.get(name, os.environ.get(name.lower(), default))


def _env_flag(name: str) -> bool:
    return _env_value(name, "0").lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_value(name, str(default)))
    except ValueError:
        return default


def _rank() -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _timing_should_sample(phase: Literal["forward", "backward"]) -> tuple[bool, int]:
    global _TIMING_CONFIG_LOGGED
    if not _env_flag("SLIME_LINEAR_CE_FALLBACK_TIMING"):
        return False, 0
    rank_filter = _env_int("SLIME_LINEAR_CE_FALLBACK_TIMING_RANK", 0)
    rank = _rank()
    if rank_filter >= 0 and rank != rank_filter:
        return False, 0

    if not _TIMING_CONFIG_LOGGED:
        _TIMING_CONFIG_LOGGED = True
        logger.info(
            "Linear CE fallback timing enabled: rank=%s interval=%s limit=%s vocab_chunk=%s",
            rank_filter,
            _env_int("SLIME_LINEAR_CE_FALLBACK_TIMING_INTERVAL", 32),
            _env_int("SLIME_LINEAR_CE_FALLBACK_TIMING_LIMIT", 64),
            _chunk_size(),
        )

    count = _TIMING_COUNTS.get(phase, 0) + 1
    _TIMING_COUNTS[phase] = count
    limit = _env_int("SLIME_LINEAR_CE_FALLBACK_TIMING_LIMIT", 64)
    interval = max(1, _env_int("SLIME_LINEAR_CE_FALLBACK_TIMING_INTERVAL", 32))
    return count <= limit and (count == 1 or count % interval == 0), count


def _timing_start(enabled: bool):
    if not enabled:
        return None, 0.0
    if torch.cuda.is_available():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event, 0.0
    return None, time.perf_counter()


def _timing_elapsed_ms(start_event, start_wall: float) -> float:
    if start_event is not None:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        return float(start_event.elapsed_time(end_event))
    return (time.perf_counter() - start_wall) * 1000.0


def _world_info(tp_group):
    if tp_group is None:
        return 0, 1
    return dist.get_rank(tp_group), dist.get_world_size(tp_group)


def _chunk_size() -> int:
    return max(1, int(os.environ.get("SLIME_LINEAR_CE_FALLBACK_VOCAB_CHUNK", "1024")))


def _labels_for_local_vocab(labels: torch.Tensor, vocab_size: int, tp_rank: int) -> torch.Tensor:
    return labels - tp_rank * vocab_size


def _valid_labels(labels: torch.Tensor, ignore_index: int, vocab_size: int, tp_world_size: int) -> torch.Tensor:
    global_vocab_size = vocab_size * tp_world_size
    return labels.ne(ignore_index) & labels.ge(0) & labels.lt(global_vocab_size)


def _assert_no_valid_tail_labels(
    labels: torch.Tensor,
    *,
    common_tokens: int,
    ignore_index: int,
    vocab_size: int,
    tp_world_size: int,
    hidden_tokens: int,
) -> None:
    if labels.numel() <= common_tokens:
        return
    tail = labels[common_tokens:]
    valid_tail = _valid_labels(tail, ignore_index, vocab_size, tp_world_size)
    if valid_tail.any():
        first = torch.nonzero(valid_tail, as_tuple=False).flatten()[0].item()
        raise RuntimeError(
            "Linear CE fallback would drop valid tail labels because hidden is shorter than labels: "
            f"hidden_tokens={hidden_tokens}, label_tokens={labels.numel()}, common_tokens={common_tokens}, "
            f"first_valid_tail_offset={common_tokens + first}, label={tail[first].item()}"
        )


def _reduce_loss(losses: torch.Tensor, valid: torch.Tensor, reduction: Literal["none", "sum", "mean"]):
    losses = losses.masked_fill(~valid, 0.0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        denom = torch.clamp_min(valid.sum(), 1)
        return losses.sum() / denom
    raise ValueError(f"Unsupported reduction: {reduction}")


def _forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    tp_group=None,
    reduction: Literal["none", "sum", "mean"] = "mean",
    ignore_index: int = -100,
    sequence_parallel: bool = False,
):
    should_time, timing_sample = _timing_should_sample("forward")
    timing_event, timing_wall = _timing_start(should_time)

    global _SHAPE_MISMATCH_LOGGED
    tp_rank, tp_world_size = _world_info(tp_group)
    in_tp_mode = tp_group is not None and tp_world_size > 1

    hidden_shape = hidden.shape
    hidden_view = hidden.view(-1, hidden.shape[-1])
    labels_view = labels.view(-1)
    global_hidden = hidden

    if in_tp_mode and sequence_parallel:
        partial_shape = hidden.shape
        global_shape = (partial_shape[0] * tp_world_size, *partial_shape[1:])
        global_hidden = torch.empty(global_shape, dtype=hidden.dtype, device=hidden.device)
        dist.all_gather_into_tensor(global_hidden, hidden, group=tp_group)
        hidden_view = global_hidden.view(-1, global_hidden.shape[-1])

    label_tokens = labels_view.numel()
    hidden_tokens = hidden_view.shape[0]
    if hidden_tokens != label_tokens and not _SHAPE_MISMATCH_LOGGED:
        logger.warning(
            "Linear CE fallback aligning padded hidden/label rows: hidden_tokens=%s labels=%s "
            "hidden_shape=%s labels_shape=%s sequence_parallel=%s tp_world_size=%s",
            hidden_tokens,
            label_tokens,
            tuple(hidden_shape),
            tuple(labels.shape),
            sequence_parallel,
            tp_world_size,
        )
        _SHAPE_MISMATCH_LOGGED = True
    common_tokens = min(hidden_tokens, label_tokens)
    hidden_used = hidden_view[:common_tokens]
    labels_used = labels_view[:common_tokens]

    num_tokens = common_tokens
    vocab_size = weight.shape[0]
    chunk = _chunk_size()
    _assert_no_valid_tail_labels(
        labels_view,
        common_tokens=common_tokens,
        ignore_index=ignore_index,
        vocab_size=vocab_size,
        tp_world_size=tp_world_size,
        hidden_tokens=hidden_tokens,
    )
    labels_local = _labels_for_local_vocab(labels_used, vocab_size, tp_rank)
    valid = _valid_labels(labels_used, ignore_index, vocab_size, tp_world_size)

    local_max = torch.full((num_tokens,), -float("inf"), dtype=torch.float32, device=hidden.device)
    for start in range(0, vocab_size, chunk):
        weight_chunk = weight[start : start + chunk]
        logits = torch.matmul(hidden_used, weight_chunk.t()).float()
        local_max = torch.maximum(local_max, logits.max(dim=-1).values)

    maximum = local_max
    if in_tp_mode:
        maximum = maximum.clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)

    accumulate = torch.zeros((num_tokens,), dtype=torch.float32, device=hidden.device)
    target_logits = torch.zeros((num_tokens,), dtype=torch.float32, device=hidden.device)
    for start in range(0, vocab_size, chunk):
        end = min(start + chunk, vocab_size)
        weight_chunk = weight[start:end]
        logits = torch.matmul(hidden_used, weight_chunk.t()).float()
        accumulate.add_(torch.exp(logits - maximum[:, None]).sum(dim=-1))

        in_chunk = valid & labels_local.ge(start) & labels_local.lt(end)
        if in_chunk.any():
            row_idx = torch.nonzero(in_chunk, as_tuple=False).flatten()
            col_idx = labels_local[row_idx] - start
            target_logits[row_idx] = logits[row_idx, col_idx].float()

    if in_tp_mode:
        dist.all_reduce(accumulate, op=dist.ReduceOp.SUM, group=tp_group)
        dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=tp_group)

    losses = torch.log(accumulate) + maximum - target_logits
    if reduction == "none" and label_tokens > common_tokens:
        padded_losses = torch.zeros((label_tokens,), dtype=losses.dtype, device=losses.device)
        padded_losses[:common_tokens] = losses
        padded_valid = torch.zeros((label_tokens,), dtype=torch.bool, device=valid.device)
        padded_valid[:common_tokens] = valid
        logprobs = _reduce_loss(padded_losses, padded_valid, reduction)
    else:
        logprobs = _reduce_loss(losses, valid, reduction)
    num_valid_tokens = valid.sum().to(dtype=torch.int64)

    if should_time:
        elapsed_ms = _timing_elapsed_ms(timing_event, timing_wall)
        logger.info(
            "Linear CE fallback timing: phase=forward sample=%s rank=%s elapsed_ms=%.3f "
            "hidden_shape=%s weight_shape=%s labels_shape=%s common_tokens=%s valid_tokens=%s "
            "vocab_size=%s chunk=%s reduction=%s sequence_parallel=%s tp_world_size=%s",
            timing_sample,
            _rank(),
            elapsed_ms,
            tuple(hidden_shape),
            tuple(weight.shape),
            tuple(labels.shape),
            common_tokens,
            int(num_valid_tokens.detach().item()),
            vocab_size,
            chunk,
            reduction,
            sequence_parallel,
            tp_world_size,
        )

    return (
        logprobs,
        maximum,
        accumulate,
        num_valid_tokens,
        tp_rank,
        tp_world_size,
        global_hidden,
    )


def _grad_factor(
    dlogprobs: torch.Tensor,
    labels_view: torch.Tensor,
    valid: torch.Tensor,
    num_valid_tokens: torch.Tensor,
    reduction: Literal["none", "sum", "mean"],
) -> torch.Tensor:
    if reduction == "none":
        grad = dlogprobs.view(-1).float()
    elif reduction == "sum":
        grad = torch.ones_like(labels_view, dtype=torch.float32) * dlogprobs.float()
    elif reduction == "mean":
        denom = torch.clamp_min(num_valid_tokens, 1).float()
        grad = torch.ones_like(labels_view, dtype=torch.float32) * dlogprobs.float() / denom
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    return grad.masked_fill(~valid, 0.0)


def _backward(
    dlogprobs: torch.Tensor,
    global_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    maximum: torch.Tensor,
    accumulate: torch.Tensor,
    num_valid_tokens: torch.Tensor,
    reduction: Literal["none", "sum", "mean"] = "mean",
    ignore_index: int = -100,
    tp_group=None,
    tp_rank: int = 0,
    tp_world_size: int = 1,
    sequence_parallel: bool = False,
):
    should_time, timing_sample = _timing_should_sample("backward")
    timing_event, timing_wall = _timing_start(should_time)

    in_tp_mode = tp_group is not None and tp_world_size > 1
    hidden_view = global_hidden.view(-1, global_hidden.shape[-1])
    labels_view = labels.view(-1)
    label_tokens = labels_view.numel()
    hidden_tokens = hidden_view.shape[0]
    common_tokens = min(hidden_tokens, label_tokens)
    hidden_used = hidden_view[:common_tokens]
    labels_used = labels_view[:common_tokens]
    vocab_size = weight.shape[0]
    chunk = _chunk_size()
    _assert_no_valid_tail_labels(
        labels_view,
        common_tokens=common_tokens,
        ignore_index=ignore_index,
        vocab_size=vocab_size,
        tp_world_size=tp_world_size,
        hidden_tokens=hidden_tokens,
    )
    labels_local = _labels_for_local_vocab(labels_used, vocab_size, tp_rank)
    valid = _valid_labels(labels_used, ignore_index, vocab_size, tp_world_size)
    dlogprobs_used = dlogprobs.view(-1)[:common_tokens] if reduction == "none" else dlogprobs
    grad = _grad_factor(dlogprobs_used, labels_used, valid, num_valid_tokens, reduction)

    d_hidden_used = torch.zeros_like(hidden_used)
    d_weight = torch.zeros_like(weight)

    for start in range(0, vocab_size, chunk):
        end = min(start + chunk, vocab_size)
        weight_chunk = weight[start:end]
        logits = torch.matmul(hidden_used, weight_chunk.t()).float()
        probs = torch.exp(logits - maximum[:, None]) / accumulate[:, None]

        in_chunk = valid & labels_local.ge(start) & labels_local.lt(end)
        if in_chunk.any():
            row_idx = torch.nonzero(in_chunk, as_tuple=False).flatten()
            col_idx = labels_local[row_idx] - start
            probs[row_idx, col_idx] -= 1.0

        probs.mul_(grad[:, None])
        probs_for_mm = probs.to(dtype=hidden_used.dtype)
        d_hidden_used.addmm_(probs_for_mm, weight_chunk)
        d_weight[start:end] = torch.matmul(probs_for_mm.t(), hidden_used)

    d_hidden = torch.zeros_like(hidden_view)
    d_hidden[:common_tokens] = d_hidden_used

    if in_tp_mode:
        dist.all_reduce(d_hidden, op=dist.ReduceOp.SUM, group=tp_group)
        if sequence_parallel:
            partial_hidden_shape = (global_hidden.shape[0] // tp_world_size, *global_hidden.shape[1:])
            local_tokens = d_hidden.shape[0] // tp_world_size
            d_hidden = d_hidden[tp_rank * local_tokens : (tp_rank + 1) * local_tokens].clone()
            d_hidden = d_hidden.view(partial_hidden_shape)

    if should_time:
        elapsed_ms = _timing_elapsed_ms(timing_event, timing_wall)
        logger.info(
            "Linear CE fallback timing: phase=backward sample=%s rank=%s elapsed_ms=%.3f "
            "dlogprobs_shape=%s hidden_shape=%s weight_shape=%s labels_shape=%s common_tokens=%s "
            "vocab_size=%s chunk=%s reduction=%s sequence_parallel=%s tp_world_size=%s",
            timing_sample,
            _rank(),
            elapsed_ms,
            tuple(dlogprobs.shape),
            tuple(global_hidden.shape),
            tuple(weight.shape),
            tuple(labels.shape),
            common_tokens,
            vocab_size,
            chunk,
            reduction,
            sequence_parallel,
            tp_world_size,
        )

    return d_hidden.view_as(global_hidden if not (in_tp_mode and sequence_parallel) else d_hidden), d_weight


def apply_linear_ce_fallback_patch() -> None:
    try:
        from megatron.core.fusions import fused_linear_cross_entropy as fused_lce
    except Exception:
        return

    entry = None
    try:
        from megatron.core.fusions.linear_cross_entropy.blackwell import entry as blackwell_entry

        entry = blackwell_entry
    except Exception:
        pass

    missing_forward = entry is None or not callable(getattr(entry, "forward", None))
    missing_backward = entry is None or not callable(getattr(entry, "backward", None))
    if not (missing_forward or missing_backward):
        return

    class FallbackPlatform:
        def __init__(self) -> None:
            self.forward_func = _forward
            self.backward_func = _backward

    if entry is not None:
        entry.forward = _forward
        entry.backward = _backward
    fused_lce.Platform = FallbackPlatform
    try:
        fused_lce._get_platform.cache_clear()
    except Exception:
        pass
    logger.warning(
        "Megatron linear_cross_entropy Blackwell entry points are unavailable; "
        "using Slime chunked PyTorch fallback with vocab_chunk=%s.",
        _chunk_size(),
    )


apply_linear_ce_fallback_patch()
