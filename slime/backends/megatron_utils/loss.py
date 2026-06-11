import logging
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch

logger = logging.getLogger(__name__)
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
    vocab_parallel_reverse_kl,
    vocab_parallel_topk_reverse_kl,
)
from slime.utils.types import RolloutBatch

from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    apply_temperature: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and (optionally) applying temperature scaling,
    this function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for optional scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.
        max_seq_lens: Optional padded max sequence lengths per sample (for bshd).
        apply_temperature: If True (default), apply ``args.rollout_temperature``
            scaling to logits. Set to False when raw logits are needed, e.g.
            for full-vocabulary KL divergence computation.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    if apply_temperature and args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = mpu.get_context_parallel_rank()
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def _allgather_cp_redistribute(
    res: dict[str, list[torch.Tensor]],
    *,
    logits_local_len: int,
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> None:
    """Redistribute response tensors from allgather-CP layout to zigzag ring-attn layout.

    After allgather context parallelism, each rank holds a contiguous chunk of
    the global sequence.  This helper reconstructs per-sample full response
    tensors via a differentiable all-reduce and re-slices them into the zigzag
    CP pattern expected by downstream code.

    The *res* dict is modified **in-place**.

    Args:
        res: Dict mapping metric names to lists of per-sample tensors.
        logits_local_len: Local sequence length on this rank.
        args: Configuration (needs ``qkv_format``).
        total_lengths: Total sequence lengths (prompt + response) per sample.
        response_lengths: Response segment lengths per sample.
        max_seq_lens: Optional padded max sequence lengths per sample.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len

    for key, values in res.items():
        # Skip keys where all values are None (e.g. entropy when not computed)
        if all(v is None for v in values):
            continue

        # Determine reference dtype/device from first non-None value
        ref_value = next(v for v in values if v is not None)
        ref_dtype = ref_value.dtype
        ref_device = ref_value.device

        # Reconstruct full response tensors with each rank's contiguous contribution
        full_resps = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)

            if value is None or e <= s:
                # This rank has no response logprobs for this sample
                full_resp = torch.zeros(
                    response_length,
                    dtype=ref_dtype,
                    device=ref_device,
                    requires_grad=True,
                )
            else:
                resp_start = s - logit_global_start
                resp_end = e - logit_global_start
                full_resp = F.pad(value, (resp_start, response_length - resp_end))

            assert full_resp.size(0) == response_length, f"Expected {response_length}, got {full_resp.size(0)}"
            full_resps.append(full_resp)
            seq_start += total_length

        # Single differentiable all-reduce to gather full response from all CP ranks
        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)

        # Re-slice each sample into zigzag CP pattern
        new_values = []
        for idx, (full_resp, total_length, response_length) in enumerate(
            zip(all_cat.split(response_lengths, dim=0), total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
            new_values.append(
                slice_log_prob_with_cp(full_resp, total_length, response_length, args.qkv_format, max_seq_len)
            )

        res[key] = new_values


def _build_shifted_tokens(
    T: int,
    device: torch.device,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> torch.Tensor:
    """Build shifted target tokens for the full packed/padded logits."""
    cp_size = mpu.get_context_parallel_world_size()

    # --- zigzag CP: completely different layout ---
    if cp_size > 1 and not allgather_cp:
        full_tokens = torch.zeros(T, dtype=torch.long, device=device)
        end = 0
        for i, (tokens, total_length, response_length) in enumerate(
            zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            chunk_size_cp, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            for half, base in ((0, end), (1, end + chunk_size_cp)):
                lo = logits_offset[half][0] - chunks_offset[half][0]
                hi = logits_offset[half][1] - chunks_offset[half][0]
                full_tokens[base + lo : base + hi] = tokens[tokens_offset[half][0] : tokens_offset[half][1]]
            end += 2 * chunk_size_cp
        return full_tokens

    # --- cp1 and allgather-CP both build global shifted tokens the same way ---
    T_global = sum(total_lengths) if allgather_cp else T
    full_tokens = torch.zeros(T_global, dtype=torch.long, device=device)

    if qkv_format == "thd" or allgather_cp:
        offset = 0
        for tokens, total_length in zip(unconcat_tokens, total_lengths, strict=False):
            full_tokens[offset : offset + total_length - 1] = tokens[1:total_length]
            offset += total_length
    else:  # bshd, cp1
        for i, (tokens, total_length) in enumerate(zip(unconcat_tokens, total_lengths, strict=False)):
            seq_start = max_seq_lens[i] * i
            full_tokens[seq_start : seq_start + total_length - 1] = tokens[1:total_length]

    # allgather-CP: slice to local chunk
    if allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        chunk_start = cp_rank * T
        chunk_end = chunk_start + T
        if chunk_end <= T_global:
            return full_tokens[chunk_start:chunk_end].contiguous()
        local = torch.zeros(T, dtype=torch.long, device=device)
        valid = T_global - chunk_start
        if valid > 0:
            local[:valid] = full_tokens[chunk_start:]
        return local

    return full_tokens


def _extract_per_sample(
    log_prob_full: torch.Tensor,
    entropy_full: torch.Tensor | None,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Slice per-sample response log-probs/entropy from full-length 1-D tensors."""
    cp_size = mpu.get_context_parallel_world_size()
    log_probs_list: list[torch.Tensor] = []
    entropy_list: list[torch.Tensor] = []

    if cp_size > 1 and not allgather_cp:
        # zigzag CP
        pos = 0
        for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            chunk_size_cp, chunks_offset, logits_offset, _tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            lo0 = logits_offset[0][0] - chunks_offset[0][0]
            hi0 = logits_offset[0][1] - chunks_offset[0][0]
            lo1 = logits_offset[1][0] - chunks_offset[1][0]
            hi1 = logits_offset[1][1] - chunks_offset[1][0]

            lp = torch.cat(
                [
                    log_prob_full[pos + lo0 : pos + hi0],
                    log_prob_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                ],
                dim=0,
            )
            log_probs_list.append(lp)
            if entropy_full is not None:
                ent = torch.cat(
                    [
                        entropy_full[pos + lo0 : pos + hi0],
                        entropy_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                    ],
                    dim=0,
                )
                entropy_list.append(ent)
            pos += 2 * chunk_size_cp

    elif allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        local_len = log_prob_full.size(0)
        chunk_start = cp_rank * local_len
        chunk_end = chunk_start + local_len

        seq_start = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                log_probs_list.append(torch.zeros((0,), dtype=log_prob_full.dtype, device=log_prob_full.device))
                if entropy_full is not None:
                    entropy_list.append(torch.zeros((0,), dtype=entropy_full.dtype, device=entropy_full.device))
            else:
                log_probs_list.append(log_prob_full[s - chunk_start : e - chunk_start])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[s - chunk_start : e - chunk_start])
            seq_start += total_length

    else:
        # cp1
        if qkv_format == "thd":
            offset = 0
            for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
                end = offset + total_length
                start = end - response_length
                log_probs_list.append(log_prob_full[start - 1 : end - 1])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[start - 1 : end - 1])
                offset += total_length
        else:  # bshd
            for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
                end = max_seq_lens[i] * i + total_length
                start = end - response_length
                log_probs_list.append(log_prob_full[start - 1 : end - 1])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[start - 1 : end - 1])

    return log_probs_list, entropy_list


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    Computes on the **full** logits ``[T, V]`` tensor at once (instead of
    per-sample slicing) so backward traverses ``[T, V]`` only once, then
    extracts per-sample response portions.

    When ``entropy_coef == 0``, entropy is computed under ``torch.no_grad()``
    to avoid retaining the computation graph and to skip cloning.
    """
    assert non_loss_data
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    # Apply rollout temperature scaling to logits to match rollout-time log-probs.
    rollout_temperature = getattr(args, "rollout_temperature", 1.0)
    if rollout_temperature != 1.0:
        logits = logits / rollout_temperature
    logits = logits.contiguous()
    T = logits.size(0)
    device = logits.device
    tp_group = mpu.get_tensor_model_parallel_group()
    chunk_size = args.log_probs_chunk_size

    # --- build full shifted-token target tensor ---
    full_tokens = _build_shifted_tokens(
        T, device, unconcat_tokens, total_lengths, response_lengths, qkv_format, max_seq_lens, args.allgather_cp
    )

    # --- compute on full [T,V] logits at once via calculate_log_probs_and_entropy ---
    log_prob_full, entropy_full = calculate_log_probs_and_entropy(
        logits,
        full_tokens,
        tp_group,
        with_entropy=with_entropy,
        chunk_size=chunk_size,
    )
    log_prob_full = log_prob_full.squeeze(-1)  # [T, 1] -> [T]

    # --- extract per-sample response portions ---
    log_probs_list, entropy_list = _extract_per_sample(
        log_prob_full,
        entropy_full,
        total_lengths,
        response_lengths,
        qkv_format,
        max_seq_lens,
        args.allgather_cp,
    )

    res = {"log_probs": log_probs_list}
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=T,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=device), res


def get_logits_for_distill(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-sample response logits for full-vocab distillation (MOPD full_vocab mode).

    Similar to ``get_log_probs_and_entropy``, but returns the full logits tensor
    ``[R, V]`` per sample (where R is response length, V is vocab size) instead of
    log-probabilities. This is needed for computing exact KL divergence over the
    full vocabulary: D_KL(π_θ ∥ π_d) = Σ_y π_θ(y) [log π_θ(y) - log π_d(y)].

    No temperature scaling is applied — the raw logits are returned so the caller
    can apply the desired softmax/log_softmax independently.

    Args:
        logits: Model outputs with shape ``[1, T, V]``. Must be float32.
        args: Configuration (needs ``qkv_format``, ``allgather_cp``).
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.
        max_seq_lens: Optional padded max sequence lengths per sample (for bshd).

    Returns:
        Dict with key ``"logits"`` mapping to a list of ``[R, V]`` tensors per sample.
    """
    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    device = logits.device

    # Extract per-sample response logits chunks
    # NOTE: apply_temperature=False — raw logits are needed for correct
    # softmax/log_softmax in KL divergence computation.
    # get_responses handles qkv_format reshaping internally.
    logits_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        apply_temperature=False,
    ):
        logits_list.append(logits_chunk)

    res = {"logits": logits_list}

    # Handle allgather-CP redistribution
    # NOTE: _allgather_cp_redistribute assumes 1D per-sample tensors (log_probs, entropy).
    # Full-vocab logits are 2D [R_i, V], which is not supported by the current
    # redistribution logic.  Raise an explicit error so users don't get silent
    # shape mismatches.
    if args.allgather_cp:
        cp_size = getattr(mpu, "get_context_parallel_world_size", lambda: 1)()
        if cp_size > 1:
            raise NotImplementedError(
                "MOPD full_vocab/top_k (get_logits_for_distill) does not support "
                "allgather-CP with context_parallel_size > 1. The CP redistribution "
                "logic assumes 1D tensors but logits are 2D [R, V]. Please disable "
                "allgather_cp or set context_parallel_size=1 when using full_vocab/top_k mode."
            )
        _allgather_cp_redistribute(
            res,
            logits_local_len=logits_list[0].size(0) if logits_list else 0,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=device), res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration passed to `get_responses`; temperature scaling is
            disabled for value outputs.
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        apply_temperature=False,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=logits.size(1),
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=logits.device), res


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply on-policy distillation KL penalty to advantages.

    Computes reverse KL (student_logp - teacher_logp) and adds weighted penalty
    to advantages in-place. This is orthogonal to the base advantage estimator.

    Args:
        args: Configuration containing `use_opd` and `opd_kl_coef`.
        rollout_data: Dict containing "teacher_log_probs".
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of student log-probability tensors.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """

    if student_log_probs is None:
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")

    device = student_log_probs[0].device
    teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]

    reverse_kls = []
    for i, adv in enumerate(advantages):
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]
        advantages[i] = adv - args.opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)

    # Store reverse KL for logging
    rollout_data["opd_reverse_kl"] = reverse_kls


def apply_mopd_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply Multi-Teacher On-Policy Distillation (MOPD) to advantages.

    MOPD computes per-teacher reverse KL advantages and importance sampling weights,
    then applies a weighted proxy loss. The core formulas are:

        Â_MOPD,t = sg[log(π_domain(y_t|x,y_<t) / π_θ(y_t|x,y_<t))]
        w_t = sg[π_θ(y_t) / μ_θ(y_t)]  (clipped to [eps_low, eps_high], zeroed otherwise)
        L_MOPD(θ) = -E[1/|y| Σ_t w_t * Â_MOPD,t * log π_θ(y_t|x,y_<t)]

    When `mopd_alpha > 0`, the ORM advantage is combined:
        Â_MOPD,t = sg[log(π_domain/π_θ)] + α * Â_ORM

    Args:
        args: Configuration containing `use_mopd`, `mopd_alpha`, `mopd_eps_low`, `mopd_eps_high`,
              and `mopd_sampling_logprobs_key`.
        rollout_data: Dict containing "mopd_teacher_log_probs" (dict: domain -> list[Tensor])
                      and optionally the sampling log-probs key.
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of student (training) log-probability tensors.
    """

    if student_log_probs is None:
        return

    mopd_teacher_log_probs: dict[str, list[torch.Tensor]] = rollout_data.get("mopd_teacher_log_probs")
    if not mopd_teacher_log_probs:
        raise ValueError(
            "MOPD requires mopd_teacher_log_probs in rollout_data, but it is missing. "
            "Ensure teacher log-probs are collected during rollout or training."
        )

    # Get sampling log-probs μ_θ for importance sampling weight
    sampling_logprobs_key = args.mopd_sampling_logprobs_key
    sampling_log_probs = rollout_data.get(sampling_logprobs_key)
    if sampling_log_probs is None and sampling_logprobs_key == "rollout_log_probs":
        # Fall back to old_log_probs (which may be rollout_log_probs depending on config)
        sampling_log_probs = rollout_data.get("log_probs")
    if sampling_log_probs is None:
        raise ValueError(
            f"MOPD requires '{sampling_logprobs_key}' in rollout_data for importance sampling, " f"but it is missing."
        )

    device = student_log_probs[0].device
    sampling_log_probs = [s.to(device=device) for s in sampling_log_probs]

    # Compute MOPD advantages from each teacher and aggregate
    # For each teacher, compute reverse KL and IS weights, then sum weighted advantages
    all_mopd_advantages = []
    all_is_weights_list = []
    all_reverse_kls = []

    for _domain, teacher_lp_list in mopd_teacher_log_probs.items():
        domain_advantages = []
        domain_is_weights = []
        domain_reverse_kls = []

        for i in range(len(advantages)):
            # If this sample has no teacher log-probs for this domain (per-sample routing),
            # use zeros as placeholder — this domain contributes nothing to this sample.
            # Also detect fallback sentinel tensors (all -inf) that were inserted when
            # MOPD teacher requests failed, to avoid contaminating advantages with -inf.
            if teacher_lp_list[i] is None:
                domain_advantages.append(None)
                domain_is_weights.append(None)
                domain_reverse_kls.append(None)
                continue

            teacher_lp = teacher_lp_list[i].to(device=device)
            if teacher_lp.isinf().all():
                # All -inf: teacher data was unavailable (fallback sentinel).
                # Treat same as None — this domain contributes nothing.
                domain_advantages.append(None)
                domain_is_weights.append(None)
                domain_reverse_kls.append(None)
                continue

            # reverse_kl = log(π_domain(y_t)) - log(π_θ(y_t)), with stop-gradient
            # student_log_probs here is π_θ (the training engine log-probs)
            with torch.no_grad():
                reverse_kl = teacher_lp - student_log_probs[i]

                # Importance sampling weight: w_t = π_θ(y_t) / μ_θ(y_t)
                # = exp(student_log_probs[i] - sampling_log_probs[i])
                is_weight = torch.exp(student_log_probs[i] - sampling_log_probs[i])

                # Zero out weights outside [eps_low, eps_high]
                is_weight = torch.where(
                    (is_weight >= args.mopd_eps_low) & (is_weight <= args.mopd_eps_high),
                    is_weight,
                    torch.zeros_like(is_weight),
                )

            # MOPD advantage: Â_MOPD,t = reverse_kl + α * Â_ORM
            mopd_adv = reverse_kl
            if args.mopd_alpha > 0:
                mopd_adv = reverse_kl + args.mopd_alpha * advantages[i]

            domain_advantages.append(mopd_adv)
            domain_is_weights.append(is_weight)
            domain_reverse_kls.append(reverse_kl)

        all_mopd_advantages.append(domain_advantages)
        all_is_weights_list.append(domain_is_weights)
        all_reverse_kls.append(domain_reverse_kls)

    # Aggregate across teachers: average the weighted advantages
    # For each sample, only average over domains that have valid (non-None) entries.
    # This supports per-sample domain routing where different samples may use different teachers.
    aggregated_mopd_advantages = []
    aggregated_is_weights = []

    for i in range(len(advantages)):
        # Collect valid (non-None) teacher contributions for this sample
        valid_advs = [
            all_mopd_advantages[t][i] for t in range(len(all_mopd_advantages)) if all_mopd_advantages[t][i] is not None
        ]
        valid_is = [
            all_is_weights_list[t][i] for t in range(len(all_is_weights_list)) if all_is_weights_list[t][i] is not None
        ]

        if len(valid_advs) == 0:
            # No valid teachers for this sample — use zero advantages and zero IS weights
            aggregated_mopd_advantages.append(torch.zeros_like(advantages[i]))
            aggregated_is_weights.append(torch.zeros_like(advantages[i]))
        else:
            avg_adv = sum(valid_advs) / len(valid_advs)
            avg_is_weight = sum(valid_is) / len(valid_is)
            aggregated_mopd_advantages.append(avg_adv)
            aggregated_is_weights.append(avg_is_weight)

    # Store MOPD data for use in policy_loss_function
    rollout_data["mopd_advantages"] = aggregated_mopd_advantages
    rollout_data["mopd_is_weights"] = aggregated_is_weights

    # Also store per-teacher reverse KL for logging
    # Use zeros for samples that don't have this domain (per-sample routing)
    per_teacher_reverse_kl = {}
    for t_idx, domain in enumerate(mopd_teacher_log_probs.keys()):
        per_teacher_reverse_kl[domain] = [
            all_reverse_kls[t_idx][i] if all_reverse_kls[t_idx][i] is not None else torch.zeros_like(advantages[i])
            for i in range(len(advantages))
        ]
    rollout_data["mopd_reverse_kl"] = per_teacher_reverse_kl


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". When `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    If ``args.custom_advantage_function_path`` is set, it is called after KL computation
    and must populate ``rollout_data["advantages"]`` and
    ``rollout_data["returns"]``.

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    rollout_log_probs: list[torch.Tensor] | None = rollout_data.get("rollout_log_probs")
    log_probs: list[torch.Tensor] | None = (
        rollout_log_probs if args.use_rollout_logprobs else rollout_data.get("log_probs")
    )
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if not mpu.is_pipeline_last_stage():
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs or rollout_log_probs or values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]
    rollout_data["kl"] = kl

    if args.custom_advantage_function_path is not None:
        custom_adv_fn = load_function(args.custom_advantage_function_path)
        custom_adv_fn(args, rollout_data)
        advantages, returns = rollout_data["advantages"], rollout_data["returns"]

    elif args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
    if args.use_opd:
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    # Apply Multi-Teacher On-Policy Distillation (MOPD) to advantages
    # Skip token-level MOPD when using full_vocab distillation type;
    # in that mode, the KL is computed directly in the loss function.
    if args.use_mopd and getattr(args, "mopd_distill_type", "token_level") == "token_level":
        apply_mopd_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def apply_mopd_full_vocab_to_loss(
    args: Namespace,
    batch: RolloutBatch,
    student_logits_per_sample: list[torch.Tensor],
    teacher_logits_per_domain: dict[str, list[torch.Tensor | None]],
    loss_masks: list[torch.Tensor],
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    current_log_probs: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the full-vocabulary reverse KL divergence loss for MOPD.

    Instead of approximating the reverse KL from sampled tokens (token_level mode),
    this computes the exact KL divergence over the full vocabulary:

        D_KL(π_θ ∥ π_d) = Σ_y π_θ(y) [log π_θ(y) - log π_d(y)]

    For each teacher domain d, the per-token KL is computed and then averaged
    across teachers. Per-token importance sampling weights are applied identically
    to token_level mode.

    When mopd_alpha > 0, the total loss is:
        L = L_full_vocab_kl + alpha * L_pg_orm
    where L_pg_orm is the standard policy gradient loss with ORM advantages.
    When mopd_alpha == 0, L = L_full_vocab_kl (pure distillation).

    Args:
        args: Configuration containing MOPD parameters.
        batch: Mini-batch containing IS weight data and loss_masks.
        student_logits_per_sample: List of per-sample student logits [R_i, V].
        teacher_logits_per_domain: Dict mapping domain to list of per-sample
            teacher logits [R_i, V], with None for samples not routed to that domain.
        loss_masks: List of per-sample loss masks.
        sum_of_sample_mean: Reduction function for averaging.
        current_log_probs: List of per-sample log-probs from the current training
            forward pass. Used for importance sampling weight computation.
            If None, falls back to batch["log_probs"] (pre-training forward pass).

    Returns:
        Tuple of (kl_loss, metrics) where kl_loss is a scalar tensor and
        metrics is a dict with logging tensors.
    """
    # Get sampling log-probs μ_θ for importance sampling weight
    sampling_logprobs_key = getattr(args, "mopd_sampling_logprobs_key", "rollout_log_probs")
    sampling_log_probs = batch.get(sampling_logprobs_key)
    if sampling_logprobs_key == "rollout_log_probs" and sampling_log_probs is None:
        sampling_log_probs = batch.get("log_probs")
    if sampling_log_probs is None:
        raise ValueError(f"MOPD full_vocab requires '{sampling_logprobs_key}' in batch for importance sampling.")

    num_samples = len(student_logits_per_sample)
    if len(sampling_log_probs) != num_samples:
        raise ValueError(
            f"MOPD full_vocab: sampling_log_probs length ({len(sampling_log_probs)}) "
            f"!= student_logits length ({num_samples})."
        )
    all_kl_per_token = []  # will hold per-token KL tensors for all samples
    tp_group = mpu.get_tensor_model_parallel_group()
    # Stash per-domain per-sample KL for logging (detached)
    per_domain_kls: dict[str, list[torch.Tensor]] = {}

    # Collect per-sample KL contributions across all teacher domains.
    # For each sample, we average the KL across valid (non-None) teachers.
    for i in range(num_samples):
        R_i = student_logits_per_sample[i].size(0)
        sample_kl_values = []  # collect KL contributions from each valid teacher
        valid_teacher_count = 0

        for domain, teacher_logits_list in teacher_logits_per_domain.items():
            if i >= len(teacher_logits_list) or teacher_logits_list[i] is None:
                continue  # skip this domain for this sample

            teacher_logits_i = teacher_logits_list[i]  # [R_i, V_local]

            # D_KL(π_θ ∥ π_d) = Σ_y π_θ(y) [log π_θ(y) - log π_d(y)]
            # Uses TP-aware computation when vocab is sharded across TP ranks.
            kl_i = vocab_parallel_reverse_kl(
                student_logits_per_sample[i],
                teacher_logits_i,
                tp_group,
            )  # [R_i]
            sample_kl_values.append(kl_i)
            valid_teacher_count += 1

            # Save for per-domain logging
            if domain not in per_domain_kls:
                per_domain_kls[domain] = []
            per_domain_kls[domain].append(kl_i.detach())

        if valid_teacher_count > 0:
            # Average KL across valid teachers
            avg_kl_i = sum(sample_kl_values) / valid_teacher_count  # [R_i]
        else:
            avg_kl_i = torch.zeros(R_i, device=student_logits_per_sample[i].device)

        all_kl_per_token.append(avg_kl_i)

    # Compute IS weights
    # w_t = π_θ(y_t) / μ_θ(y_t)  clipped to [eps_low, eps_high]
    # We need the student's current log prob at the sampled token (π_θ(y_t)).
    # This comes from the current training forward pass (not the stale pre-training
    # pass in batch["log_probs"]). The caller passes these via current_log_probs.
    # Fall back to batch["log_probs"] only if current_log_probs is not provided.
    student_log_probs_at_sampled = current_log_probs if current_log_probs is not None else batch.get("log_probs")
    if student_log_probs_at_sampled is not None and len(student_log_probs_at_sampled) != num_samples:
        raise ValueError(
            f"MOPD full_vocab: student_log_probs length ({len(student_log_probs_at_sampled)}) "
            f"!= student_logits length ({num_samples})."
        )

    is_weight_per_sample = []
    for i in range(num_samples):
        with torch.no_grad():
            if student_log_probs_at_sampled is not None:
                # Use the per-token log probs from the current training forward pass
                s_lp_i = student_log_probs_at_sampled[i].to(device=student_logits_per_sample[i].device)
            else:
                # Fallback: zero IS weights (effectively disabling IS correction)
                s_lp_i = torch.zeros(
                    student_logits_per_sample[i].size(0),
                    device=student_logits_per_sample[i].device,
                )
            samp_lp_i = sampling_log_probs[i].to(device=s_lp_i.device)
            is_w_i = torch.exp(s_lp_i - samp_lp_i)
            # Zero out weights outside [eps_low, eps_high]
            is_w_i = torch.where(
                (is_w_i >= args.mopd_eps_low) & (is_w_i <= args.mopd_eps_high),
                is_w_i,
                torch.zeros_like(is_w_i),
            )
            is_weight_per_sample.append(is_w_i)

    # Apply IS weights to KL: per-token KL * IS weight * loss_mask
    weighted_kl_tokens = []
    for i in range(num_samples):
        mask_i = loss_masks[i].to(device=all_kl_per_token[i].device)
        # Mask and weight the KL
        weighted_kl_i = all_kl_per_token[i] * is_weight_per_sample[i] * mask_i  # [R_i]
        # Sum over tokens in the response, divide by response length for mean
        R_i = mask_i.sum().clamp(min=1)  # number of valid tokens
        weighted_kl_tokens.append(weighted_kl_i.sum() / R_i)

    # Average across samples
    if len(weighted_kl_tokens) > 0:
        kl_loss = torch.stack(weighted_kl_tokens).mean()
    else:
        kl_loss = torch.tensor(0.0, device=student_logits_per_sample[0].device)

    # Logging metrics
    all_kl_cat = torch.cat(all_kl_per_token, dim=0)
    kl_mean = sum_of_sample_mean(all_kl_cat)

    is_weights_cat = torch.cat(is_weight_per_sample, dim=0)
    is_weight_mean = sum_of_sample_mean(is_weights_cat)
    is_nonzero_frac = sum_of_sample_mean((is_weights_cat != 0).float())

    metrics = {
        "mopd_fv_kl": kl_mean.clone().detach(),
        "mopd_is_weight_mean": is_weight_mean.clone().detach(),
        "mopd_is_nonzero_frac": is_nonzero_frac.clone().detach(),
    }

    # Per-teacher KL for logging (re-use KL values computed in the main loop).
    # Iterate over ALL configured teacher domains (not just per_domain_kls) so
    # that every microbatch emits the same set of metric keys.  Without this
    # Megatron's cross-microbatch loss reduction (model.py: values += x["values"])
    # can crash with a tensor-size mismatch when different microbatches have
    # different subsets of active domains.
    for domain in teacher_logits_per_domain:
        if domain in per_domain_kls and len(per_domain_kls[domain]) > 0:
            metrics[f"mopd_fv_kl/{domain}"] = (
                sum_of_sample_mean(torch.cat(per_domain_kls[domain], dim=0)).clone().detach()
            )
        else:
            metrics[f"mopd_fv_kl/{domain}"] = torch.tensor(0.0, device=all_kl_cat.device)

    return kl_loss, metrics


def apply_mopd_topk_to_loss(
    args: Namespace,
    batch: RolloutBatch,
    student_logits_per_sample: list[torch.Tensor],
    teacher_topk_logits_per_domain: dict[str, list[torch.Tensor | None]],
    teacher_topk_indices_per_domain: dict[str, list[torch.Tensor | None]],
    loss_masks: list[torch.Tensor],
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    current_log_probs: list[torch.Tensor] | None = None,
    teacher_topk_log_sum_exp_per_domain: dict[str, list[torch.Tensor | None]] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the top-k approximate reverse KL divergence loss for MOPD.

    Instead of computing the exact full-vocab KL (which stores [R, V] per sample),
    this uses teacher's top-k logits plus a tail probability correction:

        KL ≈ KL_topk + KL_tail

    where:
        KL_topk = Σ_{y ∈ top-k} π_s(y) [log π_s(y) - log π_t(y)]
        KL_tail ≈ π_s_tail * log(π_s_tail / π_t_tail)

    Teacher provides pre-computed top-k logits and indices, while student has
    full logits. This reduces memory from O(R*V) to O(R*k) per sample per teacher.

    Args:
        args: Configuration containing MOPD parameters including mopd_topk_k.
        batch: Mini-batch containing IS weight data and loss_masks.
        student_logits_per_sample: List of per-sample student logits [R_i, V_local].
        teacher_topk_logits_per_domain: Dict mapping domain to list of per-sample
            teacher top-k logits [R_i, k], with None for samples not routed to that domain.
        teacher_topk_indices_per_domain: Dict mapping domain to list of per-sample
            teacher top-k LOCAL indices [R_i, k] (within TP shard), with None for
            samples not routed to that domain.
        loss_masks: List of per-sample loss masks.
        sum_of_sample_mean: Reduction function for averaging.
        current_log_probs: List of per-sample log-probs from the current training
            forward pass. Used for importance sampling weight computation.
            If None, falls back to batch["log_probs"].
        teacher_topk_log_sum_exp_per_domain: Optional dict mapping domain to list of
            per-sample teacher log_sum_exp tensors [R_i] (computed from full-vocab logits
            during Megatron teacher forward pass). Used for exact tail mass estimation
            in Megatron mode. When provided, teacher_tail_mass is computed exactly as
            1 - sum(exp(topk_logits - log_sum_exp)) instead of the uniform assumption
            (V - V_eff) / V.

    Returns:
        Tuple of (kl_loss, metrics) where kl_loss is a scalar tensor and
        metrics is a dict with logging tensors.
    """
    # Get sampling log-probs μ_θ for importance sampling weight
    sampling_logprobs_key = getattr(args, "mopd_sampling_logprobs_key", "rollout_log_probs")
    sampling_log_probs = batch.get(sampling_logprobs_key)
    if sampling_logprobs_key == "rollout_log_probs" and sampling_log_probs is None:
        sampling_log_probs = batch.get("log_probs")
    if sampling_log_probs is None:
        raise ValueError(f"MOPD top_k requires '{sampling_logprobs_key}' in batch for importance sampling.")

    vocab_size = args.vocab_size
    num_samples = len(student_logits_per_sample)
    if len(sampling_log_probs) != num_samples:
        raise ValueError(
            f"MOPD top_k: sampling_log_probs length ({len(sampling_log_probs)}) "
            f"!= student_logits length ({num_samples})."
        )

    tp_group = mpu.get_tensor_model_parallel_group()
    all_kl_per_token = []
    per_domain_kls: dict[str, list[torch.Tensor]] = {}

    for i in range(num_samples):
        R_i = student_logits_per_sample[i].size(0)
        sample_kl_values = []
        valid_teacher_count = 0

        for domain in teacher_topk_logits_per_domain:
            if i >= len(teacher_topk_logits_per_domain[domain]) or teacher_topk_logits_per_domain[domain][i] is None:
                continue

            t_topk_logits = teacher_topk_logits_per_domain[domain][i]  # [R_i, k]

            # IMPORTANT: Do NOT skip the vocab_parallel_topk_reverse_kl call even
            # when all teacher logits are -inf.  Each TP rank independently shards
            # the top-k tokens into its vocab range, so one rank may see all -inf
            # (no tokens in its shard) while another rank has valid entries.
            # Skipping on only some ranks creates an inconsistent TP collective call
            # (all_reduce inside vocab_parallel_topk_reverse_kl), causing an
            # irreversible NCCL deadlock.  When all entries are -inf,
            # vocab_parallel_topk_reverse_kl correctly produces KL=0 (the
            # valid_topk_mask is all-False), so the numerical result is identical
            # -- but the collective operations remain consistent across TP ranks.

            t_topk_indices = teacher_topk_indices_per_domain[domain][i]  # [R_i, k]

            # Get teacher log_sum_exp for exact tail mass (Megatron mode only)
            t_topk_log_sum_exp = None
            if teacher_topk_log_sum_exp_per_domain and domain in teacher_topk_log_sum_exp_per_domain:
                if (
                    i < len(teacher_topk_log_sum_exp_per_domain[domain])
                    and teacher_topk_log_sum_exp_per_domain[domain][i] is not None
                ):
                    t_topk_log_sum_exp = teacher_topk_log_sum_exp_per_domain[domain][i]  # [R_i]

            kl_i = vocab_parallel_topk_reverse_kl(
                student_logits_per_sample[i],
                t_topk_logits,
                t_topk_indices,
                vocab_size,
                tp_group,
                is_log_probs=(getattr(args, "mopd_teacher_mode", "megatron") == "sglang"),
                teacher_log_sum_exp=t_topk_log_sum_exp,
            )  # [R_i]
            sample_kl_values.append(kl_i)
            valid_teacher_count += 1

            if domain not in per_domain_kls:
                per_domain_kls[domain] = []
            per_domain_kls[domain].append(kl_i.detach())

        if valid_teacher_count > 0:
            avg_kl_i = sum(sample_kl_values) / valid_teacher_count
        else:
            avg_kl_i = torch.zeros(R_i, device=student_logits_per_sample[i].device)

        all_kl_per_token.append(avg_kl_i)

    # Compute IS weights (identical logic to full_vocab)
    student_log_probs_at_sampled = current_log_probs if current_log_probs is not None else batch.get("log_probs")
    if student_log_probs_at_sampled is not None and len(student_log_probs_at_sampled) != num_samples:
        raise ValueError(
            f"MOPD top_k: student_log_probs length ({len(student_log_probs_at_sampled)}) "
            f"!= student_logits length ({num_samples})."
        )

    is_weight_per_sample = []
    for i in range(num_samples):
        with torch.no_grad():
            if student_log_probs_at_sampled is not None:
                s_lp_i = student_log_probs_at_sampled[i].to(device=student_logits_per_sample[i].device)
            else:
                s_lp_i = torch.zeros(
                    student_logits_per_sample[i].size(0),
                    device=student_logits_per_sample[i].device,
                )
            samp_lp_i = sampling_log_probs[i].to(device=s_lp_i.device)
            is_w_i = torch.exp(s_lp_i - samp_lp_i)
            is_w_i = torch.where(
                (is_w_i >= args.mopd_eps_low) & (is_w_i <= args.mopd_eps_high),
                is_w_i,
                torch.zeros_like(is_w_i),
            )
            is_weight_per_sample.append(is_w_i)

    # Apply IS weights to KL
    weighted_kl_tokens = []
    for i in range(num_samples):
        mask_i = loss_masks[i].to(device=all_kl_per_token[i].device)
        weighted_kl_i = all_kl_per_token[i] * is_weight_per_sample[i] * mask_i
        R_i = mask_i.sum().clamp(min=1)
        weighted_kl_tokens.append(weighted_kl_i.sum() / R_i)

    if len(weighted_kl_tokens) > 0:
        kl_loss = torch.stack(weighted_kl_tokens).mean()
    else:
        kl_loss = torch.tensor(0.0, device=student_logits_per_sample[0].device)

    # Logging metrics
    all_kl_cat = torch.cat(all_kl_per_token, dim=0)
    kl_mean = sum_of_sample_mean(all_kl_cat)

    is_weights_cat = torch.cat(is_weight_per_sample, dim=0)
    is_weight_mean = sum_of_sample_mean(is_weights_cat)
    is_nonzero_frac = sum_of_sample_mean((is_weights_cat != 0).float())

    metrics = {
        "mopd_topk_kl": kl_mean.clone().detach(),
        "mopd_is_weight_mean": is_weight_mean.clone().detach(),
        "mopd_is_nonzero_frac": is_nonzero_frac.clone().detach(),
    }

    for domain in teacher_topk_logits_per_domain:
        if domain in per_domain_kls and len(per_domain_kls[domain]) > 0:
            metrics[f"mopd_topk_kl/{domain}"] = (
                sum_of_sample_mean(torch.cat(per_domain_kls[domain], dim=0)).clone().detach()
            )
        else:
            # No samples contributed valid teacher data for this domain in this
            # microbatch.  Emit a zero metric so that every microbatch produces
            # the same set of metric keys — this is required for Megatron's
            # loss-reduction across microbatches which uses tensor addition
            # (model.py: values += x["values"]) and demands identical sizes.
            metrics[f"mopd_topk_kl/{domain}"] = torch.tensor(0.0, device=all_kl_cat.device)

    return kl_loss, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    if not args.use_rollout_logprobs and not old_log_probs:
        old_log_probs = [log_prob.detach() for log_prob in log_probs]
    train_log_probs_for_tis = batch.get("log_probs")
    if not train_log_probs_for_tis:
        train_log_probs_for_tis = [log_prob.detach() for log_prob in log_probs]

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    # Save list-form log_probs before concatenation for potential use in MOPD full_vocab IS weights
    current_log_probs_list = log_probs
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply MOPD token_level: replace advantages with mopd_advantages and apply IS weights
    # L_MOPD(θ) = -E[1/|y| Σ_t w_t * Â_MOPD,t * log π_θ(y_t|x,y_<t)]
    # We recompute pg_loss with mopd_advantages and then multiply by IS weights.
    mopd_distill_type = (
        getattr(args, "mopd_distill_type", "token_level") if getattr(args, "use_mopd", False) else "none"
    )
    use_mopd_full_vocab = mopd_distill_type == "full_vocab"
    use_mopd_top_k = mopd_distill_type == "top_k"
    # token_level mode uses advantages + IS weights in the standard pg_loss
    use_mopd_logits_based = use_mopd_full_vocab or use_mopd_top_k
    mopd_fv_metrics = {}
    if getattr(args, "use_mopd", False) and not use_mopd_logits_based and "mopd_advantages" in batch:
        mopd_advantages = torch.cat(batch["mopd_advantages"], dim=0).detach()
        pg_loss_mopd, _ = compute_policy_loss(ppo_kl, mopd_advantages, args.eps_clip, args.eps_clip_high)
        pg_loss = pg_loss_mopd

    if getattr(args, "use_mopd", False) and not use_mopd_logits_based and "mopd_is_weights" in batch:
        mopd_is_weights = torch.cat(batch["mopd_is_weights"], dim=0).detach()
        pg_loss = pg_loss * mopd_is_weights

    # MOPD full_vocab: compute full-vocabulary reverse KL divergence loss
    # L = (1/D) Σ_d w_d · D_KL(π_θ ∥ π_d) + alpha * pg_loss
    if use_mopd_full_vocab:
        # Extract per-sample student logits from the forward pass.
        # NOTE: apply_temperature=False — raw logits are needed for correct
        # softmax/log_softmax in KL divergence computation.
        student_logits_per_sample = []
        for logits_chunk, _ in get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
            apply_temperature=False,
        ):
            student_logits_per_sample.append(logits_chunk)

        # Collect teacher logits per domain from the batch
        mopd_teachers_parsed = getattr(args, "_mopd_teachers_parsed", [])
        teacher_logits_per_domain = {}
        for teacher_cfg in mopd_teachers_parsed:
            domain = teacher_cfg["domain"]
            logits_key = f"mopd_teacher_{domain}_fv_logits"
            if logits_key in batch and batch[logits_key] is not None:
                teacher_logits_per_domain[domain] = batch[logits_key]

        if teacher_logits_per_domain:
            fv_kl_loss, mopd_fv_metrics = apply_mopd_full_vocab_to_loss(
                args=args,
                batch=batch,
                student_logits_per_sample=student_logits_per_sample,
                teacher_logits_per_domain=teacher_logits_per_domain,
                loss_masks=batch["loss_masks"],
                sum_of_sample_mean=sum_of_sample_mean,
                current_log_probs=current_log_probs_list,
            )
            # Store the fv_kl_loss for later combination with pg_loss.
            # The actual combination happens after pg_loss is reduced.
            # When alpha == 0 (pure distillation): loss = fv_kl_loss
            # When alpha > 0: loss = fv_kl_loss + alpha * pg_loss
            batch["_mopd_fv_kl_loss"] = fv_kl_loss
        else:
            logger.warning(
                "MOPD full_vocab enabled but no teacher logits found in batch. Skipping full_vocab KL loss."
            )

        # Ensure per-domain metric keys AND base MOPD metric keys exist for
        # ALL configured teacher domains, even when the batch contains no valid
        # data for some (or all) domains.  Megatron's loss-reduction
        # (model.py: values += x["values"]) requires every microbatch to emit
        # the same set of metric keys; missing keys cause a tensor-size
        # mismatch across microbatches.
        _device = logits.device
        for teacher_cfg in mopd_teachers_parsed:
            domain = teacher_cfg["domain"]
            _domain_key = f"mopd_fv_kl/{domain}"
            if _domain_key not in mopd_fv_metrics:
                mopd_fv_metrics[_domain_key] = torch.tensor(0.0, device=_device)
        # Ensure base MOPD metrics are present even when no teacher data was
        # available for the entire microbatch (apply_mopd_full_vocab_to_loss not called).
        for _base_key in ("mopd_fv_kl", "mopd_is_weight_mean", "mopd_is_nonzero_frac"):
            if _base_key not in mopd_fv_metrics:
                mopd_fv_metrics[_base_key] = torch.tensor(0.0, device=_device)

    # MOPD top_k: compute top-k approximate reverse KL divergence loss
    # L = (1/D) Σ_d w_d · KL_topk+d(π_θ ∥ π_d) + alpha * pg_loss
    if use_mopd_top_k:
        student_logits_per_sample = []
        for logits_chunk, _ in get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
            apply_temperature=False,
        ):
            student_logits_per_sample.append(logits_chunk)

        # Collect teacher top-k logits and indices per domain from the batch
        mopd_teachers_parsed = getattr(args, "_mopd_teachers_parsed", [])
        teacher_topk_logits_per_domain = {}
        teacher_topk_indices_per_domain = {}
        teacher_topk_log_sum_exp_per_domain = {}
        for teacher_cfg in mopd_teachers_parsed:
            domain = teacher_cfg["domain"]
            topk_logits_key = f"mopd_teacher_{domain}_topk_logits"
            topk_indices_key = f"mopd_teacher_{domain}_topk_indices"
            topk_log_sum_exp_key = f"mopd_teacher_{domain}_topk_log_sum_exp"
            if topk_logits_key in batch and batch[topk_logits_key] is not None:
                teacher_topk_logits_per_domain[domain] = batch[topk_logits_key]
                teacher_topk_indices_per_domain[domain] = batch[topk_indices_key]
                # log_sum_exp is only available in Megatron mode (computed from full logits)
                if topk_log_sum_exp_key in batch and batch[topk_log_sum_exp_key] is not None:
                    teacher_topk_log_sum_exp_per_domain[domain] = batch[topk_log_sum_exp_key]

        if teacher_topk_logits_per_domain:
            topk_kl_loss, mopd_fv_metrics = apply_mopd_topk_to_loss(
                args=args,
                batch=batch,
                student_logits_per_sample=student_logits_per_sample,
                teacher_topk_logits_per_domain=teacher_topk_logits_per_domain,
                teacher_topk_indices_per_domain=teacher_topk_indices_per_domain,
                teacher_topk_log_sum_exp_per_domain=teacher_topk_log_sum_exp_per_domain,
                loss_masks=batch["loss_masks"],
                sum_of_sample_mean=sum_of_sample_mean,
                current_log_probs=current_log_probs_list,
            )
            batch["_mopd_fv_kl_loss"] = topk_kl_loss
        else:
            logger.warning("MOPD top_k enabled but no teacher top-k data found in batch. Skipping top_k KL loss.")

        # Ensure per-domain metric keys AND base MOPD metric keys exist for
        # ALL configured teacher domains, even when the batch contains no valid
        # data for some (or all) domains.  Megatron's loss-reduction
        # (model.py: values += x["values"]) requires every microbatch to emit
        # the same set of metric keys; missing keys cause a tensor-size
        # mismatch across microbatches.
        _device = logits.device
        for teacher_cfg in mopd_teachers_parsed:
            domain = teacher_cfg["domain"]
            _domain_key = f"mopd_topk_kl/{domain}"
            if _domain_key not in mopd_fv_metrics:
                mopd_fv_metrics[_domain_key] = torch.tensor(0.0, device=_device)
        # Ensure base MOPD metrics are present even when no teacher data was
        # available for the entire microbatch (apply_mopd_topk_to_loss not called).
        for _base_key in ("mopd_topk_kl", "mopd_is_weight_mean", "mopd_is_nonzero_frac"):
            if _base_key not in mopd_fv_metrics:
                mopd_fv_metrics[_base_key] = torch.tensor(0.0, device=_device)

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": train_log_probs_for_tis,
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with
        # modified_response_masks for numerator correction (rejected tokens
        # zeroed in pg_loss). Denominators stay the precomputed per-rollout
        # totals from ``rollout_mask_sums`` (based on original loss_masks) —
        # same normalizer as the outer reducer, so pg_loss and the rest of the
        # reported metrics live in the same per-rollout-mean space.
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            batch["rollout_mask_sums"],
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    # MOPD logits-based distillation (full_vocab / top_k): combine KL loss with pg loss
    # L = L_kl + alpha * L_pg
    # When alpha == 0 (pure distillation): L = L_kl (pg_loss is zeroed out)
    # When alpha > 0: L = L_kl + alpha * L_pg (ORM policy gradient)
    if use_mopd_logits_based and "_mopd_fv_kl_loss" in batch:
        kl_distill_loss = batch.pop("_mopd_fv_kl_loss")
        if args.mopd_alpha > 0:
            # Combine: distillation KL + alpha * policy gradient loss
            loss = kl_distill_loss + args.mopd_alpha * loss
        else:
            # Pure distillation: only use distillation KL loss
            loss = kl_distill_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Add OPD metrics if available
    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    # Log MOPD metrics (IS weights and per-teacher reverse KL are already applied during
    # advantage computation and pg_loss re-weighting in the MOPD section above)
    if getattr(args, "use_mopd", False) and not use_mopd_full_vocab and "mopd_is_weights" in batch:
        mopd_is_weights = torch.cat(batch["mopd_is_weights"], dim=0)
        reported_loss["mopd_is_weight_mean"] = sum_of_sample_mean(mopd_is_weights).clone().detach()
        mopd_is_nonzero = (mopd_is_weights != 0).float()
        reported_loss["mopd_is_nonzero_frac"] = sum_of_sample_mean(mopd_is_nonzero).clone().detach()

        if "mopd_reverse_kl" in batch:
            # Iterate over ALL configured teacher domains — not just the
            # keys present in this microbatch — so that every microbatch
            # produces the same set of metric keys (required for Megatron's
            # loss-reduction across microbatches).
            _all_mopd_domains = [t["domain"] for t in getattr(args, "_mopd_teachers_parsed", [])]
            _mopd_reverse_kl_domains = (
                _all_mopd_domains if _all_mopd_domains else list(batch["mopd_reverse_kl"].keys())
            )
            for domain in _mopd_reverse_kl_domains:
                if domain in batch["mopd_reverse_kl"]:
                    domain_kl_tensor = torch.cat(batch["mopd_reverse_kl"][domain], dim=0)
                    reported_loss[f"mopd_reverse_kl/{domain}"] = sum_of_sample_mean(domain_kl_tensor).clone().detach()
                else:
                    reported_loss[f"mopd_reverse_kl/{domain}"] = torch.tensor(0.0, device=mopd_is_weights.device)

        if "mopd_advantages" in batch:
            mopd_advantages = torch.cat(batch["mopd_advantages"], dim=0)
            reported_loss["mopd_advantage_mean"] = sum_of_sample_mean(mopd_advantages).clone().detach()

    # Log MOPD logits-based distillation metrics (full_vocab / top_k).
    # mopd_fv_metrics already contains zero-valued entries for domains that
    # had no valid teacher data in this microbatch (see apply_mopd_topk_to_loss
    # and apply_mopd_full_vocab_to_loss), ensuring consistent key sets.
    if use_mopd_logits_based:
        for key, value in mopd_fv_metrics.items():
            reported_loss[key] = value

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    step_global_batch_size: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        step_global_batch_size: Sample count for the current training step
            (total across DP). Replaces the legacy ``args.global_batch_size``
            fallback so the train side stops depending on "every DP rank holds
            the same N samples".
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        batch["rollout_mask_sums"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean, use_reentrant=False)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # With allgather-CP, some CP ranks may have no loss-contributing tokens (e.g., all
    # padding). Without this, gradient doesn't flow through their attention path, so
    # the CP gather's backward (reduce-scatter) is not called, deadlocking other CP
    # ranks that call it. Adding this zero loss forces autograd to traverse the full
    # graph on every rank without changing gradient values.
    if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    if not args.calculate_per_token_loss:
        loss = (
            loss
            * num_microbatches
            / step_global_batch_size
            * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    return (
        loss,
        (num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device)),
        {
            "keys": list(log.keys()),
            # values[0] is the consumer's reporting denominator after
            # all-reduce. For per-token-loss it must equal step total tokens
            # (only known by summing per-mb num_tokens across mbs / DP). For
            # per-rollout-mean it is a constant — ``step_global_batch_size`` —
            # so we leave a 0 placeholder here and let ``train_one_step``
            # substitute the constant directly, instead of routing it through
            # per-mb fractions.
            "values": torch.tensor(
                [
                    num_tokens if args.calculate_per_token_loss else 0,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
