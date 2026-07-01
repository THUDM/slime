import logging
import os
from argparse import Namespace
from collections.abc import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training.global_vars import get_args
from torch.nn.utils.rnn import pad_sequence

from slime.utils import train_metric_utils
from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step
from slime.utils.types import RolloutBatch

from ...utils import logging_utils
from .cp_utils import (
    gather_and_reduce_log_dict,
    get_sum_of_sample_mean,
    maybe_padded_total_lengths,
    rollout_log_metric_contribution,
    slice_with_cp,
)

logger = logging.getLogger(__name__)

PAD_RULES = {
    "input_features": dict(
        transpose=(0, 2),
        padding_value=0.0,
    ),
    "feature_attention_mask": dict(
        transpose=(0, 1),
        padding_value=0,
    ),
}


def has_multimodal_train_inputs(multimodal_train_inputs) -> bool:
    if multimodal_train_inputs is None:
        return False
    if isinstance(multimodal_train_inputs, dict):
        return any(has_multimodal_train_inputs(value) for value in multimodal_train_inputs.values())
    if isinstance(multimodal_train_inputs, (list, tuple)):
        return any(has_multimodal_train_inputs(value) for value in multimodal_train_inputs)
    if torch.is_tensor(multimodal_train_inputs):
        return multimodal_train_inputs.numel() > 0
    return True


def qwen_vl_unsplit_only_with_mm() -> bool:
    return os.environ.get("SLIME_QWENVL_UNSPLIT_ONLY_WITH_MM", "0").lower() in {"1", "true", "yes", "on"}


def qwen_vl_text_language_fastpath() -> bool:
    return os.environ.get("SLIME_QWENVL_TEXT_LANGUAGE_FASTPATH", "0").lower() in {"1", "true", "yes", "on"}


def qwen_vl_text_fastpath_requires_unsplit_input() -> bool:
    args = get_args()
    return (
        qwen_vl_text_language_fastpath()
        and os.environ.get("SLIME_QWENVL_TEXT_FASTPATH_LOCAL_MROPE", "1").lower() not in {"1", "true", "yes", "on"}
        and getattr(args, "position_embedding_type", "rope") == "mrope"
        and getattr(args, "qkv_format", "thd") == "thd"
        and mpu.get_context_parallel_world_size() > 1
    )


def pad_and_flatten(
    tensor_list,
    transpose=None,
    padding_value=0,
):
    if len(tensor_list) == 1:
        t = tensor_list[0]
        return t, [t.size(0)]

    if transpose is not None:
        tensor_list = [t.transpose(*transpose) for t in tensor_list]

    padded = pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)

    if transpose is not None:
        padded_list = [t.transpose(*transpose) for t in padded]
    else:
        padded_list = [t for t in padded]

    num_items = [t.size(0) for t in padded_list]
    return torch.cat(padded_list, dim=0), num_items


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
    allgather_cp: bool = False,
) -> dict[str, torch.Tensor | PackedSeqParams | list[torch.Tensor] | None]:
    """
    Generate a CP-ready micro-batch with packed sequence parameters.

    Steps:
    - Fetch raw fields via iterator.
    - Save original token tensors under "unconcat_tokens".
    - Slice tokens into two chunks for Context Parallelism (CP), concatenate, and pad to a configurable multiple.
    - Build cu_seqlens and `PackedSeqParams` with T-H-D layout (T: sequence length, H: attention heads, D: head dimension).

    Args:
        data_iterator: Iterator providing micro-batch data.
        keys: List of keys to fetch from the iterator.
        pad_multiplier: Multiplier for padding size calculation (default: 128).

    Returns a dict including:
    - "tokens": torch.LongTensor of shape [1, T_padded] on the current CUDA device
    - "unconcat_tokens": list[torch.LongTensor] for the micro-batch before CP slicing/concat
    - "packed_seq_params": PackedSeqParams with T-H-D settings (cu_seqlens on CUDA, dtype=int)
    Plus any other requested keys forwarded from the iterator.
    """

    assert "tokens" in keys
    batch = data_iterator.get_next(keys)

    tokens = batch["tokens"]
    # use 0 as the pad token id should be fine?
    pad_token_id = 0
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier

    # for cp, we need all tokens to calculate logprob
    batch["unconcat_tokens"] = tokens

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if qkv_format == "bshd":
        max_seqlen = batch["max_seq_lens"][0]
        assert max([t.size(0) for t in tokens]) <= max_seqlen

        if cp_size > 1:
            chunk_size = (max_seqlen + 2 * cp_size - 1) // (2 * cp_size)
            padded_len = 2 * cp_size * chunk_size
            unsplit = [F.pad(t, (0, padded_len - t.size(0)), value=pad_token_id) for t in tokens]
            batch["unsplit_tokens"] = torch.stack(unsplit)

        tokens = [slice_with_cp(t, pad_token_id, qkv_format, max_seqlen) for t in tokens]
        tokens = torch.stack(tokens)
        packed_seq_params = None

    elif qkv_format == "thd":
        is_vl_model = has_multimodal_train_inputs(batch.get("multimodal_train_inputs"))
        uses_unsplit_forward = getattr(get_args(), "uses_unsplit_forward", False)
        needs_unsplit_input = is_vl_model or (
            uses_unsplit_forward
            and (not qwen_vl_unsplit_only_with_mm() or qwen_vl_text_fastpath_requires_unsplit_input())
        )
        if needs_unsplit_input and cp_size > 1:
            tp_size = mpu.get_tensor_model_parallel_world_size()
            align_size = tp_size * cp_size * 2
            device = tokens[0].device

            seqlens = torch.tensor([t.size(0) for t in tokens], dtype=torch.int32, device=device)
            seqlens_padded = (seqlens + align_size - 1) // align_size * align_size
            cu_seqlens_padded = torch.zeros(len(tokens) + 1, dtype=torch.int32, device=device)
            cu_seqlens_padded[1:] = torch.cumsum(seqlens_padded, dim=0)
            max_seqlen_padded = int(seqlens_padded.max().item())

            unsplit_tokens = pad_sequence(tokens, batch_first=True, padding_value=pad_token_id)
            unsplit_attention_mask = torch.zeros_like(unsplit_tokens, dtype=torch.bool)
            for i, seqlen in enumerate(seqlens.tolist()):
                unsplit_attention_mask[i, :seqlen] = True

            batch["unsplit_tokens"] = unsplit_tokens
            batch["unsplit_attention_mask"] = unsplit_attention_mask
            batch["vlm_packed_seq_params"] = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens_padded,
                cu_seqlens_kv=cu_seqlens_padded,
                max_seqlen_q=max_seqlen_padded,
                max_seqlen_kv=max_seqlen_padded,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv_padded=cu_seqlens_padded,
            )
            batch["padded_total_lengths"] = seqlens_padded.tolist()

        if allgather_cp:
            # DSA mode: concatenate all sequences first, then slice once with CP.
            # We also pad the *global* concatenated stream to make per-rank chunks equal.
            cu_seqlens_list: list[int] = [0]
            for t in tokens:
                cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))

        # Pad global stream so (1) divisible by cp_size (equal chunks),
        # (2) divisible by pad_size (reduce fragmentation).
        global_pad_size = cp_size * pad_size
        pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
        if pad != 0:
            tokens = F.pad(tokens, (0, pad), value=pad_token_id)
            cu_seqlens_list.append(cu_seqlens_list[-1] + pad)

            # Pad global stream so (1) divisible by cp_size (equal chunks),
            # (2) divisible by pad_size (reduce fragmentation).
            global_pad_size = cp_size * pad_size
            pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens_list.append(cu_seqlens_list[-1] + pad)

            cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int, device=torch.cuda.current_device())
            tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
        else:
            tokens = [slice_with_cp(t, pad_token_id, qkv_format) for t in tokens]

            cu_seqlens = [0]
            for t in tokens:
                cu_seqlens.append(cu_seqlens[-1] + t.size(0))

            tokens = torch.cat(tokens)

            # Always pad to reduce memory fragmentation and maybe make the computation faster
            pad = (pad_size - tokens.size(0) % pad_size) % pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens.append(cu_seqlens[-1] + pad)

            # thd requires the cu_seqlens to be of the origin length
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int).cuda() * cp_size

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        packed_seq_params_kwargs = {}
        if (
            cp_size > 1
            and uses_unsplit_forward
            and qwen_vl_unsplit_only_with_mm()
            and not is_vl_model
            and not allgather_cp
        ):
            packed_seq_params_kwargs = {
                "cu_seqlens_q_padded": cu_seqlens,
                "cu_seqlens_kv_padded": cu_seqlens,
            }

        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
            **packed_seq_params_kwargs,
        )

        tokens = tokens.unsqueeze(0)
    else:
        tokens = [slice_with_cp(t, pad_token_id) for t in tokens]

        cu_seqlens = [0]
        for t in tokens:
            cu_seqlens.append(cu_seqlens[-1] + t.size(0))

        tokens = torch.cat(tokens)

        # Always pad to reduce memory fragmentation and maybe make the computation faster
        pad = (pad_size - tokens.size(0) % pad_size) % pad_size
        if pad != 0:
            tokens = F.pad(tokens, (0, pad), value=pad_token_id)
            cu_seqlens.append(cu_seqlens[-1] + pad)

        # thd requires the cu_seqlens to be of the origin length
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int).cuda() * cp_size

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )

    tokens = tokens.unsqueeze(0)

    batch["tokens"] = tokens
    batch["packed_seq_params"] = packed_seq_params

    # loss masks
    loss_masks = []
    for loss_mask, total_length, response_length in zip(
        batch["loss_masks"],
        batch["total_lengths"],
        batch["response_lengths"],
        strict=True,
    ):
        prompt_length = total_length - response_length
        # Align mask to token stream positions (prompt_length-1 left pad, 1 right pad)
        loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
        if allgather_cp:
            loss_masks.append(loss_mask)
            continue
        loss_mask = slice_with_cp(loss_mask, 0)
        loss_masks.append(loss_mask)

    if allgather_cp:
        # DSA: concatenate first (same as tokens), pad globally (same pad as above), then slice once.
        loss_masks = torch.cat(loss_masks, dim=0)
        if pad != 0:
            loss_masks = F.pad(loss_masks, (0, pad), value=0)
        loss_masks = loss_masks.chunk(cp_size, dim=0)[cp_rank].unsqueeze(0)
    else:
        loss_masks = torch.cat(loss_masks)
        loss_masks = F.pad(loss_masks, (0, pad), value=0).unsqueeze(0)

    assert loss_masks.shape == tokens.shape, f"loss_masks.shape: {loss_masks.shape}, tokens.shape: {tokens.shape}"
    batch["full_loss_masks"] = loss_masks

    # Process multimodal training tensors if present
    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if has_multimodal_train_inputs(multimodal_train_inputs):
        multimodal_data = {}  # key -> concatenated tensor
        multimodal_num_items = {}
        tensor_dict_list = {}
        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is None:
                continue
            for key, mm_tensor in mm_input_dict.items():
                if isinstance(mm_tensor, list):
                    mm_tensor = torch.tensor(mm_tensor, device=batch["tokens"].device)
                tensor_dict_list.setdefault(key, []).append(mm_tensor)

        for key, tensor_list in tensor_dict_list.items():
            if key in PAD_RULES:
                multimodal_data[key], multimodal_num_items[key] = pad_and_flatten(
                    tensor_list,
                    **PAD_RULES[key],
                )
            else:
                if len(tensor_list) == 1:
                    multimodal_data[key] = tensor_list[0]
                else:
                    multimodal_data[key] = torch.cat(tensor_list, dim=0)
                multimodal_num_items[key] = [t.size(0) for t in tensor_list]
        batch["multimodal_train_inputs"] = multimodal_data or None
        batch["multimodal_num_items"] = multimodal_num_items
    else:
        batch["multimodal_train_inputs"] = None
        batch["multimodal_num_items"] = {}

    return move_tensors_to_device(batch, batch["tokens"].device)


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, "float | tuple[float, float]"],
) -> dict[str, float] | None:
    """
    Gather per-rank metrics, reduce on the DP source rank, and log to W&B / TB.

    Each value in ``log_dict`` is either:
      * a ``(sum, count)`` tuple → reduced as ``Σsum / Σcount``;
      * a plain scalar → reduced as ``Σ / dp_size`` (mean across ranks).

    The gather + reduce step is delegated to
    :func:`cp_utils.gather_and_reduce_log_dict` so it can be exercised by
    CPU multi-process unit tests directly. This function adds the
    ``metric_name`` prefix and the W&B / TB logging side effects.
    """
    reduced = gather_and_reduce_log_dict(
        log_dict,
        dp_size=mpu.get_data_parallel_world_size(with_context_parallel=True),
        dp_src_rank=mpu.get_data_parallel_src_rank(with_context_parallel=True),
        dp_group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
    )
    if reduced is None:
        return None
    reduced_log_dict = {f"{metric_name}/{k}": v for k, v in reduced.items()}
    logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")
    # Calculate step once to avoid duplication
    step = compute_rollout_step(args, rollout_id)
    reduced_log_dict["rollout/step"] = step
    logging_utils.log(args, reduced_log_dict, step_key="rollout/step")
    return reduced_log_dict


class DataIterator:
    """Iterator over a rollout dict following an explicit micro-batch index schedule."""

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_indices: list[list[int]],
    ) -> None:
        """Initialize an iterator over ``rollout_data``.

        Args:
            rollout_data: Dict of per-sample fields for this DP rank.
            micro_batch_indices: List of mbs, each mbs being the local sample indices to select.
        """
        self.rollout_data = rollout_data
        self.micro_batch_indices = micro_batch_indices
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        """Return the next micro-batch for the requested keys.

        Returns a dict mapping each key to a list subset (or None if absent).
        """
        batch = {}
        indices = self.micro_batch_indices[self.offset]
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                batch[key] = [vals[i] for i in indices]
        self.offset += 1
        return batch

    def reset(self) -> "DataIterator":
        """Reset internal offset to the start and return self."""
        self.offset = 0
        return self


def move_tensors_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_tensors_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_tensors_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(move_tensors_to_device(v, device) for v in data)
    return data


def get_data_iterator(rollout_data: RolloutBatch) -> list[DataIterator]:
    """Build one ``DataIterator`` per VPP stage from the pre-computed schedule in ``rollout_data``."""
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    micro_batch_indices = rollout_data["micro_batch_indices"]
    return [DataIterator(rollout_data, micro_batch_indices) for _ in range(vpp_size)]


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    rollout_data: RolloutBatch,
) -> None:
    """
    Summarize rollout fields and log reduced metrics on PP last stage, TP rank 0.

    - Tensor-valued lists are concatenated and averaged. For token-level metrics
      like log-probs/returns/advantages/values, computes a CP-correct sample mean
      using `loss_masks` and total/response lengths.
    - Non-tensor lists are averaged elementwise.
    - Scalars are converted to Python numbers.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        cp_size = mpu.get_context_parallel_world_size()
        log_dict = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]
        max_seq_lens = rollout_data.get("max_seq_lens", None)
        padded_total_lengths = maybe_padded_total_lengths(
            total_lengths,
            args.qkv_format,
            rollout_data.get("multimodal_train_inputs") is not None or getattr(args, "uses_unsplit_forward", False),
        )
        # Same per-rollout denominators the training loss uses, so reported
        # log_probs / returns / advantages / etc. live in the same per-rollout
        # mean space (rather than per-sample) as the gradient signal.
        rollout_mask_sums = rollout_data.get("rollout_mask_sums", None)
        # For per-rollout-mean metrics: ``rollout_log_metric_contribution``
        # produces the ``(sum, count)`` tuple so gather_log_data's
        # ``Σsum / Σcount`` lands on ``sum_DP_full / num_rollouts`` — the
        # same number train_one_step reports for the same samples.
        dp_world = mpu.get_data_parallel_world_size(with_context_parallel=False)
        num_rollouts_in_rollout = sum(rollout_data["global_batch_sizes"])

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_ids",
                "rollout_mask_sums",
                "rollout_top_p_token_ids",
                "rollout_top_p_token_offsets",
                "rollout_routed_experts",
                "global_batch_sizes",
                "num_microbatches",
                "micro_batch_indices",
            ]:
                continue
            # Emit (sum, count) so gather_log_data can do a weighted average across
            # DP ranks. This stops the legacy "every rank has the same N samples"
            # assumption from biasing means once uneven-DP partitioning lands.
            if isinstance(val, (list, tuple)):
                count = len(val)
                if isinstance(val[0], torch.Tensor):
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    if key in [
                        "log_probs",
                        "ref_log_probs",
                        "rollout_log_probs",
                        "returns",
                        "advantages",
                        "values",
                        "teacher_log_probs",
                        "opd_reverse_kl",
                    ]:
                        tensor = torch.cat(val).clone().detach()
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            rollout_mask_sums,
                            qkv_format=args.qkv_format,
                            max_seq_lens=max_seq_lens,
                            padded_total_lengths=padded_total_lengths,
                        )
                        # Compute (sum, count) via the shared helper so this
                        # path and the unit tests stay in sync.
                        sum_value, count = rollout_log_metric_contribution(
                            sum_of_sample_mean(tensor).item(),
                            cp_size=cp_size,
                            num_rollouts_in_rollout=num_rollouts_in_rollout,
                            dp_size=dp_world,
                        )
                        log_dict[key] = (sum_value, count)
                        continue
                    tensor = torch.cat(val).clone().detach()
                    # val.mean() * cp_size is the per-sample mean for one rank;
                    # multiply by count to get the per-rank sum.
                    per_rank_sum = tensor.mean() * cp_size * count
                    sum_value = per_rank_sum.item()
                else:
                    sum_value = sum(val)
                log_dict[key] = (sum_value, count)
            elif isinstance(val, torch.Tensor):
                # Scalar tensor (one per rank): treat as count=1.
                log_dict[key] = (val.float().mean().item(), 1)
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict)
        if args.ci_test and reduced_log_dict is not None:
            # This is an initial actor/ref zero-KL check. R3 replays rollout
            # routing for the actor forward, while the reference forward
            # intentionally falls through to normal routing, so their
            # log-probs are not expected to match bit-for-bit in CI.
            if (
                rollout_id == 0
                and not getattr(args, "ci_disable_kl_checker", False)
                and not getattr(args, "use_rollout_routing_replay", False)
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                # TODO: figure out why there is a small numerical difference in log_probs and ref_log_probs in CI test, and whether it's expected or not.
                # assert reduced_log_dict["rollout/log_probs"] == reduced_log_dict["rollout/ref_log_probs"]
                assert abs(reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]) < 1e-8
            if "rollout/log_probs" in reduced_log_dict:
                assert -1 < reduced_log_dict["rollout/log_probs"] < 0
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 1

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data)

    if args.log_correct_samples:
        if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
            cp_size = mpu.get_context_parallel_world_size()
            log_dict = {}
            response_lengths = rollout_data["response_lengths"]
            loss_masks = rollout_data["loss_masks"]
            total_lengths = rollout_data["total_lengths"]

            def quantile(total_value, n_quantiles, data) -> dict:
                import math

                assert n_quantiles > 1, f"n_quantiles({n_quantiles}) must be greater than 1."

                quantiles = [((i + 1) / n_quantiles) for i in range(n_quantiles)]
                cut_points = [total_value * q for q in quantiles]
                cut_points[-1] = total_value

                count = [0] * n_quantiles
                for d in data:
                    for i, point in enumerate(cut_points):
                        if d <= point:
                            count[i] += 1
                            break

                total = sum(count) + 1e-9
                percentile = [c / total for c in count]

                percentile = {f"p{min(math.ceil(q*100),100)}": p for q, p in zip(quantiles, percentile, strict=True)}
                return percentile

            raw_rewards = rollout_data["raw_reward"]
            # Additional metrics for correct cases are calculated separately below.
            padded_total_lengths = maybe_padded_total_lengths(
                total_lengths,
                args.qkv_format,
                rollout_data.get("multimodal_train_inputs") is not None
                or getattr(args, "uses_unsplit_forward", False),
            )
            correct_response_lengths = []
            correct_total_lengths = []
            correct_loss_masks = []
            correct_padded_total_lengths = [] if padded_total_lengths is not None else None
            correct_entropy = []
            for i, raw_reward in enumerate(raw_rewards):
                if raw_reward == 1:
                    correct_response_lengths.append(response_lengths[i])
                    correct_total_lengths.append(total_lengths[i])
                    correct_loss_masks.append(loss_masks[i])
                    if correct_padded_total_lengths is not None:
                        correct_padded_total_lengths.append(padded_total_lengths[i])
                    correct_entropy.append(-rollout_data["log_probs"][i])
            num_correct_responses = len(correct_total_lengths)
            rollout_data["correct_response_lengths"] = correct_response_lengths
            correct_response_length_percentile = quantile(
                args.rollout_max_response_len, 4, rollout_data["correct_response_lengths"]
            )
            for p, val in correct_response_length_percentile.items():
                rollout_data[f"correct_length/{p}"] = [val] * num_correct_responses
            if len(correct_entropy) > 0:
                # NOTE: per-sample-mean over the correct subset, not per-rollout.
                # A rollout's siblings may not all be correct, and slicing
                # ``rollout_mask_sums`` here would leave a denom that still
                # includes incorrect siblings — meaningless for a "correct-only"
                # entropy report. Per-sample-mean over the filtered subset is
                # the cleanest semantic.
                sum_of_sample_mean = get_sum_of_sample_mean(
                    correct_total_lengths,
                    correct_response_lengths,
                    correct_loss_masks,
                    sample_denoms=None,
                    qkv_format=args.qkv_format,
                    padded_total_lengths=correct_padded_total_lengths,
                )
                correct_entropy = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
                rollout_data["correct_entropy"] = [correct_entropy.item()] * num_correct_responses
            else:
                rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Log multi-turn auxiliary metrics such as raw/observed response lengths and rounds.

    Operates only on PP last stage and TP rank 0. Uses GPU tensors when available
    to compute statistics without host transfers.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key == "loss_masks":
                if val:  # Check if val is not empty
                    device = val[0].device  # Get device from first tensor

                    # Vectorized length calculation using torch
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )

                    # Vectorized sum calculation using torch - stay on GPU
                    wo_obs_response_lengths = torch.tensor(
                        [v.sum().item() for v in val], dtype=torch.float32, device=device
                    )
                    log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
                    log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
                    log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
            if key == "round_number":
                # Use numpy for vectorized round number statistics
                round_number_array = np.array(val)
                log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
                log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
                log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
        gather_log_data("multi_turn", args, rollout_id, log_dict)


def log_passrate(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Compute pass@k metrics from `raw_reward` groups and log the results.

    `raw_reward` is reshaped to `[group_number, group_size]`, then pass@k is
    estimated per problem and averaged.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key != "raw_reward":
                continue

            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )

        gather_log_data("passrate", args, rollout_id, log_dict)


def log_perf_data(rollout_id: int, args: Namespace, extra_metrics: dict | None = None) -> None:
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.is_pipeline_last_stage()
            and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        ),
        compute_total_fwd_flops=lambda seq_lens: calculate_fwd_flops(seqlens=seq_lens, args=args)
        / dist.get_world_size()
        / 1e12,
        extra_metrics=extra_metrics,
    )


def tensors_to_cpu(tensor_list):
    """Move a list of GPU tensors to CPU for Ray object store transfer.

    Args:
        tensor_list: List of GPU tensors, or None.

    Returns:
        List of CPU tensors (detached), or None if input is None.
    """
    if tensor_list is None:
        return None
    return [t.detach().cpu() for t in tensor_list]


def tensors_to_gpu(tensor_list, device=None):
    """Move a list of CPU tensors back to GPU.

    Args:
        tensor_list: List of CPU tensors, or None.
        device: Target CUDA device. If None, uses current device.

    Returns:
        List of GPU tensors, or None if input is None.
    """
    if tensor_list is None:
        return None
    if device is None:
        device = torch.cuda.current_device()
    return [t.to(device=device, dtype=torch.float32) for t in tensor_list]
