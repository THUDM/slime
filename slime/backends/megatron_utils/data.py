import logging
from argparse import Namespace
from collections.abc import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

from slime.utils import train_metric_utils
from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step
from slime.utils.types import RolloutBatch

from ...utils import logging_utils
from .cp_utils import get_sum_of_sample_mean, slice_with_cp

logger = logging.getLogger(__name__)


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
    qkv_format: str = "thd",
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

    if "dynamic_global_batch_size" in data_iterator.rollout_data:
        batch["dynamic_global_batch_size"] = data_iterator.rollout_data["dynamic_global_batch_size"]

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
        tokens = [slice_with_cp(t, pad_token_id, qkv_format, max_seqlen) for t in tokens]
        tokens = torch.stack(tokens)
        packed_seq_params = None

    elif qkv_format == "thd":
        if allgather_cp:
            # DSA mode: concatenate all sequences first, then slice once with CP.
            # We also pad the *global* concatenated stream to make per-rank chunks equal.
            cu_seqlens_list: list[int] = [0]
            for t in tokens:
                cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))

            tokens = torch.cat(tokens, dim=0)

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
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )

        tokens = tokens.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported qkv_format: {qkv_format}")

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
        loss_mask = slice_with_cp(loss_mask, 0, qkv_format, max_seqlen)
        loss_masks.append(loss_mask)

    if qkv_format == "bshd":
        loss_masks = torch.stack(loss_masks)
    elif qkv_format == "thd" and allgather_cp:
        # DSA: concatenate first (same as tokens), pad globally (same pad as above), then slice once.
        loss_masks = torch.cat(loss_masks, dim=0)
        if pad != 0:
            loss_masks = F.pad(loss_masks, (0, pad), value=0)
        loss_masks = loss_masks.chunk(cp_size, dim=0)[cp_rank].unsqueeze(0)
    elif qkv_format == "thd":
        loss_masks = torch.cat(loss_masks)
        loss_masks = F.pad(loss_masks, (0, pad), value=0).unsqueeze(0)

    assert loss_masks.shape == tokens.shape, f"loss_masks.shape: {loss_masks.shape}, tokens.shape: {tokens.shape}"
    batch["full_loss_masks"] = loss_masks

    # Process multimodal training tensors if present
    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None:
        multimodal_data = {}  # key -> concatenated tensor
        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is not None:
                for key, mm_tensor in mm_input_dict.items():
                    if key not in multimodal_data:
                        multimodal_data[key] = mm_tensor
                    else:
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
        batch["multimodal_train_inputs"] = multimodal_data

    return batch


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, float],
) -> dict[str, float] | None:
    """
    Gather per-rank metrics, reduce by mean on the DP source rank, and log.

    Expects `log_dict` to contain plain scalars. The DP source rank prints and
    optionally logs to WandB/TensorBoard with a step derived from `rollout_id` and
    batch sizes. Returns the reduced dict on the DP source rank; returns None on others.
    """

    if mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)

        gathered_log_dict = [None] * dp_size
        # Not sure if this will be a performance bottleneck.
        dist.gather_object(
            log_dict,
            gathered_log_dict,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )

        reduced_log_dict = {
            f"{metric_name}/{key}": sum([d[key] for d in gathered_log_dict]) / dp_size for key in log_dict
        }
        logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")

        # Calculate step once to avoid duplication
        step = compute_rollout_step(args, rollout_id)
        reduced_log_dict["rollout/step"] = step
        logging_utils.log(args, reduced_log_dict, step_key="rollout/step")

        return reduced_log_dict
    else:
        dist.gather_object(
            log_dict,
            None,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )
        return None


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

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_routed_experts",
                "max_seq_lens",
                "dynamic_global_batch_size",
                "num_microbatches",
                "micro_batch_indices",
            ]:
                continue
            # Upload per sample mean for each rollout value
            # There are the following assumptions:
            # - Each dp rank has the same number of samples
            if isinstance(val, (list, tuple)):
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
                        val = torch.cat(val).clone().detach()
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            qkv_format=args.qkv_format,
                            max_seq_lens=max_seq_lens,
                        )
                        val = cp_size * sum_of_sample_mean(val) / len(loss_masks)
                    else:
                        val = torch.cat(val).clone().detach()
                        val = val.mean() * cp_size
                else:
                    val = sum(val) / len(val)
            elif isinstance(val, torch.Tensor):
                val = val.float().mean()
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")
            log_dict[key] = val.item() if isinstance(val, torch.Tensor) else val

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict)
        if args.ci_test and reduced_log_dict is not None:
            if (
                rollout_id == 0
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                # TODO: figure out why there is a small numerical difference in log_probs and ref_log_probs in CI test, and whether it's expected or not.
                # assert reduced_log_dict["rollout/log_probs"] == reduced_log_dict["rollout/ref_log_probs"]
                assert abs(reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]) < 1e-8
            if "rollout/log_probs" in reduced_log_dict:
                assert -0.5 < reduced_log_dict["rollout/log_probs"] < 0
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 0.5

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
            correct_response_lengths = []
            correct_total_lengths = []
            correct_loss_masks = []
            correct_entropy = []
            for i, raw_reward in enumerate(raw_rewards):
                if raw_reward == 1:
                    correct_response_lengths.append(response_lengths[i])
                    correct_total_lengths.append(total_lengths[i])
                    correct_loss_masks.append(loss_masks[i])
                    correct_entropy.append(-rollout_data["log_probs"][i])
            num_correct_responses = len(correct_total_lengths)
            rollout_data["correct_response_lengths"] = correct_response_lengths
            correct_response_length_percentile = quantile(
                args.rollout_max_response_len, 4, rollout_data["correct_response_lengths"]
            )
            for p, val in correct_response_length_percentile.items():
                rollout_data[f"correct_length/{p}"] = [val] * num_correct_responses
            if len(correct_entropy) > 0:
                sum_of_sample_mean = get_sum_of_sample_mean(
                    correct_total_lengths, correct_response_lengths, correct_loss_masks
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


def log_perf_data(rollout_id: int, args: Namespace) -> None:
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
