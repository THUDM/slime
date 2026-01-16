import logging
from argparse import Namespace
from math import isclose

import numpy as np
import torch
import torch.distributed as dist

from slime.utils import train_metric_utils
from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step
from slime.utils.types import RolloutBatch

from ...utils import logging_utils
from .cp_utils import get_sum_of_sample_mean
from .data import DataIterator
from .parallel import ParallelState

logger = logging.getLogger(__name__)


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, float],
    parallel_state: ParallelState,
) -> dict[str, float] | None:
    """
    Gather per-rank metrics, reduce by mean on the DP source rank, and log.

    Expects `log_dict` to contain plain scalars. The DP source rank prints and
    optionally logs to WandB/TensorBoard with a step derived from `rollout_id` and
    batch sizes. Returns the reduced dict on the DP source rank; returns None on others.
    """

    if parallel_state.dp_cp_rank == 0:
        dp_size = parallel_state.dp_cp_size

        gathered_log_dict = [None] * dp_size
        # Not sure if this will be a performance bottleneck.
        dist.gather_object(
            log_dict,
            gathered_log_dict,
            dst=parallel_state.dp_src_rank,
            group=parallel_state.dp_cp_group_gloo,
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
            dst=parallel_state.dp_src_rank,
            group=parallel_state.dp_cp_group_gloo,
        )
        return None


def aggregate_forward_results(
    forward_data_store: list[dict[str, list]],
    data_iterator: DataIterator,
    args: Namespace,
    store_prefix: str = "",
) -> dict[str, list]:
    rollout_data = {}
    if not forward_data_store:
        return rollout_data

    keys = forward_data_store[0].keys()
    for key in keys:
        values = []
        for batch_result in forward_data_store:
            assert isinstance(batch_result[key], list), f"Expected list for key {key}, got {type(batch_result[key])}"
            values += batch_result[key]

        # Handle dynamic batch size: restore original order
        if args.use_dynamic_batch_size and hasattr(data_iterator, "micro_batch_indices"):
            origin_values = [None] * len(values)
            origin_indices = sum(data_iterator.micro_batch_indices, [])
            for value, origin_index in zip(values, origin_indices, strict=False):
                origin_values[origin_index] = value
            values = origin_values

        rollout_data[key] = values

    return rollout_data


def log_rollout_data(
    rollout_id: int, args: Namespace, rollout_data: RolloutBatch, parallel_state: ParallelState
) -> None:
    """
    Summarize rollout fields and log reduced metrics on PP last stage, TP rank 0.

    - Tensor-valued lists are concatenated and averaged. For token-level metrics
      like log-probs/returns/advantages/values, computes a CP-correct sample mean
      using `loss_masks` and total/response lengths.
    - Non-tensor lists are averaged elementwise.
    - Scalars are converted to Python numbers.
    """
    if parallel_state.tp_rank == 0 and parallel_state.is_pp_last_stage:
        cp_size = parallel_state.cp_size
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
            ]:
                continue
            # Upload per sample mean for each rollout value
            # There are the following assumptions:
            # - Each dp rank has the same number of samples
            if isinstance(val, (list, tuple)):
                if isinstance(val[0], torch.Tensor):
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    val = torch.cat(val).clone().detach()
                    if key in ["log_probs", "ref_log_probs", "rollout_log_probs", "returns", "advantages", "values"]:
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            parallel_state,
                            qkv_format=args.qkv_format,
                            max_seq_lens=max_seq_lens,
                        )
                        val = cp_size * sum_of_sample_mean(val) / len(loss_masks)
                    else:
                        val = val.mean() * cp_size
                else:
                    val = sum(val) / len(val)
            elif isinstance(val, torch.Tensor):
                val = val.float().mean()
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")
            log_dict[key] = val.item() if isinstance(val, torch.Tensor) else val

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict, parallel_state)
        if args.ci_test and reduced_log_dict is not None:
            if (
                rollout_id == 0
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                assert reduced_log_dict["rollout/log_probs"] == reduced_log_dict["rollout/ref_log_probs"]
            if "rollout/log_probs" in reduced_log_dict and "rollout/rollout_log_probs" in reduced_log_dict:
                assert isclose(
                    reduced_log_dict["rollout/log_probs"], reduced_log_dict["rollout/rollout_log_probs"], abs_tol=0.03
                )
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 0.7

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data, parallel_state)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data)

    if args.log_correct_samples:
        if parallel_state.tp_rank == 0 and parallel_state.is_pp_last_stage:
            cp_size = parallel_state.cp_size
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
                    correct_total_lengths, correct_response_lengths, correct_loss_masks, parallel_state
                )
                correct_entropy = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
                rollout_data["correct_entropy"] = [correct_entropy.item()] * num_correct_responses
            else:
                rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(
    rollout_id: int, args: Namespace, rollout_data: RolloutBatch, parallel_state: ParallelState
) -> None:
    """
    Log multi-turn auxiliary metrics such as raw/observed response lengths and rounds.

    Operates only on PP last stage and TP rank 0. Uses GPU tensors when available
    to compute statistics without host transfers.
    """
    if parallel_state.tp_rank == 0 and parallel_state.is_pp_last_stage:
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
        gather_log_data("multi_turn", args, rollout_id, log_dict, parallel_state)


def log_passrate(rollout_id: int, args: Namespace, rollout_data: RolloutBatch, parallel_state: ParallelState) -> None:
    """
    Compute pass@k metrics from `raw_reward` groups and log the results.

    `raw_reward` is reshaped to `[group_number, group_size]`, then pass@k is
    estimated per problem and averaged.
    """
    if parallel_state.tp_rank == 0 and parallel_state.is_pp_last_stage:
        log_dict = {}
        for key, val in rollout_data.items():
            if key != "raw_reward":
                continue

            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )

        gather_log_data("passrate", args, rollout_id, log_dict, parallel_state)


def log_perf_data(rollout_id: int, args: Namespace, parallel_state: ParallelState) -> None:
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            parallel_state.tp_rank == 0 and parallel_state.is_pp_last_stage and parallel_state.dp_cp_rank == 0
        ),
        compute_total_fwd_flops=lambda seq_lens: calculate_fwd_flops(seqlens=seq_lens, args=args)
        / dist.get_world_size()
        / 1e12,
    )


def aggregate_train_losses(
    losses_reduced: list[dict[str, list[str] | torch.Tensor]],
    parallel_state: ParallelState,
) -> dict[str, float]:
    """Aggregate loss metrics across micro-batches.

    Sums loss values across all micro-batches, performs all-reduce across
    the data-parallel group, and computes per-sample/token averages.

    Args:
        losses_reduced: List of log_dict from each micro-batch.
            Each log_dict has format: {"keys": list[str], "values": torch.Tensor}
        parallel_state: Parallel state containing dp_group and cp_size.

    Returns:
        Dictionary mapping metric names to averaged values.
    """
    if not losses_reduced:
        return {}

    keys = losses_reduced[0]["keys"]

    values = None
    for log_dict in losses_reduced:
        if values is None:
            values = log_dict["values"].clone()
        else:
            values += log_dict["values"]

    assert len(keys) + 1 == values.numel(), f"Expected {len(keys) + 1} values, got {values.numel()}"

    dist.all_reduce(values, op=dist.ReduceOp.SUM, group=parallel_state.dp_cp_group)

    loss_reduced = {}
    values = values.tolist()
    num_samples_or_tokens = values[0]

    for key, value in zip(keys, values[1:], strict=False):
        loss_reduced[key] = value * parallel_state.cp_size / num_samples_or_tokens

    return loss_reduced


def log_train_step(
    args: Namespace,
    loss_dict: dict[str, float],
    grad_norm: float,
    rollout_id: int,
    step_id: int,
    num_steps_per_rollout: int,
    role: str = "actor",
    extra_metrics: dict[str, float] | None = None,
    should_log: bool | None = None,
) -> dict[str, float]:
    """Log training metrics for one step.

    Formats loss metrics, gradient norm, and extra metrics (e.g., learning rates, MTP loss) for tracking.

    Args:
        args: Configuration.
        loss_dict: Dictionary of loss metrics from aggregate_train_losses.
        grad_norm: Gradient norm after clipping.
        rollout_id: Rollout ID.
        step_id: Step ID within the rollout.
        num_steps_per_rollout: Total number of steps per rollout.
        role: Role name (e.g., "actor", "critic").
        extra_metrics: Optional extra metrics to log (e.g., learning rates, MTP loss).
        should_log: Optional override for logging condition. If None, uses rank == 0.

    Returns:
        The formatted log_dict (for CI tests or other uses).
    """
    accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
    role_tag = "" if role == "actor" else f"{role}-"

    log_dict_out = {
        f"train/{role_tag}{key}": val.mean().item() if isinstance(val, torch.Tensor) else val
        for key, val in loss_dict.items()
    }
    log_dict_out[f"train/{role_tag}grad_norm"] = float(grad_norm)

    if extra_metrics:
        for key, val in extra_metrics.items():
            log_dict_out[f"train/{role_tag}{key}"] = val

    log_dict_out["train/step"] = accumulated_step_id

    if should_log is None:
        should_log = dist.get_rank() == 0

    if should_log:
        logging_utils.log(args, log_dict_out, step_key="train/step")
        logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict_out}")

    return log_dict_out
