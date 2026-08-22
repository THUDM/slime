import logging
from argparse import Namespace
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist

from slime.observability import logging_utils
from slime.observability.metric_utils import compute_pass_rate, compute_rollout_step
from slime.observability.timer import Timer
from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


def reduce_train_step_metrics(
    losses_reduced: list[dict],
    *,
    calculate_per_token_loss: bool,
    step_global_batch_size: int,
    cp_size: int,
    dp_with_cp_group,
) -> dict[str, float]:
    """Aggregate per-mb log dicts into the dict ``train_one_step`` reports.

    Pipeline (1:1 with what the train loop used to do inline):
      1. Sum each metric's per-mb ``values`` tensor locally on this rank.
      2. All-reduce across the DP*CP group (``dp_with_cp_group``).
      3. Apply the per-mode divisor / cp_factor:
         - per-token-loss: divisor = ``values[0]`` = all-reduced ``num_tokens``,
           CP-inflated by ``cp_size`` because every CP rank computes the same
           num_tokens off the FULL (not chunked) masks; the
           ``cp_factor = cp_size`` multiplier cancels that inflation, leaving
           the genuine per-token average.
         - per-rollout-mean: divisor = constant ``step_global_batch_size`` from
           the rollout side, never all-reduced, so no CP inflation to cancel
           and ``cp_factor = 1``.

    Tests pass a mock ``dp_with_cp_group`` and monkeypatch ``dist.all_reduce``
    to a no-op, then pre-aggregate virtual ranks themselves — this exercises
    the same call shape as production while staying single-process.
    """
    keys = losses_reduced[0]["keys"]
    values = None
    for item in losses_reduced:
        values = item["values"] if values is None else values + item["values"]
    assert len(keys) + 1 == values.numel()
    dist.all_reduce(values, group=dp_with_cp_group)
    values = values.tolist()

    if calculate_per_token_loss:
        num_samples_or_tokens = values[0]
        cp_factor = cp_size
    else:
        num_samples_or_tokens = step_global_batch_size
        cp_factor = 1
    return {key: value * cp_factor / num_samples_or_tokens for key, value in zip(keys, values[1:], strict=False)}


def rollout_log_metric_contribution(
    per_rank_reducer_sum: float,
    *,
    cp_size: int,
    num_rollouts_in_rollout: int,
    dp_size: int,
) -> tuple[float, float]:
    """``(sum, count)`` tuple for a per-rollout-mean metric.

    Sum across DP*CP ranks of ``count`` lands on ``num_rollouts_in_rollout``
    (``dp_size`` here is the no-CP DP width; the gather covers ``dp_size *
    cp_size`` ranks, and each rank emits the same ``count``, so the totals
    cancel out the ``cp_size`` in the sum). Result: ``Σsum / Σcount =
    sum_DP_full / num_rollouts`` — the same number ``train_one_step`` reports
    for the same samples (when ``num_steps_per_rollout == 1``).

    Pair with :func:`gather_and_reduce_log_dict` to do the full end-to-end
    in tests.
    """
    sum_value = cp_size * per_rank_reducer_sum
    count = num_rollouts_in_rollout / dp_size
    return sum_value, count


def gather_and_reduce_log_dict(
    log_dict: dict,
    *,
    dp_size: int,
    dp_src_rank: int,
    dp_group,
) -> dict | None:
    """Gather per-rank log dicts and reduce each metric on ``dp_src_rank``.

    ``(sum, count)`` tuples reduce to ``Σsum / Σcount``; plain values reduce
    to a mean across ranks. The helper stays free of reporting side effects so
    CPU multi-process tests can exercise it with real ``torch.distributed``.
    """
    if dist.get_rank() == dp_src_rank:
        gathered = [None] * dp_size
        dist.gather_object(log_dict, gathered, dst=dp_src_rank, group=dp_group)
        reduced: dict = {}
        for key in log_dict:
            values = [item[key] for item in gathered]
            first = values[0]
            if isinstance(first, tuple) and len(first) == 2:
                total_sum = sum(value[0] for value in values)
                total_count = sum(value[1] for value in values)
                reduced[key] = total_sum / total_count if total_count else 0.0
            else:
                reduced[key] = sum(values) / dp_size
        return reduced
    dist.gather_object(log_dict, None, dst=dp_src_rank, group=dp_group)
    return None


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, "float | tuple[float, float]"],
) -> dict[str, float] | None:
    """Gather per-rank metrics and report them through the configured trackers."""
    from megatron.core import mpu

    reduced = gather_and_reduce_log_dict(
        log_dict,
        dp_size=mpu.get_data_parallel_world_size(with_context_parallel=True),
        dp_src_rank=mpu.get_data_parallel_src_rank(with_context_parallel=True),
        dp_group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
    )
    if reduced is None:
        return None
    reduced_log_dict = {f"{metric_name}/{key}": value for key, value in reduced.items()}
    logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")
    step = compute_rollout_step(args, rollout_id)
    reduced_log_dict["rollout/step"] = step
    logging_utils.log(args, reduced_log_dict, step_key="rollout/step")
    return reduced_log_dict


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    rollout_data: RolloutBatch,
) -> None:
    """Summarize and report Megatron-side rollout fields."""
    from megatron.core import mpu

    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        cp_size = mpu.get_context_parallel_world_size()
        log_dict = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]
        rollout_mask_sums = rollout_data.get("rollout_mask_sums", None)
        dp_world = mpu.get_data_parallel_world_size(with_context_parallel=False)
        num_rollouts_in_rollout = sum(rollout_data["global_batch_sizes"])

        ignored_keys = {
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
            "source_names",
            "local_raw_reward",
        }
        per_rollout_mean_keys = {
            "log_probs",
            "ref_log_probs",
            "rollout_log_probs",
            "returns",
            "advantages",
            "values",
            "teacher_log_probs",
            "opd_reverse_kl",
        }

        for key, value in rollout_data.items():
            if key in ignored_keys:
                continue
            if isinstance(value, (list, tuple)):
                count = len(value)
                if isinstance(value[0], torch.Tensor):
                    tensor = torch.cat(value).clone().detach()
                    if key in per_rollout_mean_keys:
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            rollout_mask_sums,
                        )
                        sum_value, count = rollout_log_metric_contribution(
                            sum_of_sample_mean(tensor).item(),
                            cp_size=cp_size,
                            num_rollouts_in_rollout=num_rollouts_in_rollout,
                            dp_size=dp_world,
                        )
                        log_dict[key] = (sum_value, count)
                        continue
                    per_rank_sum = tensor.mean() * cp_size * count
                    sum_value = per_rank_sum.item()
                else:
                    sum_value = sum(value)
                log_dict[key] = (sum_value, count)
            elif isinstance(value, torch.Tensor):
                log_dict[key] = (value.float().mean().item(), 1)
            else:
                raise ValueError(f"Unsupported type: {type(value)} for key: {key}")

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict)
        if args.ci_test and reduced_log_dict is not None:
            if (
                rollout_id == 0
                and not getattr(args, "ci_disable_kl_checker", False)
                and not getattr(args, "use_rollout_routing_replay", False)
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                assert abs(reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]) < 1e-8
            if "rollout/log_probs" in reduced_log_dict:
                assert -1 < reduced_log_dict["rollout/log_probs"] < 0
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 1

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data)

    if args.log_correct_samples and mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]

        def quantile(total_value, n_quantiles, data) -> dict:
            import math

            assert n_quantiles > 1, f"n_quantiles({n_quantiles}) must be greater than 1."
            quantiles = [(i + 1) / n_quantiles for i in range(n_quantiles)]
            cut_points = [total_value * quantile for quantile in quantiles]
            cut_points[-1] = total_value

            count = [0] * n_quantiles
            for value in data:
                for i, point in enumerate(cut_points):
                    if value <= point:
                        count[i] += 1
                        break

            total = sum(count) + 1e-9
            percentile = [value / total for value in count]
            return {
                f"p{min(math.ceil(quantile * 100), 100)}": value
                for quantile, value in zip(quantiles, percentile, strict=True)
            }

        raw_rewards = rollout_data["local_raw_reward"]
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
            args.rollout_max_response_len,
            4,
            rollout_data["correct_response_lengths"],
        )
        for percentile, value in correct_response_length_percentile.items():
            rollout_data[f"correct_length/{percentile}"] = [value] * num_correct_responses
        if correct_entropy:
            sum_of_sample_mean = get_sum_of_sample_mean(
                correct_total_lengths,
                correct_response_lengths,
                correct_loss_masks,
                sample_denoms=None,
            )
            correct_entropy_value = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
            rollout_data["correct_entropy"] = [correct_entropy_value.item()] * num_correct_responses
        else:
            rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """Report multi-turn response-length and round-count metrics."""
    from megatron.core import mpu

    if mpu.get_tensor_model_parallel_rank() != 0 or not mpu.is_pipeline_last_stage():
        return

    log_dict = {}
    for key, value in rollout_data.items():
        if key == "loss_masks" and value:
            device = value[0].device
            raw_response_lengths = torch.tensor(
                [item.shape[0] for item in value],
                dtype=torch.float32,
                device=device,
            )
            log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
            log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
            log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
            log_dict["raw_response_length/response_length_clip_ratio"] = (
                (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
            )

            wo_obs_response_lengths = torch.tensor(
                [item.sum().item() for item in value],
                dtype=torch.float32,
                device=device,
            )
            log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
            log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
            log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
        if key == "round_number":
            round_number_array = np.array(value)
            log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
            log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
            log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
    gather_log_data("multi_turn", args, rollout_id, log_dict)


def log_passrate(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute and report pass@k metrics from grouped ``raw_reward`` values."""
    from megatron.core import mpu

    if mpu.get_tensor_model_parallel_rank() != 0 or not mpu.is_pipeline_last_stage():
        return

    log_dict = {}
    for key, value in rollout_data.items():
        if key == "raw_reward":
            log_dict |= compute_pass_rate(
                flat_rewards=value,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )
    gather_log_data("passrate", args, rollout_id, log_dict)


def log_perf_data(
    rollout_id: int,
    args: Namespace,
    extra_metrics: dict | None = None,
) -> None:
    from megatron.core import mpu

    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.is_pipeline_last_stage()
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    ):
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}
    if extra_metrics:
        log_dict.update(extra_metrics)

    if "perf/actor_train_time" in log_dict:
        total_fwd_flops = (
            calculate_fwd_flops(seqlens=timer_instance.seq_lens, args=args) / dist.get_world_size() / 1e12
        )

        if "perf/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    logger.info(f"perf {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")
