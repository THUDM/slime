import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

import wandb
from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.timer import Timer

from .cp_utils import get_sum_of_sample_mean, slice_with_cp


def get_batch(data_iterator, keys):
    """Generate a batch."""

    assert "tokens" in keys
    batch = data_iterator.get_next(keys)

    packed_seq_params = None
    tokens = batch["tokens"]
    # use 0 as the pad token id should be fine?
    pad_token_id = 0

    # for cp, we need all tokens to calculate logprob
    batch["unconcat_tokens"] = tokens

    cp_size = mpu.get_context_parallel_world_size()
    tokens = [slice_with_cp(t, pad_token_id) for t in tokens]

    cu_seqlens = [0]
    for t in tokens:
        cu_seqlens.append(cu_seqlens[-1] + t.size(0))

    tokens = torch.cat(tokens)

    # Always pad to 128 to reduce memory fragmentation and maybe make the computation faster
    # TODO: make this configurable?
    pad = (128 - tokens.size(0) % 128) % 128
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
    return batch


def gather_log_data(metic_name, args, rollout_id, log_dict):
    if mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
        gathered_log_dict = [None] * mpu.get_data_parallel_world_size(with_context_parallel=True)
        # Not sure if this will be a performance bottleneck.
        dist.gather_object(
            log_dict,
            gathered_log_dict,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        reduced_log_dict = {
            f"{metic_name}/{key}": sum([d[key] for d in gathered_log_dict]) / dp_size for key in log_dict
        }
        print(f"{metic_name} {rollout_id}: {reduced_log_dict}")
        if args.use_wandb:
            reduced_log_dict["rollout/step"] = (
                rollout_id
                if not args.wandb_always_use_train_step
                else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
            )
            wandb.log(reduced_log_dict)
        return reduced_log_dict
    else:
        dist.gather_object(
            log_dict,
            None,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )
        return None


def log_rollout_data(rollout_id, args, rollout_data):
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        cp_size = mpu.get_context_parallel_world_size()
        log_dict = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]

        for key, val in rollout_data.items():
            if key == "tokens" or key == "loss_masks" or key == "sample_indices":
                continue
            # Upload per sample mean for each rollout value
            # There are the following assumptions:
            # - Each dp rank has the same number of samples
            if isinstance(val, list):
                if isinstance(val[0], torch.Tensor):
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    val = torch.cat(val).clone().detach()
                    if key in ["log_probs", "ref_log_probs", "rollout_log_probs", "returns", "advantages"]:
                        sum_of_sample_mean = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks)
                        val = cp_size * sum_of_sample_mean(val) / len(loss_masks)
                    else:
                        val = val.mean() * cp_size
                else:
                    val = sum(val) / len(val)
            elif isinstance(val, torch.Tensor):
                val = val.float().mean()
            else:
                raise ValueError(f"Unsupported type: {type(val)}")
            log_dict[key] = val.item() if isinstance(val, torch.Tensor) else val

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict)
        if args.ci_test and reduced_log_dict is not None:
            if (
                rollout_id == 0
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                assert reduced_log_dict["rollout/log_probs"] == reduced_log_dict["rollout/ref_log_probs"]
            if "rollout/log_probs" in reduced_log_dict:
                assert -0.5 < reduced_log_dict["rollout/log_probs"] < 0
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 0.5

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data)


def log_multi_turn_data(rollout_id, args, rollout_data):
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


def log_passrate(rollout_id, args, rollout_data):
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key != "raw_reward":
                continue

            group_size = args.n_samples_per_prompt
            group_number = args.rollout_batch_size
            assert len(val) == group_number * group_size
            pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

            val = np.array(val).reshape(group_number, group_size)

            def estimate_pass_at_k(num_samples, num_correct, k):
                """
                Estimates pass@k of each problem and returns them in an array.
                """

                def estimator(n, c, k):
                    """
                    Calculates 1 - comb(n - c, k) / comb(n, k).
                    """
                    if n - c < k:
                        return 1.0
                    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

                return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct)])

            for k in pass_rate_name_list:
                num_correct = np.sum(val == 1, axis=1)
                num_samples = np.full(group_number, group_size)

                pass_k_estimates = estimate_pass_at_k(num_samples, num_correct, k)

                pass_k = np.mean(pass_k_estimates)
                log_dict[f"pass@{k}"] = pass_k

        gather_log_data("passrate", args, rollout_id, log_dict)


def log_perf_data(rollout_id, args):
    timer_instance = Timer()
    if (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.is_pipeline_last_stage()
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    ):
        log_dict = {f"perf/{key}_time": val for key, val in timer_instance.log_dict().items()}

        if "perf/actor_train_time" in log_dict:
            world_size = dist.get_world_size()
            total_fwd_flops = calculate_fwd_flops(seqlens=timer_instance.seq_lens, args=args) / world_size / 1e12

            if "perf/log_probs_time" in log_dict:
                log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

            if "perf/ref_log_probs_time" in log_dict:
                log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

            if log_dict["perf/actor_train_time"] > 0:
                log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]

        if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
            total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
            if total_time > 0:
                log_dict["perf/total_train_time"] = total_time
                log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

        print(f"perf {rollout_id}: {log_dict}")
        if args.use_wandb:
            log_dict["rollout/step"] = (
                rollout_id
                if not args.wandb_always_use_train_step
                else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
            )
            wandb.log(log_dict)
    timer_instance.reset()
