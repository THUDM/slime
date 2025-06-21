import math
from typing import Any, Optional

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config

import wandb
from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.memory_utils import clear_memory
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.timer import Timer

LOCAL_STORAGE = {}
LOCAL_METADATA = {}


def set_local_storage(key: str, value: Any):
    LOCAL_STORAGE[key] = value


def get_local_storage(key: Optional[str] = None):
    if key is None:
        return LOCAL_STORAGE
    return LOCAL_STORAGE.get(key, None)


def clear_local_storage():
    LOCAL_STORAGE.clear()
    clear_memory()


def set_metadata(key: str, value: Any):
    LOCAL_METADATA[key] = value


def get_metadata(key: Optional[str] = None):
    return LOCAL_METADATA.get(key, None)


def get_batch(data_iterator, keys):
    """Generate a batch."""

    assert "tokens" in keys
    batch = data_iterator.get_next(keys)

    packed_seq_params = None
    tokens = batch["tokens"]
    pad_token_id = get_metadata("padding_token_id")

    # for cp, we need all tokens to calculate logprob
    batch["unconcat_tokens"] = tokens

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    if cp_size > 1:

        def pad_and_split_tokens(tokens: list[torch.Tensor]):
            # pad
            chunk_size = (len(tokens) + 2 * cp_size - 1) // (2 * cp_size)
            pad = 2 * cp_size * chunk_size - len(tokens)
            tokens = F.pad(tokens, (0, pad), value=pad_token_id)
            # get 2 chunk for thd cp
            start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
            start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
            return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])

        tokens = [pad_and_split_tokens(t) for t in tokens]

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


class DataIterator:
    def __init__(
        self,
        micro_batch_size: Optional[int] = None,
        micro_batch_indices: Optional[list[list[int]]] = None,
    ):
        self.micro_batch_size = micro_batch_size
        self.micro_batch_indices = micro_batch_indices
        assert micro_batch_size is None or micro_batch_indices is None
        self.offset = 0

    def get_next(self, keys):
        batch = {}
        for key in keys:
            vals = get_local_storage(key)
            if vals is None:
                batch[key] = None
            else:
                if self.micro_batch_indices is not None:
                    indices = self.micro_batch_indices[self.offset]
                    batch[key] = [vals[i] for i in indices]
                else:
                    assert self.offset + self.micro_batch_size <= len(
                        vals
                    ), f"offset: {self.offset}, micro_batch_size: {self.micro_batch_size}, len(vals): {len(vals)}"
                    batch[key] = vals[self.offset : self.offset + self.micro_batch_size]

        if self.micro_batch_indices is not None:
            self.offset += 1
        else:
            self.offset += self.micro_batch_size
        return batch

    def reset(self):
        self.offset = 0
        return self


def ceildiv(a, b):
    return -(-a // b)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # use first fit to get the number of micro batches
    max_tokens_per_gpu *= mpu.get_context_parallel_world_size()
    batches = []
    for l in total_lengths:
        for i in range(len(batches)):
            if batches[i] + l <= max_tokens_per_gpu:
                batches[i] += l
                break
        else:
            batches.append(l)

    return len(batches)


def get_data_iterator(args, model):
    num_local_samples = (
        args.rollout_batch_size
        * args.n_samples_per_prompt
        // mpu.get_data_parallel_world_size(with_context_parallel=False)
    )
    num_local_gbs = args.global_batch_size // mpu.get_data_parallel_world_size(with_context_parallel=False)
    num_steps_per_rollout = num_local_samples // num_local_gbs

    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    config = get_model_config(model[0])

    if vpp_size is None:
        vpp_size = 1

    if not args.use_dynamic_batch_size:
        log_probs_num_microbatches = num_local_samples // args.ref_micro_batch_size
        train_num_microbatches = [num_local_gbs // args.micro_batch_size for _ in range(num_steps_per_rollout)]

        log_probs_data_iterator = []
        train_data_iterator = []
        for i in range(vpp_size):
            log_probs_data_iterator.append(DataIterator(args.ref_micro_batch_size))
            train_data_iterator.append(DataIterator(args.micro_batch_size))
    else:
        assert args.max_tokens_per_gpu is not None
        # calculate the number of mirobatches for each step
        samples = LOCAL_STORAGE["total_lengths"]
        assert len(samples) == num_local_samples
        num_microbatches = []
        for i in range(num_steps_per_rollout):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            num_microbatches.append(get_minimum_num_micro_batch_size(samples[start:end], args.max_tokens_per_gpu))

        num_microbatches.append(get_minimum_num_micro_batch_size(samples, args.max_tokens_per_gpu))

        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=mpu.get_data_parallel_group())

        # vpp requies the number of microbatches to be divisible by vpp_size
        if config.microbatch_group_size_per_vp_stage:
            num_microbatches = torch.clamp(
                num_microbatches
                // config.microbatch_group_size_per_vp_stage
                * config.microbatch_group_size_per_vp_stage,
                min=1,
            )

        num_microbatches = num_microbatches.tolist()
        log_probs_num_microbatches = num_microbatches.pop()
        train_num_microbatches = num_microbatches

        # balance the each micro batch
        samples = LOCAL_STORAGE["total_lengths"]
        # get log_probs data iterator
        partitions = get_seqlen_balanced_partitions(samples, log_probs_num_microbatches, equal_size=False)

        log_probs_data_iterator = []
        for i in range(vpp_size):
            log_probs_data_iterator.append(DataIterator(None, micro_batch_indices=partitions))

        # balance the number of mirobatches across steps
        micro_batch_indices = []
        for i, num_mbs in enumerate(train_num_microbatches):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            samples = LOCAL_STORAGE["total_lengths"][start:end]
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
            micro_batch_indices.extend(partitions)

        assert len(set(sum(micro_batch_indices, []))) == num_local_samples
        train_data_iterator = DataIterator(None, micro_batch_indices=micro_batch_indices)

        train_data_iterator = []
        for i in range(vpp_size):
            train_data_iterator.append(DataIterator(None, micro_batch_indices=micro_batch_indices))

    return (
        log_probs_data_iterator,
        log_probs_num_microbatches,
        train_data_iterator,
        train_num_microbatches,
    )


def process_rollout_data(rollout_id, args, data_buffer):
    rank = dist.get_rank()
    dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)

    if rank == 0:
        data = ray.get(data_buffer.get_data.remote(rollout_id))
        dist.broadcast_object_list([data], src=0)
    else:
        data = [None]
        dist.broadcast_object_list(data, src=0)
        data = data[0]

    # save the unprocessed reward for logging
    rewards = data["rewards"]
    if "raw_reward" in data:
        raw_rewards = data["raw_reward"]
    else:
        raw_rewards = rewards
    set_local_storage("raw_reward", raw_rewards)

    if args.advantage_estimator == "grpo" and args.rewards_normalization:
        # group norm
        rewards = torch.tensor([r for r in rewards], dtype=torch.float)
        rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean
        if args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)
        rewards = rewards.flatten().tolist()
        data["rewards"] = rewards

    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths

    if args.balance_data:
        parititions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)

    def get_partition(val):
        if args.balance_data:
            return [val[i] for i in parititions[dp_rank]]
        else:
            return val[dp_rank::dp_size]

    for key in [
        "tokens",
        "total_lengths",
        "response_lengths",
        "rewards",
        "truncated",
        "loss_masks",
        "round_number",
    ]:
        if key not in data:
            continue
        val = get_partition(data[key])
        # move tokens to GPU in advance
        if key == "tokens":
            val = [torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in val]
        elif key == "loss_masks":
            val = [torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in val]

        # save the data to local storage
        set_local_storage(key, val)


def log_rollout_data(rollout_id, args):
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        cp_size = mpu.get_context_parallel_world_size()
        log_dict = {}
        response_lengths = get_local_storage("response_lengths")
        for key, val in get_local_storage().items():
            if key == "tokens" or key == "loss_masks":
                continue
            # Upload per sample mean for each rollout value
            # There are the following assumptions:
            # - Each dp rank has the same number of samples
            if isinstance(val, list):
                if isinstance(val[0], torch.Tensor):
                    if cp_size == 1:
                        val = sum([v.mean() for v in val]) / len(val)
                    else:
                        # When cp_size > 1, the denominator should be the length of the response lengths. Also, to make
                        # sure these values can be divided by `mpu.get_data_parallel_world_size(with_context_parallel=True)`
                        # multiply by the cp_size.
                        val = sum([cp_size * v.sum() / l for v, l in zip(val, response_lengths)]) / len(val)
                else:
                    val = sum(val) / len(val)
            elif isinstance(val, torch.Tensor):
                val = val.float().mean()
            else:
                raise ValueError(f"Unsupported type: {type(val)}")
            log_dict[key] = val.item() if isinstance(val, torch.Tensor) else val

        if mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
            gathered_log_dict = [None] * mpu.get_data_parallel_world_size(with_context_parallel=True)
            # Not sure if this will be a performance bottleneck.
            dist.gather_object(
                log_dict,
                gathered_log_dict,
                dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
                group=mpu.get_data_parallel_group(with_context_parallel=True),
            )
            dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
            reduced_log_dict = {
                f"rollout/{key}": sum([d[key] for d in gathered_log_dict]) / dp_size for key in log_dict
            }
            print(f"rollout {rollout_id}: {reduced_log_dict}")
            if args.use_wandb:
                reduced_log_dict["rollout/step"] = (
                    rollout_id
                    if not args.wandb_always_use_train_step
                    else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
                )
                wandb.log(reduced_log_dict)
        else:
            dist.gather_object(
                log_dict,
                None,
                dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
                group=mpu.get_data_parallel_group(with_context_parallel=True),
            )

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args)
    if args.log_passrate:
        log_passrate(rollout_id, args)


def log_multi_turn_data(rollout_id, args):
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        cp_size = mpu.get_context_parallel_world_size()
        log_dict = {}
        response_lengths = get_local_storage("response_lengths")
        for key, val in get_local_storage().items():
            if key == "loss_masks":
                if val:  # Check if val is not empty
                    device = val[0].device  # Get device from first tensor

                    # Vectorized length calculation using torch
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths > args.rollout_max_response_len).float().mean().item()
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
        if mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
            gathered_log_dict = [None] * mpu.get_data_parallel_world_size(with_context_parallel=True)
            # Not sure if this will be a performance bottleneck.
            dist.gather_object(
                log_dict,
                gathered_log_dict,
                dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
                group=mpu.get_data_parallel_group(with_context_parallel=True),
            )
            dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
            reduced_log_dict = {
                f"multi_turn/{key}": sum([d[key] for d in gathered_log_dict]) / dp_size for key in log_dict
            }
            print(f"multi_turn {rollout_id}: {reduced_log_dict}")
            if args.use_wandb:
                wandb.log(reduced_log_dict)
        else:
            dist.gather_object(
                log_dict,
                None,
                dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
                group=mpu.get_data_parallel_group(with_context_parallel=True),
            )


def log_passrate(rollout_id, args):
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        cp_size = mpu.get_context_parallel_world_size()
        log_dict = {}
        response_lengths = get_local_storage("response_lengths")
        for key, val in get_local_storage().items():
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

        if mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
            gathered_log_dict = [None] * mpu.get_data_parallel_world_size(with_context_parallel=True)
            # Not sure if this will be a performance bottleneck.
            dist.gather_object(
                log_dict,
                gathered_log_dict,
                dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
                group=mpu.get_data_parallel_group(with_context_parallel=True),
            )
            dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
            reduced_log_dict = {
                f"passrate/{key}": sum([d[key] for d in gathered_log_dict]) / dp_size for key in log_dict
            }
            print(f"passrate {rollout_id}: {reduced_log_dict}")
            if args.use_wandb:
                wandb.log(reduced_log_dict)
        else:
            dist.gather_object(
                log_dict,
                None,
                dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
                group=mpu.get_data_parallel_group(with_context_parallel=True),
            )


def log_eval_data(rollout_id, args, data_buffer):
    if (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.is_pipeline_last_stage()
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    ):
        rank = dist.get_rank()
        data = ray.get(data_buffer.get_data.remote(rollout_id, evaluation=True))

        log_dict = {}
        for key in data.keys():
            rewards = data[key]["rewards"]
            log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
            if "truncated" in data[key]:
                truncated = data[key]["truncated"]
                log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)

        print(f"eval {rollout_id}: {log_dict}")
        if args.use_wandb:
            log_dict["eval/step"] = (
                rollout_id
                if not args.wandb_always_use_train_step
                else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
            )
            wandb.log(log_dict)


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
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]
            if "perf/ref_log_probs_time" in log_dict:
                log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
        if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
            log_dict["perf/total_train_time"] = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / log_dict["perf/total_train_time"]
        print(f"perf {rollout_id}: {log_dict}")
        if args.use_wandb:
            log_dict["rollout/step"] = (
                rollout_id
                if not args.wandb_always_use_train_step
                else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
            )
            wandb.log(log_dict)
    timer_instance.reset()
