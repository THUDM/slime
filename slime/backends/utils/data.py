from typing import Optional

import ray
import torch
import torch.distributed as dist

from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.timer import Timer


class DataIterator:
    def __init__(
        self,
        rollout_data,
        micro_batch_size: Optional[int] = None,
        micro_batch_indices: Optional[list[list[int]]] = None,
    ):
        self.rollout_data = rollout_data
        self.micro_batch_size = micro_batch_size
        self.micro_batch_indices = micro_batch_indices
        assert micro_batch_size is None or micro_batch_indices is None
        self.offset = 0

    def get_next(self, keys):
        batch = {}
        for key in keys:
            vals = self.rollout_data.get(key, None)
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


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu, cp_size):
    # use first fit to get the number of micro batches
    max_tokens_per_gpu *= cp_size
    batches = []
    for l in total_lengths:
        for i in range(len(batches)):
            if batches[i] + l <= max_tokens_per_gpu:
                batches[i] += l
                break
        else:
            batches.append(l)

    return len(batches)


def get_data_iterator(args, model, rollout_data):
    """
    Creates data iterators for training and log probability evaluation, supporting both static and dynamic batch sizes,
    with optional virtual pipeline parallelism and sequence length balancing.
    Args:
        args: An object containing configuration parameters, including batch sizes, micro batch sizes,
              dynamic batch size usage, and maximum tokens per GPU et.al.
        model: The model or list of model stages, used to extract configuration for parallelism.
        rollout_data: A dictionary containing rollout data, including 'total_lengths' for each sample.
    Returns:
        tuple: A tuple containing:
            - data_iterator: List of DataIterator objects for log probability evaluation.
            - num_microbatches: Number of microbatches for log probability evaluation.
    """
    from megatron.core import mpu

    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    dp_group = mpu.get_data_parallel_group()
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is None:
        vpp_size = 1
    if vpp_size > 1:
        from megatron.core.utils import get_model_config

        config = get_model_config(model[0])
        microbatch_group_size_per_vp_stage = config.microbatch_group_size_per_vp_stage
    cp_size = mpu.get_context_parallel_world_size()

    num_local_samples = len(rollout_data["total_lengths"])
    num_local_gbs = args.global_batch_size // dp_size
    num_steps_per_rollout = num_local_samples // num_local_gbs

    def _generate_data_iterator(rollout_data, micro_batch_size, micro_batch_indices=None):
        data_iterator = []
        for _ in range(vpp_size):
            data_iterator.append(DataIterator(rollout_data, micro_batch_size, micro_batch_indices))
        return data_iterator

    if not args.use_dynamic_batch_size:
        num_microbatches = [num_local_gbs // args.micro_batch_size for _ in range(num_steps_per_rollout)]
        data_iterator = _generate_data_iterator(rollout_data, args.micro_batch_size)
    else:
        assert args.max_tokens_per_gpu is not None
        # calculate the number of mirobatches for each step
        samples = rollout_data["total_lengths"]
        assert len(samples) == num_local_samples
        num_microbatches = []
        for i in range(num_steps_per_rollout):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            num_microbatches.append(
                get_minimum_num_micro_batch_size(samples[start:end], args.max_tokens_per_gpu, cp_size)
            )

        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)

        if vpp_size > 1:
            # vpp requies the number of microbatches to be divisible by vpp_size
            num_microbatches = torch.clamp(
                num_microbatches // microbatch_group_size_per_vp_stage * microbatch_group_size_per_vp_stage,
                min=1,
            )

        num_microbatches = num_microbatches.tolist()

        # balance the each micro batch
        samples = rollout_data["total_lengths"]
        # balance the number of mirobatches across steps
        micro_batch_indices = []
        for i, num_mbs in enumerate(num_microbatches):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            samples = rollout_data["total_lengths"][start:end]
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
            micro_batch_indices.extend(partitions)

        assert len(set(sum(micro_batch_indices, []))) == num_local_samples

        data_iterator = _generate_data_iterator(rollout_data, None, micro_batch_indices)

    return (
        data_iterator,
        num_microbatches,
    )


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    rollout_data = {}

    rank = dist.get_rank()
    if rank == 0:
        data = ray.get(rollout_data_ref.inner)
        dist.broadcast_object_list([data], src=0)
    else:
        data = [None]
        dist.broadcast_object_list(data, src=0)
        data = data[0]

    # save the unprocessed reward for logging
    rollout_data["raw_reward"] = data["raw_reward"]

    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths

    if args.balance_data:
        # Group-aware partitioning to keep each group together
        n_samples_per_prompt = getattr(args, "n_samples_per_prompt", 1)
        # Calculate group-level lengths (sum of lengths for each group)
        num_groups = len(total_lengths) // n_samples_per_prompt
        group_lengths = []
        for i in range(num_groups):
            start_idx = i * n_samples_per_prompt
            end_idx = start_idx + n_samples_per_prompt
            group_total_length = sum(total_lengths[start_idx:end_idx])
            group_lengths.append(group_total_length)

        # Get partitions at group level
        group_partitions = get_seqlen_balanced_partitions(group_lengths, dp_size, equal_size=True)

        # Expand group partitions to trajectory level
        parititions = []
        for dp_rank_groups in group_partitions:
            trajectory_indices = []
            for group_idx in dp_rank_groups:
                # Add all trajectories in this group
                start_idx = group_idx * n_samples_per_prompt
                end_idx = start_idx + n_samples_per_prompt
                trajectory_indices.extend(range(start_idx, end_idx))
            parititions.append(trajectory_indices)

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
        "sample_indices",
        "rollout_log_probs",
    ]:
        if key not in data:
            continue
        val = get_partition(data[key])
        rollout_data[key] = val

    return rollout_data
