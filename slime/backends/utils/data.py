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
    rewards = data["rewards"]
    if "raw_reward" in data:
        raw_rewards = data["raw_reward"]
    else:
        raw_rewards = rewards
    rollout_data["raw_reward"] = raw_rewards

    if args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
        # group norm
        rewards = torch.tensor([r for r in rewards], dtype=torch.float)
        rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)

        rewards = rewards.flatten().tolist()
        data["rewards"] = rewards

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
    ]:
        if key not in data:
            continue
        val = get_partition(data[key])
        # move tokens to GPU in advance
        if key == "tokens":
            val = [torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in val]
        elif key == "loss_masks":
            val = [torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in val]

        rollout_data[key] = val

    return rollout_data
