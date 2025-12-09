import copy
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.types import Sample


# TODO may further refactor data-loading part later
class RolloutDataSource:
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        if args.rollout_global_dataset:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def get_samples(self, num_samples):
        # TODO further improve code
        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            print(f"Checkpoint {path} does not exist.")
            return

        print(f"load metadata from {path}")
        print(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    """
    Enhanced data source with unified buffer support for both on-policy and off-policy GRPO.

    Features:
    - Auto-detection: Automatically enables/disables buffer based on loss_type
    - On-Policy: Buffer disabled by default (use data once)
    - Off-Policy: Buffer enabled by default (reuse data within staleness limit)
    - Configurable: Can override with --use_buffer flag
    - Size-limited: Automatically evicts oldest data when buffer is full
    """

    def __init__(self, args):
        super().__init__(args)

        # === Buffer Storage ===
        self.buffer = []

        # === Buffer Configuration ===
        self.buffer_max_size = getattr(args, 'buffer_max_size', 1000)
        self.buffer_enabled = getattr(args, 'use_buffer', None)

        # === Auto-detect buffer usage based on loss_type ===
        if self.buffer_enabled is None:
            loss_type = getattr(args, 'loss_type', 'policy_loss')
            self.buffer_enabled = (loss_type == 'decoupled_policy_loss')

            if self.buffer_enabled:
                print(f"[Buffer] Auto-enabled for off-policy GRPO (loss_type={loss_type})")
            else:
                print(f"[Buffer] Disabled for on-policy GRPO (loss_type={loss_type})")

        # === Policy Version Tracking ===
        self.current_policy_version = 0

        # === Buffer Sampling Strategy ===
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

        # === Statistics ===
        self.total_added = 0
        self.total_sampled = 0

        print(f"[Buffer] Initialized: enabled={self.buffer_enabled}, "
              f"max_size={self.buffer_max_size}, "
              f"filter={self.buffer_filter.__name__}")

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Get samples with automatic buffer mixing.

        Behavior:
        - If buffer disabled: return only new samples from dataset (on-policy)
        - If buffer enabled: prioritize buffer, then fill with dataset (off-policy)

        Args:
            num_samples: Number of sample groups to retrieve

        Returns:
            List of sample groups
        """
        if not self.buffer_enabled:
            # Buffer disabled: use only new data (on-policy behavior)
            return super().get_samples(num_samples=num_samples)

        # Buffer enabled: sample from buffer first (off-policy behavior)
        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        # If buffer doesn't have enough, get new data from dataset
        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        """Sample from buffer using configured filter"""
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        # Pass current policy version to filter for staleness-aware sampling
        samples = self.buffer_filter(self.args, self.current_policy_version, self.buffer, num_samples)
        self.total_sampled += len(samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to buffer with automatic size management.

        Automatically handles both formats:
        - list[Sample]: Flat list (will be grouped by n_samples_per_prompt)
        - list[list[Sample]]: Already grouped

        Args:
            samples: List of samples or list of sample groups
        """
        if not samples:
            return

        if not self.buffer_enabled:
            # Buffer disabled: do nothing (on-policy - use data once)
            return

        # Validation
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"

        # === Auto-detect format and convert if needed ===
        if len(samples) > 0 and not isinstance(samples[0], list):
            # Format: list[Sample] (flattened)
            # Need to group by n_samples_per_prompt
            print(f"[Buffer] Auto-converting flat list to grouped format "
                  f"(n_samples_per_prompt={self.args.n_samples_per_prompt})")

            grouped_samples = []
            for i in range(0, len(samples), self.args.n_samples_per_prompt):
                group = samples[i:i + self.args.n_samples_per_prompt]
                # Only add complete groups
                if len(group) == self.args.n_samples_per_prompt:
                    grouped_samples.append(group)
                else:
                    print(f"[Buffer] Warning: Incomplete group with {len(group)} samples "
                          f"(expected {self.args.n_samples_per_prompt}), skipping")

            samples = grouped_samples

        # Now samples is guaranteed to be list[list[Sample]]
        # Validate group structure
        for i, group in enumerate(samples):
            assert isinstance(group, list), \
                f"Group {i} must be a list, got {type(group)}"
            assert len(group) == self.args.n_samples_per_prompt, \
                f"Group {i} has {len(group)} samples, expected {self.args.n_samples_per_prompt}"

        # Add samples to buffer
        for group in samples:
            self.buffer.append(group)
            self.total_added += 1

        # Enforce buffer size limit (FIFO eviction)
        evicted_count = 0
        while len(self.buffer) > self.buffer_max_size:
            evicted = self.buffer.pop(0)
            evicted_count += 1

        if evicted_count > 0:
            print(f"[Buffer] Evicted {evicted_count} oldest groups. "
                  f"Size: {len(self.buffer)}/{self.buffer_max_size}")

    def get_buffer_stats(self) -> dict:
        """
        Get comprehensive buffer statistics for monitoring.

        Returns:
            Dictionary with buffer metrics
        """
        if not self.buffer_enabled:
            return {
                "enabled": False,
                "buffer_size": 0,
            }

        stats = {
            "enabled": True,
            "buffer_size": len(self.buffer),
            "buffer_max_size": self.buffer_max_size,
            "buffer_utilization": len(self.buffer) / max(self.buffer_max_size, 1),
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
        }

        # Calculate policy version distribution
        policy_versions = []
        for group in self.buffer:
            for sample in group:
                if hasattr(sample, 'policy_version') and sample.policy_version is not None:
                    policy_versions.append(sample.policy_version)

        if policy_versions:
            stats.update({
                "min_policy_version": min(policy_versions),
                "max_policy_version": max(policy_versions),
                "avg_policy_version": sum(policy_versions) / len(policy_versions),
                "current_policy_version": self.current_policy_version,
            })

        return stats

    def update_policy_version(self, version: int):
        """Update current policy version for staleness-aware sampling"""
        self.current_policy_version = version

    def clear_buffer(self):
        """Clear all buffered samples"""
        self.buffer.clear()
        print(f"[Buffer] Cleared. total_added={self.total_added}, total_sampled={self.total_sampled}")

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, current_policy_version, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    """Simple FIFO buffer filter (default)"""
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
