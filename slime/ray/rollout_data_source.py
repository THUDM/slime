import copy
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.types import Sample
from slime.utils.buffer_sampling_strategies import get_sampling_strategy


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

        # === Sampling Strategy (NEW: Extensible strategy framework) ===
        if self.buffer_enabled:
            # Use new sampling strategy framework
            self.sampling_strategy = get_sampling_strategy(args, self.current_policy_version)
            print(f"[Buffer] Sampling strategy: {self.sampling_strategy.get_name()}")
            print(f"[Buffer] Remove on sample: {self.sampling_strategy.remove_on_sample}")
            print(f"[Buffer] Max reuse count: {self.sampling_strategy.max_reuse_count}")
        else:
            self.sampling_strategy = None

        # === Backward compatibility: Keep buffer_filter for legacy code ===
        if self.args.buffer_filter_path is None:
            # Legacy behavior
            from slime.utils.buffer_sampling_strategies import pop_first
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

        # === Statistics ===
        self.total_added = 0
        self.total_sampled = 0

        print(f"[Buffer] Initialized: enabled={self.buffer_enabled}, "
              f"max_size={self.buffer_max_size}")

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

        # 🔧 FIX: When buffer is exhausted/empty, don't fallback to dataset
        # In off-policy mode with buffer, getting new prompts from dataset here
        # would pollute the buffer with incomplete samples.
        # The correct flow is:
        # 1. Buffer exhausted -> return what we have (even if less than requested)
        # 2. Caller (rollout) will notice and trigger new rollout generation
        # 3. New rollout generates complete samples and adds to buffer

        if len(samples) < num_samples:
            print(f"[Buffer] WARNING: Buffer only provided {len(samples)}/{num_samples + len(samples)} requested samples.")

            if len(samples) == 0:
                # Buffer is completely empty or all exhausted
                print(f"[Buffer] Buffer is empty/exhausted. Need new rollout to generate data.")
                print(f"[Buffer] Falling back to dataset for {num_samples} new prompts - these MUST go through rollout generation.")
                # Get new prompts that will be generated in this rollout
                samples = super().get_samples(num_samples=num_samples)

            else:
                # Buffer provided some but not enough
                # For now, just return what we have. The caller can decide whether to:
                # - Use fewer samples for this iteration
                # - Or get more from dataset (but they must be generated!)
                print(f"[Buffer] Returning {len(samples)} samples (short by {num_samples}).")
                print(f"[Buffer] Consider: (1) increasing buffer size, (2) decreasing reuse count limit, or (3) generating more frequently.")

        return samples

    def get_training_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Get samples ONLY for training purposes.

        CRITICAL DIFFERENCE from get_samples():
        - get_samples(): Used for rollout generation, may fallback to dataset for new prompts
        - get_training_samples(): Used for training, ONLY returns complete samples from buffer

        This method:
        - ONLY samples from buffer (never falls back to dataset)
        - Returns empty list if buffer is exhausted
        - Ensures all returned samples are COMPLETE (have response and reward)

        Args:
            num_samples: Number of sample groups requested

        Returns:
            List of complete sample groups from buffer (may be less than requested or empty)
        """
        if not self.buffer_enabled:
            # If buffer disabled, fall back to get_samples (on-policy mode)
            return self.get_samples(num_samples=num_samples)

        # Sample from buffer ONLY - no fallback to dataset
        samples = self._get_samples_from_buffer(num_samples)

        if len(samples) < num_samples:
            if len(samples) == 0:
                print(f"[Buffer Training] No samples available in buffer for training.")
                print(f"[Buffer Training] Buffer is exhausted. Next rollout will generate new data.")
            else:
                print(f"[Buffer Training] Only {len(samples)}/{num_samples} samples available.")
                print(f"[Buffer Training] Consider increasing buffer size or reducing reuse limit.")

        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        """Sample from buffer using configured sampling strategy"""
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        # NEW: Use sampling strategy instead of direct buffer_filter call
        if self.sampling_strategy is not None:
            # Update strategy's current_policy_version
            self.sampling_strategy.current_policy_version = self.current_policy_version

            # Sample using strategy
            samples = self.sampling_strategy.sample(self.buffer, num_samples)

            # Update statistics
            self.total_sampled += len(samples)

            # Log sampling info
            if len(samples) > 0:
                versions = [s[0].policy_version for s in samples if s[0].policy_version is not None]
                if versions:
                    print(f"[Buffer Sampling] Sampled {len(samples)} groups. "
                          f"Version range: [{min(versions)}, {max(versions)}], "
                          f"Current version: {self.current_policy_version}")

            return samples
        else:
            # Fallback to legacy buffer_filter (backward compatibility)
            samples = self.buffer_filter(
                self.args,
                self.current_policy_version,
                self.buffer,
                num_samples
            )
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

        # 🔧 FIX: Policy-version-aware duplicate detection for Off-Policy training
        # Key insight for Off-Policy GRPO:
        # - group_index identifies the PROMPT
        # - Same prompt + different policy version = DIFFERENT valuable data (should KEEP both!)
        # - Same prompt + same policy version = TRUE duplicate (should skip)
        #
        # Off-Policy training benefits from keeping multiple versions:
        # - Diversity: Same prompt with different responses from different policies
        # - Data efficiency: Reuse historical data generated by older policies
        # - Buffer utilization: Don't waste any generated samples

        # Build index: (group_index, policy_version) -> True (for exact duplicate detection)
        buffer_index = set()
        for group in self.buffer:
            if len(group) > 0 and hasattr(group[0], 'group_index') and group[0].group_index is not None:
                group_index = group[0].group_index
                policy_version = getattr(group[0], 'policy_version', None)
                key = (group_index, policy_version)
                buffer_index.add(key)

        new_groups = []
        duplicate_count = 0
        incomplete_count = 0
        new_version_count = 0  # Track new versions of existing prompts

        for group in samples:
            if len(group) == 0:
                continue

            # 🔧 DEFENSIVE: Verify all samples in group have responses
            group_is_complete = all(
                hasattr(s, 'response') and s.response is not None and s.response != ''
                for s in group
            )

            if not group_is_complete:
                incomplete_count += 1
                print(f"[Buffer] WARNING: Skipping incomplete group {group[0].group_index if hasattr(group[0], 'group_index') else '?'}: "
                      f"samples lack response. This may indicate a generation failure.")
                continue  # Skip incomplete group
            
            # TODO
            # # Check for exact duplicates (same prompt + same policy version)
            # if hasattr(group[0], 'group_index') and group[0].group_index is not None:
            #     group_index = group[0].group_index
            #     policy_version = getattr(group[0], 'policy_version', None)
            #     key = (group_index, policy_version)

            #     # Exact duplicate check
            #     if key in buffer_index:
            #         duplicate_count += 1
            #         continue  # Skip exact duplicate

            #     # Check if this is a new version of an existing prompt
            #     existing_versions = [k for k in buffer_index if k[0] == group_index]
            #     if len(existing_versions) > 0:
            #         new_version_count += 1

            new_groups.append(group)

        if duplicate_count > 0:
            print(f"[Buffer] Filtered {duplicate_count} exact duplicate groups (same prompt + same policy version)")

        if incomplete_count > 0:
            print(f"[Buffer] Filtered {incomplete_count} incomplete groups (no response)")

        if new_version_count > 0:
            print(f"[Buffer] Adding {new_version_count} new versions of existing prompts (Off-Policy diversity)")

        if len(new_groups) == 0 and len(samples) > 0:
            print(f"[Buffer] No new groups to add ({duplicate_count} exact duplicates, {incomplete_count} incomplete)")
            return

        # Add samples to buffer
        for group in new_groups:
            self.buffer.append(group)
            self.total_added += 1

        print(f"[Buffer] Added {len(new_groups)} new groups to buffer. "
              f"Buffer size: {len(self.buffer)}/{self.buffer_max_size} "
              f"({duplicate_count} duplicates skipped, {incomplete_count} incomplete skipped)")

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

        # Add sampling strategy stats (NEW)
        if self.sampling_strategy is not None:
            strategy_stats = self.sampling_strategy.get_statistics()
            stats.update({f"strategy_{k}": v for k, v in strategy_stats.items()})

        # Calculate policy version distribution
        policy_versions = []
        reuse_counts = []
        for group in self.buffer:
            for sample in group:
                if hasattr(sample, 'policy_version') and sample.policy_version is not None:
                    policy_versions.append(sample.policy_version)

                # Check reuse count (NEW)
                if sample.metadata and 'buffer_reuse_count' in sample.metadata:
                    reuse_counts.append(sample.metadata['buffer_reuse_count'])

        if policy_versions:
            # Version statistics
            stats.update({
                "min_policy_version": min(policy_versions),
                "max_policy_version": max(policy_versions),
                "avg_policy_version": sum(policy_versions) / len(policy_versions),
                "current_policy_version": self.current_policy_version,
            })

            # Staleness statistics (NEW)
            staleness_values = [
                self.current_policy_version - v
                for v in policy_versions
            ]
            stats.update({
                "min_staleness": min(staleness_values),
                "max_staleness": max(staleness_values),
                "avg_staleness": sum(staleness_values) / len(staleness_values),
            })

        if reuse_counts:
            stats.update({
                "avg_reuse_count": sum(reuse_counts) / len(reuse_counts),
                "max_reuse_count_observed": max(reuse_counts),
            })

        return stats

    def update_policy_version(self, version: int):
        """
        Update current policy version for staleness-aware sampling.

        Args:
            version: New policy version
        """
        self.current_policy_version = version

        # Update strategy's version (NEW)
        if self.sampling_strategy is not None:
            self.sampling_strategy.current_policy_version = version

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
