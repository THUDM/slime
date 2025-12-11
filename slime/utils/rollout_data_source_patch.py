"""
Enhanced rollout_data_source.py with flexible sampling strategies.

This is a PATCH file showing the modifications needed.
Apply these changes to slime/ray/rollout_data_source.py
"""

# ============================================================================
# MODIFICATION 1: Import the new strategies module
# ============================================================================
# Add this import at the top of the file (around line 10)

from slime.utils.buffer_sampling_strategies import get_sampling_strategy


# ============================================================================
# MODIFICATION 2: Update RolloutDataSourceWithBuffer.__init__
# ============================================================================
# Replace the __init__ method (around line 134-170) with this enhanced version:

class RolloutDataSourceWithBuffer(RolloutDataSource):
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

        # === Sampling Strategy ===
        # NEW: Initialize sampling strategy instead of simple buffer_filter
        if self.buffer_enabled:
            self.sampling_strategy = get_sampling_strategy(args, self.current_policy_version)
            print(f"[Buffer] Sampling strategy: {self.sampling_strategy.get_name()}")
            print(f"[Buffer] Remove on sample: {self.sampling_strategy.remove_on_sample}")
            print(f"[Buffer] Max reuse count: {self.sampling_strategy.max_reuse_count}")
        else:
            self.sampling_strategy = None

        # === Backward compatibility: Keep buffer_filter for legacy code ===
        if self.args.buffer_filter_path is None:
            # Use legacy pop_first for backward compatibility
            from slime.utils.buffer_sampling_strategies import pop_first
            self.buffer_filter = pop_first
        else:
            from slime.utils.misc import load_function
            self.buffer_filter = load_function(self.args.buffer_filter_path)

        # === Statistics ===
        self.total_added = 0
        self.total_sampled = 0

        print(f"[Buffer] Initialized: enabled={self.buffer_enabled}, "
              f"max_size={self.buffer_max_size}")


# ============================================================================
# MODIFICATION 3: Update _get_samples_from_buffer method
# ============================================================================
# Replace the _get_samples_from_buffer method (around line 200-208) with this:

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


# ============================================================================
# MODIFICATION 4: Add update_policy_version method
# ============================================================================
# Add this new method to RolloutDataSourceWithBuffer class:

    def update_policy_version(self, version: int):
        """
        Update current policy version (called after training).

        Args:
            version: New policy version
        """
        self.current_policy_version = version

        # Update strategy's version
        if self.sampling_strategy is not None:
            self.sampling_strategy.current_policy_version = version


# ============================================================================
# MODIFICATION 5: Enhanced get_buffer_stats method
# ============================================================================
# Replace the get_buffer_stats method (around line 278-309) with this enhanced version:

    def get_buffer_stats(self) -> dict:
        """Get comprehensive buffer statistics including strategy info."""
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

        # Add sampling strategy stats
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

                # Check reuse count
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

            # Staleness statistics
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


# ============================================================================
# MODIFICATION 6: Keep legacy pop_first at module level (no change needed)
# ============================================================================
# The pop_first function at line 333-338 should remain for backward compatibility
# It's now imported from buffer_sampling_strategies.py
