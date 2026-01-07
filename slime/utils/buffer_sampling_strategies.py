"""
Buffer Sampling Strategies for Off-Policy GRPO.

This module provides a flexible, extensible framework for different buffer sampling strategies.
All strategies follow a common interface and support key features:
- Staleness filtering
- Sample reuse control
- Priority-based sampling
- Customizable metrics

Usage:
    from slime.utils.buffer_sampling_strategies import get_sampling_strategy

    strategy = get_sampling_strategy(args)
    sampled_groups = strategy.sample(buffer, num_samples)
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from slime.utils.types import Sample


class BaseSamplingStrategy(ABC):
    """
    Abstract base class for buffer sampling strategies.

    All strategies must implement:
    - sample(): Core sampling logic
    - get_name(): Strategy identifier
    """

    def __init__(self, args, current_policy_version: int):
        """
        Initialize sampling strategy.

        Args:
            args: Configuration namespace
            current_policy_version: Current policy version for staleness computation
        """
        self.args = args
        self.current_policy_version = current_policy_version

        # Core configuration
        self.max_staleness = getattr(args, 'max_staleness', -1)
        self.remove_on_sample = getattr(args, 'buffer_remove_on_sample', True)
        self.max_reuse_count = getattr(args, 'buffer_reuse_samples', 1)

        # Statistics
        self.total_sampled = 0
        self.total_filtered = 0

    @abstractmethod
    def sample(
        self,
        buffer: List[List[Sample]],
        num_samples: int
    ) -> List[List[Sample]]:
        """
        Sample groups from buffer.

        Args:
            buffer: List of sample groups (modified in-place if remove_on_sample=True)
            num_samples: Number of groups to sample

        Returns:
            List of sampled groups
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name for logging."""
        pass

    def filter_by_staleness(
        self,
        buffer: List[List[Sample]]
    ) -> tuple[List[List[Sample]], List[List[Sample]]]:
        """
        Filter buffer by staleness constraint.

        Args:
            buffer: List of sample groups

        Returns:
            (valid_groups, stale_groups) tuple
        """
        if self.max_staleness < 0:
            # No staleness constraint
            return buffer, []

        valid_groups = []
        stale_groups = []

        for group in buffer:
            # Check first sample's policy_version (all samples in group have same version)
            sample = group[0]

            if sample.policy_version is None:
                # No version info, assume valid
                valid_groups.append(group)
                continue

            staleness = self.current_policy_version - sample.policy_version

            if staleness <= self.max_staleness:
                valid_groups.append(group)
            else:
                stale_groups.append(group)

        self.total_filtered += len(stale_groups)
        return valid_groups, stale_groups

    def filter_by_reuse_count(
        self,
        groups: List[List[Sample]]
    ) -> List[List[Sample]]:
        """
        Filter groups that have been reused too many times.

        Args:
            groups: List of sample groups

        Returns:
            Filtered list of groups
        """
        if self.max_reuse_count <= 0:
            # Unlimited reuse
            return groups

        valid_groups = []
        for group in groups:
            # Check reuse count (stored in metadata)
            sample = group[0]
            reuse_count = sample.metadata.get('buffer_reuse_count', 0) if sample.metadata else 0

            if reuse_count < self.max_reuse_count:
                valid_groups.append(group)

        return valid_groups

    def increment_reuse_count(self, groups: List[List[Sample]]):
        """Increment reuse count for sampled groups."""
        for group in groups:
            for sample in group:
                if sample.metadata is None:
                    sample.metadata = {}
                reuse_count = sample.metadata.get('buffer_reuse_count', 0)
                sample.metadata['buffer_reuse_count'] = reuse_count + 1

    def remove_from_buffer(
        self,
        buffer: List[List[Sample]],
        groups_to_remove: List[List[Sample]]
    ):
        """
        Remove groups from buffer (in-place).

        Args:
            buffer: Buffer to modify
            groups_to_remove: Groups to remove
        """
        for group in groups_to_remove:
            if group in buffer:
                buffer.remove(group)

    def get_statistics(self) -> dict:
        """Return sampling statistics."""
        return {
            "strategy": self.get_name(),
            "total_sampled": self.total_sampled,
            "total_filtered": self.total_filtered,
            "remove_on_sample": self.remove_on_sample,
            "max_reuse_count": self.max_reuse_count,
        }


class FIFOWithStalenessStrategy(BaseSamplingStrategy):
    """
    FIFO (First-In-First-Out) sampling with staleness filtering.

    This is the DEFAULT and RECOMMENDED strategy for off-policy GRPO.

    Features:
    - Filters out samples with staleness > max_staleness
    - Samples from oldest (head) to newest
    - Simple, efficient, and predictable
    - Ensures all valid data gets used

    Recommended for:
    - Standard off-policy training
    - When you want consistent, predictable behavior
    - When data diversity is important
    """

    def get_name(self) -> str:
        return "fifo_with_staleness"

    def sample(
        self,
        buffer: List[List[Sample]],
        num_samples: int
    ) -> List[List[Sample]]:
        """
        Sample using FIFO with staleness filtering and version-aware stratified sampling.

        Algorithm:
        1. Filter by staleness
        2. Filter by reuse count
        3. 🔧 NEW: Stratified sampling by version for diversity
        4. Remove from buffer if remove_on_sample=True or if exhausted
        """
        if len(buffer) == 0 or num_samples == 0:
            return []

        # Step 1: Filter by staleness
        valid_groups, stale_groups = self.filter_by_staleness(buffer)

        # Step 2: Filter by reuse count AND identify exhausted groups immediately
        reusable_groups = []
        exhausted_groups = []

        if self.max_reuse_count > 0:
            for group in valid_groups:
                sample = group[0]
                reuse_count = sample.metadata.get('buffer_reuse_count', 0) if sample.metadata else 0

                if reuse_count < self.max_reuse_count:
                    reusable_groups.append(group)
                else:
                    # 🔧 FIX: Identify exhausted groups immediately, not just when all are exhausted
                    exhausted_groups.append(group)
        else:
            reusable_groups = valid_groups

        # Step 3: 🔧 NEW: Version-aware stratified sampling for off-policy diversity
        # Instead of pure FIFO (always taking oldest), stratify by version to ensure
        # each training batch contains samples from multiple policy versions
        num_to_sample = min(len(reusable_groups), num_samples)

        if num_to_sample > 0:
            sampled = self._stratified_sample_by_version(reusable_groups, num_to_sample)
        else:
            sampled = []

        # Step 4: Update reuse count (if not removing)
        if not self.remove_on_sample and len(sampled) > 0:
            self.increment_reuse_count(sampled)

        # Step 5: Remove from buffer
        if self.remove_on_sample and len(sampled) > 0:
            # Remove sampled groups
            self.remove_from_buffer(buffer, sampled)

        # Always remove stale groups
        if stale_groups:
            print(f"[Buffer Sampling] Removing {len(stale_groups)} stale groups "
                  f"(staleness > {self.max_staleness})")
            self.remove_from_buffer(buffer, stale_groups)

        # 🔧 FIX: Remove exhausted groups immediately (not just when all exhausted)
        if exhausted_groups:
            print(f"[Buffer Sampling] Removing {len(exhausted_groups)} exhausted groups "
                  f"(reuse_count >= {self.max_reuse_count}) to free up space")
            self.remove_from_buffer(buffer, exhausted_groups)

        self.total_sampled += len(sampled)
        return sampled

    def _stratified_sample_by_version(
        self,
        groups: List[List[Sample]],
        num_samples: int
    ) -> List[List[Sample]]:
        """
        Stratified sampling by policy version to ensure diversity.

        Key insight for off-policy GRPO:
        - Training should use samples from MULTIPLE policy versions
        - Not just the oldest version (pure FIFO problem)
        - Importance weight correction handles version mismatch

        Algorithm:
        1. Group samples by policy_version
        2. Sample proportionally from each version
        3. Fill remaining slots with round-robin across versions

        Args:
            groups: Available groups (already filtered by staleness & reuse)
            num_samples: Number of groups to sample

        Returns:
            Sampled groups with version diversity
        """
        if len(groups) == 0:
            return []

        # Group by version
        version_groups = {}
        for group in groups:
            version = group[0].policy_version if group[0].policy_version is not None else -1
            if version not in version_groups:
                version_groups[version] = []
            version_groups[version].append(group)

        num_versions = len(version_groups)

        # If only one version, fall back to FIFO
        if num_versions == 1:
            return groups[:num_samples]

        # Stratified sampling: sample proportionally from each version
        sampled = []
        samples_per_version = num_samples // num_versions
        remainder = num_samples % num_versions

        # 🔧 FIX: Sort versions newest-first to prioritize fresh data for off-policy GRPO
        # This ensures that when num_samples < num_versions, we sample from newest versions
        # and discard oldest (not the other way around, which would harm IS weight stability)
        sorted_versions = sorted(version_groups.keys(), reverse=True)

        # Phase 1: Sample proportionally from each version
        for version in sorted_versions:
            version_pool = version_groups[version]
            # Take from front (FIFO within each version)
            quota = samples_per_version
            if remainder > 0:
                quota += 1
                remainder -= 1

            take = min(quota, len(version_pool))
            sampled.extend(version_pool[:take])

        # Phase 2: If we haven't sampled enough (some versions exhausted),
        # fill remainder with round-robin across versions
        if len(sampled) < num_samples:
            # Collect remaining samples from each version
            remaining_pools = {}
            for version in sorted_versions:
                used = samples_per_version + (1 if version in sorted_versions[:num_samples % num_versions] else 0)
                used = min(used, len(version_groups[version]))
                if len(version_groups[version]) > used:
                    remaining_pools[version] = version_groups[version][used:]

            # Round-robin fill
            while len(sampled) < num_samples and remaining_pools:
                for version in sorted(remaining_pools.keys()):
                    if len(sampled) >= num_samples:
                        break
                    if remaining_pools[version]:
                        sampled.append(remaining_pools[version].pop(0))
                    if not remaining_pools[version]:
                        del remaining_pools[version]

        # Log version distribution
        sampled_versions = [g[0].policy_version for g in sampled if g[0].policy_version is not None]
        if sampled_versions:
            version_counts = {}
            for v in sampled_versions:
                version_counts[v] = version_counts.get(v, 0) + 1
            print(f"[Buffer Sampling] Stratified sample: {dict(sorted(version_counts.items()))} "
                  f"(total={len(sampled)} groups from {len(version_counts)} versions)")

        return sampled[:num_samples]


class LIFOWithStalenessStrategy(BaseSamplingStrategy):
    """
    LIFO (Last-In-First-Out) sampling with staleness filtering.

    Prioritizes NEWEST data to minimize importance weight variance in off-policy GRPO.

    Key advantages:
    - Newest data has π_behavior ≈ π_θ → IS weight ≈ 1.0
    - Minimal gradient variance
    - Fastest convergence
    - Most stable training

    Recommended for:
    - Off-policy GRPO where IS weight stability is critical
    - Fast convergence scenarios
    - When rollout data is abundant
    - Aggressive off-policy training

    Trade-offs:
    - May waste older data if buffer is always full
    - Lower data utilization compared to FIFO
    - Requires higher rollout throughput
    """

    def get_name(self) -> str:
        return "lifo_with_staleness"

    def sample(
        self,
        buffer: List[List[Sample]],
        num_samples: int
    ) -> List[List[Sample]]:
        """
        Sample using LIFO (newest-first) with staleness filtering.

        Algorithm:
        1. Filter by staleness
        2. Filter by reuse count (identify exhausted groups)
        3. Take from END of buffer (newest data)
        4. Update reuse count or remove
        5. Clean up stale and exhausted groups
        """
        if len(buffer) == 0 or num_samples == 0:
            return []

        # Step 1: Filter by staleness
        valid_groups, stale_groups = self.filter_by_staleness(buffer)

        # Step 2: Filter by reuse count AND identify exhausted groups
        reusable_groups = []
        exhausted_groups = []

        if self.max_reuse_count > 0:
            for group in valid_groups:
                sample = group[0]
                reuse_count = sample.metadata.get('buffer_reuse_count', 0) if sample.metadata else 0

                if reuse_count < self.max_reuse_count:
                    reusable_groups.append(group)
                else:
                    exhausted_groups.append(group)
        else:
            reusable_groups = valid_groups

        # Step 3: LIFO - take from END (newest samples)
        num_to_sample = min(len(reusable_groups), num_samples)
        sampled = reusable_groups[-num_to_sample:] if num_to_sample > 0 else []

        # Step 4: Update reuse count (if not removing)
        if not self.remove_on_sample and sampled:
            self.increment_reuse_count(sampled)

        # Step 5: Remove from buffer
        if self.remove_on_sample and sampled:
            self.remove_from_buffer(buffer, sampled)

        # Always remove stale groups
        if stale_groups:
            print(f"[Buffer Sampling] Removing {len(stale_groups)} stale groups "
                  f"(staleness > {self.max_staleness})")
            self.remove_from_buffer(buffer, stale_groups)

        # Remove exhausted groups immediately
        if exhausted_groups:
            print(f"[Buffer Sampling] Removing {len(exhausted_groups)} exhausted groups "
                  f"(reuse_count >= {self.max_reuse_count})")
            self.remove_from_buffer(buffer, exhausted_groups)

        self.total_sampled += len(sampled)

        # Log version distribution for monitoring
        if sampled:
            versions = [g[0].policy_version for g in sampled if g[0].policy_version is not None]
            if versions:
                version_counts = {}
                for v in versions:
                    version_counts[v] = version_counts.get(v, 0) + 1
                version_range = f"[{min(versions)}, {max(versions)}]"
                print(f"[Buffer Sampling] LIFO sample: {dict(sorted(version_counts.items()))} "
                      f"(total={len(sampled)} groups, version_range={version_range}, newest-first)")

        return sampled


class PrioritySamplingStrategy(BaseSamplingStrategy):
    """
    Priority-based sampling with configurable metrics.

    Samples groups with higher priority first. Priority can be based on:
    - Reward: Higher reward → Higher priority
    - Advantage: Higher advantage → Higher priority (requires pre-computation)
    - Custom: User-defined metric

    Priority is adjusted by staleness penalty to balance data freshness.

    NEW: Normalized scoring with configurable weight balance:
    - Score = normalized_base_score × priority_weight + normalized_staleness × staleness_weight
    - All components normalized to [0, 1] for stable parameter tuning

    Recommended for:
    - When you want to focus on high-quality samples
    - Curriculum learning scenarios
    - When sample quality varies significantly
    """

    def __init__(self, args, current_policy_version: int):
        super().__init__(args, current_policy_version)

        # Priority configuration
        self.priority_metric = getattr(args, 'buffer_priority_metric', 'reward')
        self.priority_weight = getattr(args, 'buffer_priority_weight', 1.0)
        self.staleness_penalty = getattr(args, 'buffer_staleness_penalty', 0.1)

        # NEW: Normalization configuration
        self.normalize_scores = getattr(args, 'buffer_normalize_priority_scores', True)
        self.normalization_method = getattr(args, 'buffer_priority_norm_method', 'minmax')  # minmax, zscore, sigmoid

        # NEW: Running statistics for normalization
        self.base_score_stats = {'min': float('inf'), 'max': float('-inf'), 'sum': 0.0, 'count': 0}
        self.staleness_stats = {'min': float('inf'), 'max': float('-inf'), 'sum': 0.0, 'count': 0}

        # Custom metric function
        if self.priority_metric == 'custom':
            from slime.utils.misc import load_function
            custom_path = getattr(args, 'buffer_priority_custom_path', None)
            if custom_path:
                self.priority_func = load_function(custom_path)
            else:
                raise ValueError("buffer_priority_custom_path required for custom priority metric")
        else:
            self.priority_func = None

        # Statistics for monitoring
        self.priority_scores_history = []

    def get_name(self) -> str:
        return f"priority_{self.priority_metric}"

    def _normalize_value(self, value: float, stats: dict, method: str = 'minmax') -> float:
        """
        Normalize a value using running statistics.

        Args:
            value: Value to normalize
            stats: Dictionary with min, max, sum, count
            method: Normalization method (minmax, zscore, sigmoid)

        Returns:
            Normalized value in [0, 1] range (for minmax and sigmoid)
        """
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            if stats['max'] - stats['min'] > 1e-8:
                return (value - stats['min']) / (stats['max'] - stats['min'])
            else:
                return 0.5  # Default if no variation

        elif method == 'zscore':
            # Z-score normalization, then sigmoid to [0, 1]
            if stats['count'] > 0:
                mean = stats['sum'] / stats['count']
                # Estimate std using simple heuristic
                std = (stats['max'] - stats['min']) / 4.0 if stats['max'] - stats['min'] > 1e-8 else 1.0
                z_score = (value - mean) / std
                # Apply sigmoid to map to [0, 1]
                return 1.0 / (1.0 + np.exp(-z_score))
            else:
                return 0.5

        elif method == 'sigmoid':
            # Direct sigmoid (useful for unbounded scores)
            return 1.0 / (1.0 + np.exp(-value))

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _update_stats(self, value: float, stats: dict):
        """Update running statistics with new value."""
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)
        stats['sum'] += value
        stats['count'] += 1

    def compute_priority(self, group: List[Sample]) -> tuple[float, dict]:
        """
        Compute priority score for a group with optional normalization.

        NEW: Returns both the final score and detailed breakdown for monitoring.

        Args:
            group: List of samples in the group

        Returns:
            (score, breakdown_dict) where breakdown_dict contains:
                - base_score_raw: Raw base metric value
                - base_score_normalized: Normalized base score (if enabled)
                - staleness_raw: Raw staleness value
                - staleness_normalized: Normalized staleness (if enabled)
                - final_score: Final combined score
        """
        # === Compute base metric ===
        if self.priority_metric == 'reward':
            # Average reward across samples in group
            rewards = [s.reward for s in group if s.reward is not None]
            base_score_raw = np.mean(rewards) if rewards else 0.0

        elif self.priority_metric == 'advantage':
            # Average advantage (if available in metadata)
            advantages = []
            for s in group:
                if s.metadata and 'advantage' in s.metadata:
                    advantages.append(s.metadata['advantage'])
            base_score_raw = np.mean(advantages) if advantages else 0.0

        elif self.priority_metric == 'custom':
            # User-defined function
            base_score_raw = self.priority_func(group)

        else:
            raise ValueError(f"Unknown priority metric: {self.priority_metric}")

        # === Compute staleness ===
        sample = group[0]
        if sample.policy_version is not None:
            staleness_raw = self.current_policy_version - sample.policy_version
        else:
            staleness_raw = 0.0

        # === Update running statistics ===
        self._update_stats(base_score_raw, self.base_score_stats)
        self._update_stats(staleness_raw, self.staleness_stats)

        # === Apply normalization (if enabled) ===
        if self.normalize_scores:
            base_score_norm = self._normalize_value(
                base_score_raw,
                self.base_score_stats,
                self.normalization_method
            )
            staleness_norm = self._normalize_value(
                staleness_raw,
                self.staleness_stats,
                self.normalization_method
            )

            # Compute final score with normalized values
            # Score = base_score_norm × priority_weight - staleness_norm × staleness_penalty
            final_score = (base_score_norm * self.priority_weight -
                          staleness_norm * self.staleness_penalty)
        else:
            # Use raw scores (backward compatible)
            base_score_norm = base_score_raw
            staleness_norm = staleness_raw
            final_score = (base_score_raw * self.priority_weight -
                          staleness_raw * self.staleness_penalty)

        # === Prepare breakdown for monitoring ===
        breakdown = {
            'base_score_raw': base_score_raw,
            'base_score_normalized': base_score_norm,
            'staleness_raw': staleness_raw,
            'staleness_normalized': staleness_norm,
            'final_score': final_score,
            'priority_weight': self.priority_weight,
            'staleness_penalty': self.staleness_penalty,
        }

        return final_score, breakdown

    def sample(
        self,
        buffer: List[List[Sample]],
        num_samples: int
    ) -> List[List[Sample]]:
        """
        Sample using priority-based strategy.

        Algorithm:
        1. Filter by staleness
        2. Filter by reuse count AND identify exhausted groups
        3. Compute priority scores
        4. Sort by score (high → low)
        5. Take top num_samples groups
        6. Remove from buffer if remove_on_sample=True
        7. Remove stale and exhausted groups
        """
        if len(buffer) == 0 or num_samples == 0:
            return []

        # Step 1: Filter by staleness
        valid_groups, stale_groups = self.filter_by_staleness(buffer)

        # Step 2: Filter by reuse count AND identify exhausted groups immediately
        reusable_groups = []
        exhausted_groups = []

        if self.max_reuse_count > 0:
            for group in valid_groups:
                sample = group[0]
                reuse_count = sample.metadata.get('buffer_reuse_count', 0) if sample.metadata else 0

                if reuse_count < self.max_reuse_count:
                    reusable_groups.append(group)
                else:
                    # 🔧 FIX: Identify exhausted groups immediately for removal
                    exhausted_groups.append(group)
        else:
            reusable_groups = valid_groups

        # Step 3: Compute priorities
        scored_groups = []
        breakdowns = []  # Collect breakdowns for statistics

        for group in reusable_groups:
            try:
                score, breakdown = self.compute_priority(group)
                scored_groups.append((group, score))
                breakdowns.append(breakdown)
            except Exception as e:
                print(f"[Warning] Failed to compute priority for group: {e}")
                # Assign default score
                scored_groups.append((group, 0.0))
                breakdowns.append({'final_score': 0.0, 'base_score_raw': 0.0})

        # Step 4: Sort by priority (high → low)
        scored_groups.sort(key=lambda x: x[1], reverse=True)

        # Step 5: Sample top num_samples
        num_to_sample = min(len(scored_groups), num_samples)
        sampled = [group for group, score in scored_groups[:num_to_sample]]
        sampled_scores = [score for group, score in scored_groups[:num_to_sample]]
        sampled_breakdowns = breakdowns[:num_to_sample]

        # Step 6: Store statistics for wandb logging
        if len(sampled_breakdowns) > 0:
            # Aggregate statistics across sampled groups
            stats = {
                'base_score_raw_mean': np.mean([b['base_score_raw'] for b in sampled_breakdowns]),
                'base_score_raw_std': np.std([b['base_score_raw'] for b in sampled_breakdowns]),
                'base_score_normalized_mean': np.mean([b.get('base_score_normalized', 0) for b in sampled_breakdowns]),
                'staleness_raw_mean': np.mean([b['staleness_raw'] for b in sampled_breakdowns]),
                'staleness_raw_std': np.std([b['staleness_raw'] for b in sampled_breakdowns]),
                'staleness_normalized_mean': np.mean([b.get('staleness_normalized', 0) for b in sampled_breakdowns]),
                'final_score_mean': np.mean(sampled_scores),
                'final_score_std': np.std(sampled_scores),
                'final_score_min': np.min(sampled_scores),
                'final_score_max': np.max(sampled_scores),
            }
            self.priority_scores_history.append(stats)

            # Keep only recent history (last 100 samples)
            if len(self.priority_scores_history) > 100:
                self.priority_scores_history = self.priority_scores_history[-100:]

        # Step 7: Update reuse count (if not removing)
        if not self.remove_on_sample and sampled:
            self.increment_reuse_count(sampled)

        # Step 8: Remove from buffer
        if self.remove_on_sample and sampled:
            self.remove_from_buffer(buffer, sampled)

        # Always remove stale groups
        if stale_groups:
            print(f"[Buffer Sampling] Removing {len(stale_groups)} stale groups "
                  f"(staleness > {self.max_staleness})")
            self.remove_from_buffer(buffer, stale_groups)

        # 🔧 FIX: Remove exhausted groups immediately
        if exhausted_groups:
            print(f"[Buffer Sampling] Removing {len(exhausted_groups)} exhausted groups "
                  f"(reuse_count >= {self.max_reuse_count})")
            self.remove_from_buffer(buffer, exhausted_groups)

        self.total_sampled += len(sampled)
        return sampled
    def get_statistics(self) -> dict:
        """Return sampling statistics including priority-specific metrics."""
        base_stats = super().get_statistics()

        # Add priority-specific stats
        priority_stats = {
            "priority_metric": self.priority_metric,
            "priority_weight": self.priority_weight,
            "staleness_penalty": self.staleness_penalty,
            "normalize_scores": self.normalize_scores,
            "normalization_method": self.normalization_method if self.normalize_scores else None,
        }

        # Add running statistics
        if self.base_score_stats['count'] > 0:
            priority_stats.update({
                "base_score_min": self.base_score_stats['min'],
                "base_score_max": self.base_score_stats['max'],
                "base_score_mean": self.base_score_stats['sum'] / self.base_score_stats['count'],
            })

        if self.staleness_stats['count'] > 0:
            priority_stats.update({
                "staleness_min": self.staleness_stats['min'],
                "staleness_max": self.staleness_stats['max'],
                "staleness_mean": self.staleness_stats['sum'] / self.staleness_stats['count'],
            })

        # Add recent sampling statistics
        if len(self.priority_scores_history) > 0:
            latest = self.priority_scores_history[-1]
            priority_stats.update({
                "latest_" + k: v for k, v in latest.items()
            })

        base_stats.update(priority_stats)
        return base_stats


class RandomSamplingStrategy(BaseSamplingStrategy):
    """
    Random sampling with staleness filtering.

    Randomly samples from valid groups. Reduces sampling bias and
    increases data diversity compared to FIFO.

    Recommended for:
    - When you want maximum data diversity
    - Avoiding potential correlations in FIFO order
    - Experimental setups comparing sampling strategies
    """

    def __init__(self, args, current_policy_version: int):
        super().__init__(args, current_policy_version)
        self.seed = getattr(args, 'buffer_random_seed', None)
        if self.seed is not None:
            np.random.seed(self.seed)

    def get_name(self) -> str:
        return "random"

    def sample(
        self,
        buffer: List[List[Sample]],
        num_samples: int
    ) -> List[List[Sample]]:
        """
        Sample randomly with staleness filtering.

        Algorithm:
        1. Filter by staleness
        2. Filter by reuse count AND identify exhausted groups
        3. Random shuffle
        4. Take first num_samples groups
        5. Remove from buffer if remove_on_sample=True
        6. Remove stale and exhausted groups
        """
        if len(buffer) == 0 or num_samples == 0:
            return []

        # Step 1: Filter by staleness
        valid_groups, stale_groups = self.filter_by_staleness(buffer)

        # Step 2: Filter by reuse count AND identify exhausted groups immediately
        reusable_groups = []
        exhausted_groups = []

        if self.max_reuse_count > 0:
            for group in valid_groups:
                sample = group[0]
                reuse_count = sample.metadata.get('buffer_reuse_count', 0) if sample.metadata else 0

                if reuse_count < self.max_reuse_count:
                    reusable_groups.append(group)
                else:
                    # 🔧 FIX: Identify exhausted groups immediately for removal
                    exhausted_groups.append(group)
        else:
            reusable_groups = valid_groups

        # Step 3: Random sampling
        num_to_sample = min(len(reusable_groups), num_samples)

        if num_to_sample > 0:
            # Use numpy for reproducibility
            indices = np.random.choice(len(reusable_groups), num_to_sample, replace=False)
            sampled = [reusable_groups[i] for i in indices]
        else:
            sampled = []

        # Step 4: Update reuse count (if not removing)
        if not self.remove_on_sample and sampled:
            self.increment_reuse_count(sampled)

        # Step 5: Remove from buffer
        if self.remove_on_sample and sampled:
            self.remove_from_buffer(buffer, sampled)

        # Always remove stale groups
        if stale_groups:
            print(f"[Buffer Sampling] Removing {len(stale_groups)} stale groups "
                  f"(staleness > {self.max_staleness})")
            self.remove_from_buffer(buffer, stale_groups)

        # 🔧 FIX: Remove exhausted groups immediately
        if exhausted_groups:
            print(f"[Buffer Sampling] Removing {len(exhausted_groups)} exhausted groups "
                  f"(reuse_count >= {self.max_reuse_count})")
            self.remove_from_buffer(buffer, exhausted_groups)

        self.total_sampled += len(sampled)
        return sampled


class ReservoirSamplingStrategy(BaseSamplingStrategy):
    """
    Reservoir sampling for uniform random sampling over all buffer history.

    Maintains a uniform distribution over all seen samples, regardless of when
    they were added. Useful for maintaining true i.i.d. sampling.

    Recommended for:
    - Theoretical experiments requiring i.i.d. samples
    - When buffer size >> sampling frequency
    """

    def get_name(self) -> str:
        return "reservoir"

    def sample(
        self,
        buffer: List[List[Sample]],
        num_samples: int
    ) -> List[List[Sample]]:
        """
        Reservoir sampling implementation.

        This is equivalent to random sampling but with the property that
        each sample has equal probability regardless of insertion order.
        """
        if len(buffer) == 0 or num_samples == 0:
            return []

        # Filter by staleness
        valid_groups, stale_groups = self.filter_by_staleness(buffer)

        # Filter by reuse count
        valid_groups = self.filter_by_reuse_count(valid_groups)

        # Reservoir sampling
        num_to_sample = min(len(valid_groups), num_samples)

        reservoir = []
        for i, group in enumerate(valid_groups):
            if i < num_to_sample:
                reservoir.append(group)
            else:
                # Replace with decreasing probability
                j = np.random.randint(0, i + 1)
                if j < num_to_sample:
                    reservoir[j] = group

        sampled = reservoir

        # Update reuse count (if not removing)
        if not self.remove_on_sample:
            self.increment_reuse_count(sampled)

        # Remove from buffer
        if self.remove_on_sample:
            self.remove_from_buffer(buffer, sampled)

        # Remove stale groups
        if stale_groups:
            self.remove_from_buffer(buffer, stale_groups)

        self.total_sampled += len(sampled)
        return sampled


# ============================================================================
# Strategy Factory
# ============================================================================

def get_sampling_strategy(
    args,
    current_policy_version: int
) -> BaseSamplingStrategy:
    """
    Factory function to create sampling strategy.

    Args:
        args: Configuration namespace
        current_policy_version: Current policy version

    Returns:
        Sampling strategy instance

    Raises:
        ValueError: If unknown strategy type
    """
    strategy_type = getattr(args, 'buffer_sampling_strategy', 'fifo_staleness')

    # Map strategy names to classes
    STRATEGIES = {
        'fifo': FIFOWithStalenessStrategy,  # Legacy name
        'fifo_staleness': FIFOWithStalenessStrategy,
        'lifo': LIFOWithStalenessStrategy,  # NEW: Newest-first sampling
        'lifo_staleness': LIFOWithStalenessStrategy,  # Alias
        'priority': PrioritySamplingStrategy,
        'random': RandomSamplingStrategy,
        'reservoir': ReservoirSamplingStrategy,
    }

    # Check for custom strategy
    if strategy_type == 'custom':
        from slime.utils.misc import load_function
        custom_path = getattr(args, 'buffer_sampling_custom_path', None)
        if custom_path is None:
            raise ValueError("buffer_sampling_custom_path required for custom strategy")

        # Load custom strategy class
        CustomStrategy = load_function(custom_path)
        return CustomStrategy(args, current_policy_version)

    # Standard strategies
    if strategy_type not in STRATEGIES:
        available = ', '.join(STRATEGIES.keys())
        raise ValueError(
            f"Unknown buffer sampling strategy: {strategy_type}. "
            f"Available strategies: {available}, custom"
        )

    StrategyClass = STRATEGIES[strategy_type]
    return StrategyClass(args, current_policy_version)


# ============================================================================
# Backward Compatibility: Legacy pop_first function
# ============================================================================

def pop_first(args, current_policy_version, buffer: List[List[Sample]], num_samples: int) -> List[List[Sample]]:
    """
    Legacy pop_first function for backward compatibility.

    This is the original simple FIFO without staleness filtering.
    Deprecated: Use FIFOWithStalenessStrategy instead.
    """
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
