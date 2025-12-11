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
        Sample using FIFO with staleness filtering.

        Algorithm:
        1. Filter by staleness
        2. Filter by reuse count
        3. Take first num_samples groups (FIFO)
        4. Remove from buffer if remove_on_sample=True
        """
        if len(buffer) == 0 or num_samples == 0:
            return []

        # Step 1: Filter by staleness
        valid_groups, stale_groups = self.filter_by_staleness(buffer)

        # Step 2: Filter by reuse count
        valid_groups = self.filter_by_reuse_count(valid_groups)

        # Step 3: FIFO sampling
        num_to_sample = min(len(valid_groups), num_samples)
        sampled = valid_groups[:num_to_sample]

        # Step 4: Update reuse count (if not removing)
        if not self.remove_on_sample:
            self.increment_reuse_count(sampled)

        # Step 5: Remove from buffer
        if self.remove_on_sample:
            # Remove sampled groups
            self.remove_from_buffer(buffer, sampled)

        # Always remove stale groups
        if stale_groups:
            print(f"[Buffer Sampling] Removing {len(stale_groups)} stale groups "
                  f"(staleness > {self.max_staleness})")
            self.remove_from_buffer(buffer, stale_groups)

        self.total_sampled += len(sampled)
        return sampled


class PrioritySamplingStrategy(BaseSamplingStrategy):
    """
    Priority-based sampling with configurable metrics.

    Samples groups with higher priority first. Priority can be based on:
    - Reward: Higher reward → Higher priority
    - Advantage: Higher advantage → Higher priority (requires pre-computation)
    - Custom: User-defined metric

    Priority is adjusted by staleness penalty to balance data freshness.

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

    def get_name(self) -> str:
        return f"priority_{self.priority_metric}"

    def compute_priority(self, group: List[Sample]) -> float:
        """
        Compute priority score for a group.

        Score = base_metric × priority_weight - staleness × staleness_penalty
        """
        # Compute base metric
        if self.priority_metric == 'reward':
            # Average reward across samples in group
            rewards = [s.reward for s in group if s.reward is not None]
            base_score = np.mean(rewards) if rewards else 0.0

        elif self.priority_metric == 'advantage':
            # Average advantage (if available in metadata)
            advantages = []
            for s in group:
                if s.metadata and 'advantage' in s.metadata:
                    advantages.append(s.metadata['advantage'])
            base_score = np.mean(advantages) if advantages else 0.0

        elif self.priority_metric == 'custom':
            # User-defined function
            base_score = self.priority_func(group)

        else:
            raise ValueError(f"Unknown priority metric: {self.priority_metric}")

        # Compute staleness penalty
        sample = group[0]
        if sample.policy_version is not None:
            staleness = self.current_policy_version - sample.policy_version
            staleness_term = staleness * self.staleness_penalty
        else:
            staleness_term = 0.0

        # Final score
        score = base_score * self.priority_weight - staleness_term
        return score

    def sample(
        self,
        buffer: List[List[Sample]],
        num_samples: int
    ) -> List[List[Sample]]:
        """
        Sample using priority-based strategy.

        Algorithm:
        1. Filter by staleness
        2. Filter by reuse count
        3. Compute priority scores
        4. Sort by score (high → low)
        5. Take top num_samples groups
        6. Remove from buffer if remove_on_sample=True
        """
        if len(buffer) == 0 or num_samples == 0:
            return []

        # Step 1: Filter by staleness
        valid_groups, stale_groups = self.filter_by_staleness(buffer)

        # Step 2: Filter by reuse count
        valid_groups = self.filter_by_reuse_count(valid_groups)

        # Step 3: Compute priorities
        scored_groups = []
        for group in valid_groups:
            try:
                score = self.compute_priority(group)
                scored_groups.append((group, score))
            except Exception as e:
                print(f"[Warning] Failed to compute priority for group: {e}")
                # Assign default score
                scored_groups.append((group, 0.0))

        # Step 4: Sort by priority (high → low)
        scored_groups.sort(key=lambda x: x[1], reverse=True)

        # Step 5: Sample top num_samples
        num_to_sample = min(len(scored_groups), num_samples)
        sampled = [group for group, score in scored_groups[:num_to_sample]]

        # Step 6: Update reuse count (if not removing)
        if not self.remove_on_sample:
            self.increment_reuse_count(sampled)

        # Step 7: Remove from buffer
        if self.remove_on_sample:
            self.remove_from_buffer(buffer, sampled)

        # Remove stale groups
        if stale_groups:
            print(f"[Buffer Sampling] Removing {len(stale_groups)} stale groups")
            self.remove_from_buffer(buffer, stale_groups)

        self.total_sampled += len(sampled)
        return sampled


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
        2. Filter by reuse count
        3. Random shuffle
        4. Take first num_samples groups
        5. Remove from buffer if remove_on_sample=True
        """
        if len(buffer) == 0 or num_samples == 0:
            return []

        # Step 1: Filter by staleness
        valid_groups, stale_groups = self.filter_by_staleness(buffer)

        # Step 2: Filter by reuse count
        valid_groups = self.filter_by_reuse_count(valid_groups)

        # Step 3: Random sampling
        num_to_sample = min(len(valid_groups), num_samples)

        # Use numpy for reproducibility
        indices = np.random.choice(len(valid_groups), num_to_sample, replace=False)
        sampled = [valid_groups[i] for i in indices]

        # Step 4: Update reuse count (if not removing)
        if not self.remove_on_sample:
            self.increment_reuse_count(sampled)

        # Step 5: Remove from buffer
        if self.remove_on_sample:
            self.remove_from_buffer(buffer, sampled)

        # Remove stale groups
        if stale_groups:
            print(f"[Buffer Sampling] Removing {len(stale_groups)} stale groups")
            self.remove_from_buffer(buffer, stale_groups)

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
