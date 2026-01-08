"""
Experience Replay Buffer for GSPO training.

Injects successful trajectories when all online rollouts fail, preventing
gradient vanishing in sparse reward environments.

Reference: DART-GUI-7B showed +15% success with replay vs without.
"""

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Singleton instance
_replay_buffer: "ExperienceReplayBuffer | None" = None


@dataclass
class ReplaySample:
    """A successful trajectory sample for replay injection."""

    task_id: str
    conversations: list[dict[str, Any]]
    images: list[str]
    reward: float
    metadata: dict[str, Any]

    def to_sample_dict(self) -> dict[str, Any]:
        """Convert to Sample-compatible dict for GSPO training."""
        return {
            "conversations": self.conversations,
            "images": self.images,
            "reward": self.reward,
            "metadata": {
                **self.metadata,
                "task_id": self.task_id,
                "is_replay": True,
                "osworld": {
                    "task_reward": self.reward,
                    "step_signals": [],
                    "is_replay": True,
                },
            },
        }


class ExperienceReplayBuffer:
    """
    Pre-loads successful trajectories for replay injection.

    When all online rollouts for a task fail, inject a successful trajectory
    from the replay buffer to provide positive learning signal.
    """

    def __init__(
        self,
        replay_data_path: str | Path,
        success_threshold: float = 0.5,
        max_samples_per_task: int = 10,
    ):
        """
        Initialize replay buffer from JSONL file.

        Args:
            replay_data_path: Path to merged replay JSONL file
            success_threshold: Minimum reward to consider a trajectory successful
            max_samples_per_task: Maximum samples to keep per task
        """
        self.pool: dict[str, list[ReplaySample]] = {}
        # Index from base UUID to chunked task IDs for fuzzy matching
        self.base_to_chunks: dict[str, list[str]] = {}
        self.success_threshold = success_threshold
        self.max_samples_per_task = max_samples_per_task
        self.injection_count = 0
        self.total_checks = 0

        self._load_from_jsonl(Path(replay_data_path))

    def _load_from_jsonl(self, path: Path) -> None:
        """Load successful trajectories from JSONL."""
        if not path.exists():
            logger.warning(f"Replay buffer path not found: {path}")
            return

        loaded = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue

                sample = json.loads(line)
                task_id = self._extract_task_id(sample)
                if not task_id:
                    continue

                # SFT data is assumed successful (reward=1.0)
                reward = sample.get("reward", 1.0)
                if reward < self.success_threshold:
                    continue

                # Handle both 'messages' (merged replay format) and 'conversations'
                conversations = sample.get("messages") or sample.get("conversations", [])
                images = sample.get("images") or []
                replay_sample = ReplaySample(
                    task_id=task_id,
                    conversations=conversations,
                    images=images,
                    reward=reward,
                    metadata=sample.get("metadata", {}),
                )

                if task_id not in self.pool:
                    self.pool[task_id] = []

                if len(self.pool[task_id]) < self.max_samples_per_task:
                    self.pool[task_id].append(replay_sample)
                    loaded += 1

        # Build base UUID index for fuzzy matching
        for task_id in self.pool.keys():
            base = self._get_base_uuid(task_id)
            if base not in self.base_to_chunks:
                self.base_to_chunks[base] = []
            self.base_to_chunks[base].append(task_id)

        logger.info(
            f"Loaded {loaded} replay samples for {len(self.pool)} tasks "
            f"(threshold={self.success_threshold}, base_uuids={len(self.base_to_chunks)})"
        )

    def _get_base_uuid(self, task_id: str) -> str:
        """Extract base UUID by stripping _chunk* suffix if present."""
        if "_chunk" in task_id:
            return task_id.rsplit("_chunk", 1)[0]
        return task_id

    def _extract_task_id(self, sample: dict) -> str | None:
        """Extract task_id from sample."""
        if "metadata" in sample and "task_id" in sample["metadata"]:
            return sample["metadata"]["task_id"]
        return sample.get("id") or sample.get("task_id")

    def maybe_inject(
        self,
        task_id: str,
        group_rewards: list[float],
    ) -> ReplaySample | None:
        """
        Return replay sample if all online rollouts failed.

        Uses fuzzy matching: tries exact task_id first, then matches by
        base UUID (stripping _chunk* suffix) to handle chunked trajectories.

        Args:
            task_id: The task identifier
            group_rewards: Rewards from online rollouts for this task

        Returns:
            ReplaySample if injection needed and available, else None
        """
        self.total_checks += 1

        # Only inject if ALL rollouts failed
        if any(r >= self.success_threshold for r in group_rewards):
            return None

        # Try exact match first
        matched_task_id = None
        if task_id in self.pool and self.pool[task_id]:
            matched_task_id = task_id
        else:
            # Fuzzy match: try base UUID to find chunked variants
            base = self._get_base_uuid(task_id)
            if base in self.base_to_chunks:
                # Pick a random chunk that has replay data
                candidates = [t for t in self.base_to_chunks[base] if t in self.pool and self.pool[t]]
                if candidates:
                    matched_task_id = random.choice(candidates)
                    logger.debug(f"Fuzzy matched {task_id} -> {matched_task_id}")

        if matched_task_id is None:
            logger.debug(f"No replay data for task {task_id} (base={self._get_base_uuid(task_id)})")
            return None

        replay = random.choice(self.pool[matched_task_id])
        self.injection_count += 1

        logger.info(
            f"Injecting replay for task {task_id} (matched={matched_task_id}) "
            f"(online rewards: {group_rewards}, replay reward: {replay.reward})"
        )

        return replay

    def update(self, task_id: str, conversations: list[dict], reward: float, images: list[str] | None = None) -> None:
        """
        Add new successful trajectory to pool.

        Called when online rollout succeeds - grows the replay pool over time.
        """
        if reward < self.success_threshold:
            return

        if task_id not in self.pool:
            self.pool[task_id] = []

        if len(self.pool[task_id]) < self.max_samples_per_task:
            sample = ReplaySample(
                task_id=task_id,
                conversations=conversations,
                images=images or [],
                reward=reward,
                metadata={"source": "online"},
            )
            self.pool[task_id].append(sample)
            logger.info(f"Added online success to replay pool: {task_id}")

    def get_stats(self) -> dict[str, Any]:
        """Return buffer statistics."""
        return {
            "total_tasks": len(self.pool),
            "total_samples": sum(len(v) for v in self.pool.values()),
            "injection_count": self.injection_count,
            "total_checks": self.total_checks,
            "injection_rate": self.injection_count / max(1, self.total_checks),
        }

    def has_task(self, task_id: str) -> bool:
        """Check if task has replay data."""
        return task_id in self.pool and len(self.pool[task_id]) > 0

    def get_coverage(self, task_ids: list[str]) -> float:
        """Calculate replay coverage for a set of tasks."""
        if not task_ids:
            return 0.0
        covered = sum(1 for tid in task_ids if self.has_task(tid))
        return covered / len(task_ids)


def get_replay_buffer() -> ExperienceReplayBuffer | None:
    """
    Get singleton replay buffer instance.

    Initialized lazily from OSWORLD_REPLAY_BUFFER env var.
    """
    global _replay_buffer

    if _replay_buffer is not None:
        return _replay_buffer

    replay_path = os.environ.get("OSWORLD_REPLAY_BUFFER")
    if not replay_path:
        logger.info("OSWORLD_REPLAY_BUFFER not set, replay disabled")
        return None

    threshold = float(os.environ.get("OSWORLD_REPLAY_THRESHOLD", "0.5"))

    logger.info(f"Initializing replay buffer from {replay_path}")
    _replay_buffer = ExperienceReplayBuffer(
        replay_data_path=replay_path,
        success_threshold=threshold,
    )

    return _replay_buffer


def reset_replay_buffer() -> None:
    """Reset singleton (for testing)."""
    global _replay_buffer
    _replay_buffer = None
