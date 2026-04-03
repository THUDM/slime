"""Utilities for managing checkpoint retention during RL training."""

import logging
import os
import shutil
from typing import Callable, List

logger = logging.getLogger(__name__)


def should_run_cleanup(storage_type: str, global_rank: int, local_rank: int) -> tuple:
    """Determine which ranks should run checkpoint cleanup.

    Megatron: 'shared' → global rank 0; 'local' → local rank 0 per node.
    HF: global rank 0 only (save_hf_pretrained always writes from global rank 0).

    Returns:
        (should_cleanup_megatron, should_cleanup_hf)
    """
    if storage_type == "local":
        should_megatron = local_rank == 0
    else:
        should_megatron = global_rank == 0
    should_hf = global_rank == 0
    return should_megatron, should_hf


def cleanup_old_checkpoints(
    saved_rollout_ids: List[int],
    keep: int,
    path_fn: Callable[[int], str],
) -> List[str]:
    """Delete the oldest checkpoints, keeping only the newest *keep*.

    *saved_rollout_ids* is an ordered list of rollout ids saved during the
    current run (oldest first).  *path_fn* maps a rollout id to its checkpoint
    directory path on disk.

    Returns:
        List of deleted directory paths.
    """
    if len(saved_rollout_ids) <= keep:
        return []

    to_delete_ids = saved_rollout_ids[: len(saved_rollout_ids) - keep]

    deleted: list[str] = []
    for rid in to_delete_ids:
        path = path_fn(rid)
        if os.path.isdir(path):
            try:
                logger.info("Deleting old checkpoint: %s", path)
                shutil.rmtree(path)
                deleted.append(path)
            except OSError:
                logger.warning("Failed to delete checkpoint %s, will retry next save", path, exc_info=True)
    return deleted
