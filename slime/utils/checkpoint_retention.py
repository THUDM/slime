"""Safe retention for completed Megatron checkpoint directories."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


_ITERATION_DIRECTORY = re.compile(r"iter_(\d{7})\Z")


def prune_megatron_checkpoints(root: str | Path, retain_count: int) -> list[Path]:
    """Remove oldest completed checkpoints while preserving recent history.

    Megatron updates ``latest_checkpointed_iteration.txt`` only after a
    checkpoint is complete. Directories newer than that marker may still be
    in flight and are never considered for deletion.
    """
    if retain_count < 1:
        raise ValueError("retain_count must be at least 1")

    root = Path(root)
    marker = root / "latest_checkpointed_iteration.txt"
    try:
        latest_text = marker.read_text().strip()
    except FileNotFoundError:
        return []
    if not latest_text.isdigit():
        return []
    latest_iteration = int(latest_text)

    completed = []
    for path in root.iterdir():
        match = _ITERATION_DIRECTORY.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            continue
        iteration = int(match.group(1))
        if iteration <= latest_iteration:
            completed.append((iteration, path))
    completed.sort()

    removed = []
    for _, path in completed[:-retain_count]:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            continue
        removed.append(path)
    return removed
