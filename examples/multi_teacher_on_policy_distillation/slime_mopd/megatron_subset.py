"""Per-teacher subset construction for Megatron MOPD scoring.

Rather than re-packing micro-batches, filter slime's existing
``micro_batch_indices`` down to the samples routed to one teacher and remap the
surviving indices into a compact teacher-local array. This keeps slime's
``forward_only`` ordering-restoration contract valid (flattened micro-batch
indices must tile ``range(subset_size)``) and handles micro-batches that mix
samples from different teachers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_SCORER_FIELDS = (
    "tokens",
    "loss_masks",
    "multimodal_train_inputs",
    "total_lengths",
    "response_lengths",
    "rollout_log_probs",
    "rollout_topk_token_ids",
    "rollout_topk_log_probs",
    "rollout_top_p_token_ids",
    "rollout_top_p_token_offsets",
    "rollout_top_p_log_probs",
    "rollout_mask_sums",
)


def build_teacher_subset(
    rollout_data: dict[str, Any],
    teacher_ids: Sequence[str],
    teacher_id: str,
) -> tuple[dict[str, Any] | None, list[int]]:
    """Build a compact ``forward_only`` view for one teacher.

    Returns ``(subset, selected_indices)`` where ``selected_indices`` maps the
    subset's compact positions back to original rank-local positions. Returns
    ``(None, [])`` when the teacher has no samples in this batch.
    """

    sample_count = len(teacher_ids)
    selected = [position for position, routed in enumerate(teacher_ids) if routed == teacher_id]
    if not selected:
        return None, []

    old_to_new = {old: new for new, old in enumerate(selected)}
    compact: list[list[int]] = []
    for microbatch in rollout_data["micro_batch_indices"]:
        remapped = [old_to_new[index] for index in microbatch if index in old_to_new]
        if remapped:
            compact.append(remapped)

    flattened = sorted(index for microbatch in compact for index in microbatch)
    if flattened != list(range(len(selected))):
        raise ValueError(
            f"mopd: teacher {teacher_id!r} subset is not tiled exactly once by micro-batches; "
            f"got {len(flattened)} placements for {len(selected)} samples"
        )

    subset: dict[str, Any] = {"micro_batch_indices": compact}
    for key in _SCORER_FIELDS:
        values = rollout_data.get(key)
        if values is None:
            continue
        if len(values) != sample_count:
            raise ValueError(f"mopd: scorer field {key!r} has {len(values)} values; expected {sample_count}")
        subset[key] = [values[index] for index in selected]
    return subset, selected


def scatter_subset_into(full: list[Any], indices: Sequence[int], values: Sequence[Any]) -> None:
    """Write subset results back to their original local positions."""

    if len(indices) != len(values):
        raise ValueError(f"mopd: scatter size mismatch: {len(indices)} indices vs {len(values)} values")
    for position, index in enumerate(indices):
        full[index] = values[position]
