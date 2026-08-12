import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


_CONTEXT_PARALLEL_FIELDS = (
    "rollout_log_probs",
    "teacher_log_probs",
    "log_probs",
    "ref_log_probs",
    "values",
    "advantages",
    "returns",
    "kl",
    "entropy",
    "opd_reverse_kl",
)


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


@torch.no_grad()
def restore_context_parallel_fields_to_cpu(
    rollout_data: dict[str, Any],
    gather_tensor: Callable[[torch.Tensor, int, int], torch.Tensor],
    *,
    keep_restored: bool,
) -> dict[str, Any] | None:
    """Restore CP fields one tensor at a time, retaining CPU values only on the writer."""
    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]
    num_samples = len(response_lengths)
    if len(total_lengths) != num_samples:
        raise ValueError(
            "total_lengths and response_lengths must contain the same number of samples, "
            f"got {len(total_lengths)} and {num_samples}."
        )

    restored = (
        {key: _to_cpu(value) for key, value in rollout_data.items() if key not in _CONTEXT_PARALLEL_FIELDS}
        if keep_restored
        else None
    )
    for key in _CONTEXT_PARALLEL_FIELDS:
        values = rollout_data.get(key)
        if values is None:
            continue
        if not isinstance(values, (list, tuple)) or len(values) != num_samples:
            value_count = len(values) if isinstance(values, (list, tuple)) else "not a list or tuple"
            raise ValueError(
                f"CP-sharded field {key!r} must contain one tensor per sample, "
                f"got {value_count}; expected {num_samples}."
            )

        full_values = [] if keep_restored else None
        for sample_id, (value, total_length, response_length) in enumerate(
            zip(values, total_lengths, response_lengths, strict=True)
        ):
            if not torch.is_tensor(value):
                raise TypeError(
                    f"CP-sharded field {key!r} sample {sample_id} must be a tensor, got {type(value).__name__}."
                )
            full_value = gather_tensor(value, int(total_length), int(response_length)).detach()
            if full_value.size(0) != response_length:
                raise ValueError(
                    f"Restored field {key!r} sample {sample_id} has length {full_value.size(0)}, "
                    f"expected {response_length}."
                )
            if keep_restored:
                assert full_values is not None
                full_values.append(full_value.detach().cpu())
            # Drop the full GPU tensor before the next collective allocates its
            # output, keeping peak extra device memory to one response tensor.
            del full_value
        if keep_restored:
            assert restored is not None
            restored[key] = full_values

    return restored


# Per-rank scheduling arrangement (not per-sample); stored under `dp_shards` so
# the flat `samples` view can stay aligned with the rollout debug dump.
_LAYOUT_FIELDS = ("micro_batch_indices", "num_microbatches", "global_batch_sizes")
# Whole-batch fields that are identical on every DP shard; stored once at the top.
_WHOLE_BATCH_FIELDS = ("raw_reward",)


def _is_per_sample(value, num_samples: int) -> bool:
    if torch.is_tensor(value):
        return value.dim() >= 1 and value.size(0) == num_samples
    if isinstance(value, (list, tuple)):
        return len(value) == num_samples
    return False


def _build_dump_payload(dp_shards, *, rollout_id, writer_rank):
    """Turn the gathered per-rank shards into a rollout-dump-aligned payload.

    Mirrors the rollout debug dump: a ``samples`` list (one dict per training
    sample, sorted so it lines up with the rollout dump's ``samples``), plus a
    parallel ``dp_shards`` key that keeps the DP/mbs layout (``sample_indices``,
    ``rollout_positions`` + ``micro_batch_indices`` / ``num_microbatches`` /
    ``global_batch_sizes``) without duplicating any per-sample tensor.

    Ordering key preference: ``rollout_positions`` (each sample's index in the
    rollout debug dump, always available when dumping) restores exact rollout
    order regardless of ``sample.index``; ``sample_indices`` (``sample.index``,
    may be ``None``) is the fallback; failing both, DP-gather order is kept.
    """
    samples = []
    layout = []
    whole_batch = {}
    # Per-sample id columns handled specially: pulled out of the generic
    # transpose and surfaced as scalar keys on each sample dict.
    id_columns = {"sample_indices": "sample_index", "rollout_positions": "rollout_position"}
    for shard in dp_shards:
        rollout_data = shard["rollout_data"]
        dp_rank = shard["data_parallel_rank"]
        num_local = len(rollout_data["response_lengths"])

        shard_layout = {"rank": shard["rank"], "data_parallel_rank": dp_rank}
        for column in id_columns:
            value = rollout_data.get(column)
            shard_layout[column] = list(value) if value is not None else None
        per_sample_columns = {}
        for key, value in rollout_data.items():
            if key in id_columns:
                continue
            if key in _WHOLE_BATCH_FIELDS:
                whole_batch.setdefault(key, value)
            elif key in _LAYOUT_FIELDS or not _is_per_sample(value, num_local):
                # Scheduling / non-per-sample fields stay in the layout so the
                # flat sample view holds only what pairs with a single sample.
                shard_layout[key] = value
            else:
                per_sample_columns[key] = value

        for i in range(num_local):
            sample = {"data_parallel_rank": dp_rank}
            for column, scalar_key in id_columns.items():
                values = rollout_data.get(column)
                if values is not None:
                    sample[scalar_key] = values[i]
            for key, column in per_sample_columns.items():
                sample[key] = column[i]
            samples.append(sample)
        layout.append(shard_layout)

    order_key = next(
        (
            scalar_key
            for scalar_key in ("rollout_position", "sample_index")
            if samples and all(sample.get(scalar_key) is not None for sample in samples)
        ),
        None,
    )
    if order_key is not None:
        samples.sort(key=lambda sample: sample[order_key])
    else:
        logger.warning(
            "Cannot restore global sample order for the train debug dump: samples carry neither "
            "rollout_positions nor sample_index. Saving samples in DP-gather order instead."
        )

    return {
        "format_version": 2,
        "rollout_id": rollout_id,
        "rank": writer_rank,
        "samples": samples,
        "dp_shards": layout,
        **whole_batch,
    }


def save_debug_train_data(args, *, rollout_id, rollout_data):
    path_template = args.save_debug_train_data
    if path_template is None:
        return

    from megatron.core import mpu

    # The last PP stage owns the computed train fields. TP ranks hold the same
    # token values after TP reduction, so only TP0 needs to participate below.
    if not mpu.is_pipeline_last_stage(ignore_virtual=True) or mpu.get_tensor_model_parallel_rank() != 0:
        return

    from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

    # All CP ranks in the selected PP/TP group must enter these collectives.
    # CP=1 follows the same normalization path but all_gather_with_cp is an identity.
    cp_rank = mpu.get_context_parallel_rank()
    rollout_data = restore_context_parallel_fields_to_cpu(
        rollout_data,
        all_gather_with_cp,
        keep_restored=cp_rank == 0,
    )
    if cp_rank != 0:
        return
    assert rollout_data is not None

    rank = torch.distributed.get_rank()
    local_shard = {
        "rank": rank,
        "data_parallel_rank": mpu.get_data_parallel_rank(with_context_parallel=False),
        "rollout_data": rollout_data,
    }
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    dp_src_rank = mpu.get_data_parallel_src_rank(with_context_parallel=False)
    if dp_size == 1:
        dp_shards = [local_shard]
    else:
        dp_shards = [None] * dp_size if rank == dp_src_rank else None
        torch.distributed.gather_object(
            local_shard,
            dp_shards,
            dst=dp_src_rank,
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=False),
        )

    if rank != dp_src_rank:
        return

    path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
    logger.info(f"Save debug train data from {dp_size} DP shard(s) to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_build_dump_payload(dp_shards, rollout_id=rollout_id, writer_rank=rank), path)
