"""Adapter-only checkpointing: save, load and metadata validation.

Layout of an adapter directory::

    adapter_config.json                          # config + provenance + parallel sizes
    adapter_model.safetensors.index.json         # key -> shard file
    adapter_model-tp{tp}-pp{pp}.safetensors
    training_state.json                          # rollout_id / policy_version / resume notes

Tensor keys use slime's canonical *global* parameter names, so they are
independent of data/context/expert parallelism. Routed experts share one adapter
per layer, whose name carries no expert index, so factors are written once per
``(tp_rank, pp_rank)`` coordinate and can be reloaded with a different EP size.
TP/PP/ETP mismatches are rejected instead of being silently reinterpreted.
Routed-expert injection requires ETP to divide TP, with ETP rank equal to TP rank modulo ETP.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from slime.utils.lora_config import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_FORMAT_VERSION,
    ADAPTER_INDEX_FILE,
    ADAPTER_TRAINING_STATE_FILE,
    LoRAConfig,
    is_lora_param_name,
)
from .layers import expert_tensor_parallel_world_size, parallel_rank_context

logger = logging.getLogger(__name__)


class LoRACheckpointError(RuntimeError):
    """Raised when an adapter checkpoint is missing, incomplete or incompatible."""


# --------------------------------------------------------------------------- #
# parallel-layout helpers
# --------------------------------------------------------------------------- #


def _parallel_layout() -> dict[str, int]:
    try:
        from megatron.core import mpu
    except ImportError:
        return {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "tensor_parallel_rank": 0,
            "pipeline_parallel_rank": 0,
        }
    return {
        "tensor_parallel_size": mpu.get_tensor_model_parallel_world_size(),
        "pipeline_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
        "expert_parallel_size": mpu.get_expert_model_parallel_world_size(),
        "expert_tensor_parallel_size": expert_tensor_parallel_world_size(),
        "tensor_parallel_rank": mpu.get_tensor_model_parallel_rank(),
        "pipeline_parallel_rank": mpu.get_pipeline_model_parallel_rank(),
    }


def _shard_filename(layout: dict[str, int]) -> str:
    return (
        f"adapter_model-tp{layout['tensor_parallel_rank']}"
        f"-pp{layout['pipeline_parallel_rank']}.safetensors"
    )


def _is_distributed() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def _barrier() -> None:
    if not _is_distributed():
        return
    import torch.distributed as dist

    from slime.utils.distributed_utils import get_gloo_group

    dist.barrier(group=get_gloo_group())


def _global_rank() -> int:
    if not _is_distributed():
        return 0
    import torch.distributed as dist

    return dist.get_rank()


def _broadcast_object(obj: Any) -> Any:
    if not _is_distributed():
        return obj
    import torch.distributed as dist

    from slime.utils.distributed_utils import get_gloo_group

    payload = [obj]
    dist.broadcast_object_list(payload, src=0, group=get_gloo_group())
    return payload[0]


def _gather_objects(obj: Any) -> list[Any]:
    if not _is_distributed():
        return [obj]
    import torch.distributed as dist

    from slime.utils.distributed_utils import get_gloo_group

    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, obj, group=get_gloo_group())
    return gathered


def _is_tp_pp_writer(layout: dict[str, int]) -> bool:
    """Select one global writer for each TP/PP shard, independent of EP/DP."""
    if not _is_distributed():
        return True
    coordinates = _gather_objects(
        (
            _global_rank(),
            layout["tensor_parallel_rank"],
            layout["pipeline_parallel_rank"],
        )
    )
    matching_ranks = [
        rank
        for rank, tp_rank, pp_rank in coordinates
        if tp_rank == layout["tensor_parallel_rank"] and pp_rank == layout["pipeline_parallel_rank"]
    ]
    return _global_rank() == min(matching_ranks)


# --------------------------------------------------------------------------- #
# state extraction
# --------------------------------------------------------------------------- #


def lora_state_dict(args: Namespace, model: Sequence[torch.nn.Module]) -> dict[str, torch.Tensor]:
    """Canonical-name -> LoRA tensor for this rank (adapters only, never the base)."""
    from ..update_weight.common import named_params_and_buffers

    return {
        name: param
        for name, param in named_params_and_buffers(args, model, convert_to_global_name=True)
        if is_lora_param_name(name)
    }


# --------------------------------------------------------------------------- #
# save
# --------------------------------------------------------------------------- #


@torch.no_grad()
def save_lora_adapter(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    path: str | Path,
    config: LoRAConfig,
    *,
    rollout_id: int | None = None,
    policy_version: str | None = None,
) -> Path:
    """Publish an immutable checkpoint directory through an atomically replaced symlink.

    Old versions remain available to readers that have resolved the previous link.
    Legacy real directories can still be loaded, but are never overwritten.
    """
    from safetensors.torch import save_file

    final_path = Path(path)
    layout = _parallel_layout()
    state = lora_state_dict(args, model)
    is_writer = _is_tp_pp_writer(layout)

    publication = None
    if _global_rank() == 0:
        try:
            if final_path.exists() and not final_path.is_symlink():
                raise LoRACheckpointError(
                    f"Cannot atomically overwrite the existing directory {final_path}; "
                    "use a new --save-lora path (for example one containing {rollout_id})."
                )
            final_path.parent.mkdir(parents=True, exist_ok=True)
            version_path = final_path.parent / f".{final_path.name}.version-{uuid.uuid4().hex}"
            version_path.mkdir()
            publication = (str(version_path), None)
        except (OSError, LoRACheckpointError) as exc:
            publication = (None, str(exc))
    version_path, error = _broadcast_object(publication)
    if error is not None:
        raise LoRACheckpointError(error)
    tmp_path = Path(version_path)

    shard_name = _shard_filename(layout)
    write_error = None
    try:
        if is_writer and state:
            cpu_state = {
                name: tensor.detach().to(device="cpu", dtype=tensor.dtype).contiguous()
                for name, tensor in state.items()
            }
            save_file(cpu_state, str(tmp_path / shard_name))
            del cpu_state
    except Exception as exc:
        write_error = f"rank {_global_rank()}: {exc}"

    shard_results = _gather_objects(({shard_name: sorted(state)} if (is_writer and state) else {}, write_error))
    errors = [error for _, error in shard_results if error is not None]
    if errors:
        raise LoRACheckpointError(f"Adapter shard write failed: {errors}. Previous checkpoint remains published.")

    publish_error = None
    if _global_rank() == 0:
        shard_map: dict[str, list[str]] = {}
        for entry, _ in shard_results:
            shard_map.update(entry)

        metadata = config.to_metadata()
        metadata.update(
            {
                "tensor_parallel_size": layout["tensor_parallel_size"],
                "pipeline_parallel_size": layout["pipeline_parallel_size"],
                "expert_parallel_size": layout["expert_parallel_size"],
                "expert_tensor_parallel_size": layout["expert_tensor_parallel_size"],
                "policy_version": policy_version,
                "rollout_id": rollout_id,
            }
        )
        link_path = final_path.parent / f".{final_path.name}.link-{uuid.uuid4().hex}"
        try:
            _write_json(tmp_path / ADAPTER_CONFIG_FILE, metadata)
            _write_json(
                tmp_path / ADAPTER_INDEX_FILE,
                {"metadata": {"format": "slime-lora", "format_version": ADAPTER_FORMAT_VERSION}, "shards": shard_map},
            )
            _write_json(tmp_path / ADAPTER_TRAINING_STATE_FILE, {
                "rollout_id": rollout_id,
                "policy_version": policy_version,
                "contains_optimizer_state": False,
                "contains_scheduler_state": False,
                "resume_note": (
                    "Adapter-only checkpoints restore LoRA weights only. For exact resume (optimizer moments, "
                    "LR scheduler, RNG state, rollout_id) use the standard Megatron checkpoint written by --save."
                ),
            })
            link_path.symlink_to(tmp_path.name, target_is_directory=True)
            os.replace(link_path, final_path)
            logger.info("Saved LoRA adapter checkpoint to %s", final_path)
        except Exception as exc:
            publish_error = str(exc)
        finally:
            link_path.unlink(missing_ok=True)

    publish_error = _broadcast_object(publish_error)
    if publish_error is not None:
        raise LoRACheckpointError(f"Adapter publication failed: {publish_error}. Previous checkpoint remains published.")

    _barrier()
    return final_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #


def read_adapter_metadata(path: str | Path) -> dict[str, Any]:
    config_path = Path(path) / ADAPTER_CONFIG_FILE
    if not config_path.is_file():
        raise LoRACheckpointError(f"{config_path} not found; {path!r} is not a slime LoRA adapter checkpoint")
    with config_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise LoRACheckpointError(f"{config_path} does not contain a JSON object")
    return metadata


def validate_adapter_metadata(metadata: dict[str, Any], config: LoRAConfig) -> None:
    """Reject adapters that do not belong to this base model / configuration."""
    layout = _parallel_layout()
    problems: list[str] = []

    if metadata.get("format_version") != ADAPTER_FORMAT_VERSION:
        problems.append(f"format_version {metadata.get('format_version')!r} != {ADAPTER_FORMAT_VERSION}")
    if int(metadata.get("rank", -1)) != config.rank:
        problems.append(f"rank {metadata.get('rank')!r} != --lora-rank {config.rank}")
    if float(metadata.get("alpha", -1.0)) != config.alpha:
        problems.append(f"alpha {metadata.get('alpha')!r} != --lora-alpha {config.alpha}")
    if str(metadata.get("bias", "none")) != config.bias:
        problems.append(f"bias {metadata.get('bias')!r} != --lora-bias {config.bias}")
    if list(metadata.get("target_modules", [])) != list(config.target_modules):
        problems.append(
            f"target_modules {metadata.get('target_modules')!r} != resolved targets {list(config.target_modules)}"
        )
    if list(metadata.get("excluded_modules", [])) != list(config.exclude_modules):
        problems.append(
            f"excluded_modules {metadata.get('excluded_modules')!r} != resolved excludes "
            f"{list(config.exclude_modules)}"
        )

    expected_hash = config.base_model_config_hash
    stored_hash = metadata.get("base_model_config_hash")
    if expected_hash is not None and stored_hash is not None and expected_hash != stored_hash:
        problems.append(
            f"base_model_config_hash {stored_hash!r} != current base model hash {expected_hash!r} "
            f"(base={config.base_model_path!r}). Refusing to load an adapter trained on a different base model."
        )

    for key in ("tensor_parallel_size", "pipeline_parallel_size", "expert_tensor_parallel_size"):
        stored = metadata.get(key)
        if stored is not None and int(stored) != int(layout[key]):
            problems.append(f"{key} {stored!r} != current {layout[key]!r}")

    if problems:
        raise LoRACheckpointError(
            "LoRA adapter checkpoint is incompatible with the current run:\n  - "
            + "\n  - ".join(problems)
            + f"\n  ranks: {parallel_rank_context()}"
        )


@torch.no_grad()
def load_lora_adapter(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    path: str | Path,
    config: LoRAConfig,
) -> dict[str, Any]:
    """Load adapter weights in place; returns the stored ``training_state.json`` payload."""
    from safetensors.torch import load_file

    # Pin one immutable version for metadata, shards and training state, even if
    # a concurrent save replaces the public symlink while this load is running.
    root = Path(path).resolve()
    metadata = read_adapter_metadata(root)
    validate_adapter_metadata(metadata, config)

    layout = _parallel_layout()
    state = lora_state_dict(args, model)
    if not state:
        return _read_training_state(root)

    if any(".experts." in name for name in state) and "expert_tensor_parallel_size" not in metadata:
        raise LoRACheckpointError(
            "Routed-expert adapter checkpoint has no expert_tensor_parallel_size metadata; "
            "its shard layout cannot be verified. Re-export it from the original training layout"
        )

    shard_path = root / _shard_filename(layout)
    if not shard_path.is_file():
        raise LoRACheckpointError(
            f"Adapter shard {shard_path.name!r} is missing from {root}. The checkpoint was written with a "
            f"different parallel layout (ranks={parallel_rank_context()})."
        )

    loaded = load_file(str(shard_path))
    missing = sorted(set(state) - set(loaded))
    unexpected = sorted(set(loaded) - set(state))
    if missing or unexpected:
        raise LoRACheckpointError(
            f"Adapter shard {shard_path.name!r} does not match the injected adapters.\n"
            f"  missing in checkpoint : {missing[:10]}\n"
            f"  unknown in checkpoint : {unexpected[:10]}\n"
            f"  ranks: {parallel_rank_context()}"
        )

    for name, param in state.items():
        tensor = loaded[name]
        if tuple(tensor.shape) != tuple(param.shape):
            raise LoRACheckpointError(
                f"Shape mismatch for adapter tensor {name!r}: checkpoint {tuple(tensor.shape)}, "
                f"model {tuple(param.shape)} (ranks={parallel_rank_context()})"
            )
        param.copy_(tensor.to(device=param.device, dtype=param.dtype))

    del loaded
    logger.info("Loaded LoRA adapter checkpoint from %s", root)
    return _read_training_state(root)


def _read_training_state(root: Path) -> dict[str, Any]:
    state_path = root / ADAPTER_TRAINING_STATE_FILE
    if not state_path.is_file():
        return {}
    with state_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def validate_megatron_lora_checkpoint(path: str | Path, model: Sequence[torch.nn.Module]) -> None:
    """Require all requested adapter keys before delegating to Megatron's loader.

    Fresh LoRA runs use the HF loading path. A Megatron checkpoint must already
    contain the adapters, even under --finetune; we never globally relax missing
    model-key checks to accommodate a base-only checkpoint.
    """
    from megatron.core.dist_checkpointing.serialization import load_tensors_metadata

    try:
        checkpoint_keys = set(load_tensors_metadata(str(path)))
    except Exception as exc:
        raise LoRACheckpointError(
            f"Cannot inspect LoRA tensor metadata in Megatron checkpoint {path}. "
            "Use a distributed checkpoint with readable tensor metadata, or initialize "
            "a new run from HF base weights via --load HF_DIR"
        ) from exc

    expected = set()
    for chunk in model:
        stack = [chunk.sharded_state_dict()]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
            else:
                key = getattr(value, "key", None)
                if isinstance(key, str) and is_lora_param_name(key):
                    expected.add(key)
    if not expected:
        raise LoRACheckpointError("Cannot verify Megatron LoRA checkpoint: model exposes no sharded adapter keys")
    missing = sorted(expected - checkpoint_keys)
    if missing:
        raise LoRACheckpointError(
            f"Megatron checkpoint {path} is missing requested LoRA tensors: {missing[:10]}. "
            "Base-only or incomplete LoRA checkpoints cannot be loaded into this LoRA model. "
            "For a new run, use --load HF_DIR and optionally --lora-load ADAPTER_DIR. "
            "--finetune does not waive adapter completeness checks"
        )
