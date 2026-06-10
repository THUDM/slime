import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_HF_WEIGHT_FILE_NAMES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "tf_model.h5",
    "flax_model.msgpack",
}
_HF_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack")


def save_hf_model_to_path(
    args,
    output_dir: str | Path,
    model,
    *,
    model_name: str | None = None,
    quantization_config: dict[str, Any] | None = None,
    progress_desc: str = "Save HF checkpoint",
) -> None:
    """Save a Megatron model as an HF checkpoint at a concrete directory."""
    if args.megatron_to_hf_mode == "bridge":
        save_hf_model_bridge_to_path(args, output_dir, model)
    else:
        save_hf_model_direct_to_path(
            args,
            output_dir,
            model,
            model_name=model_name,
            quantization_config=quantization_config,
            progress_desc=progress_desc,
        )


def save_hf_model_direct_to_path(
    args,
    output_dir: str | Path,
    model,
    *,
    model_name: str | None = None,
    quantization_config: dict[str, Any] | None = None,
    progress_desc: str = "Save HF checkpoint",
) -> None:
    """Save a Megatron model as an HF safetensors checkpoint without Megatron Bridge."""
    path = Path(output_dir)
    hf_checkpoint = Path(args.hf_checkpoint).resolve()
    save_path = path.resolve()
    if hf_checkpoint == save_path:
        raise ValueError("HF save output path must not point to the same directory as --hf-checkpoint")
    if not hf_checkpoint.is_dir():
        raise ValueError(
            f"--hf-checkpoint must be a local directory when saving raw HuggingFace weights: {args.hf_checkpoint}"
        )

    import torch.distributed as dist
    from transformers import AutoConfig

    from .update_weight.common import named_params_and_buffers
    from .update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

    is_save_rank = _is_global_rank_zero()

    setup_error = None
    if is_save_rank:
        try:
            logger.info("Saving model in HuggingFace format to %s with raw Megatron-to-HF conversion", path)
            path.mkdir(parents=True, exist_ok=True)
            _clear_existing_hf_weights(path)
            _copy_hf_assets(args.hf_checkpoint, path)
        except Exception as e:
            setup_error = repr(e)

    _raise_if_rank_zero_failed("prepare raw HuggingFace save directory", setup_error)

    metadata_error = None
    payload: list[Any] = [None]
    if model_name is not None:
        payload = [(model_name, quantization_config)]
    else:
        if is_save_rank:
            try:
                hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                payload = [
                    (
                        type(hf_config).__name__.lower() if args.model_name is None else args.model_name,
                        getattr(hf_config, "quantization_config", None),
                    )
                ]
            except Exception as e:
                metadata_error = repr(e)
        _raise_if_rank_zero_failed("load HuggingFace conversion metadata", metadata_error)

    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    model_name, quantization_config = payload[0]

    hf_weight_iterator = HfWeightIteratorDirect(
        args=args,
        model=model,
        model_name=model_name,
        quantization_config=quantization_config,
    )
    megatron_local_weights = dict(named_params_and_buffers(args, model, convert_to_global_name=True))
    num_save_nodes, save_node_rank, is_writer_rank, writer_ranks = _get_node_save_layout(args)
    if is_save_rank:
        logger.info(
            "Raw HuggingFace save will write shards from %d node writer rank(s): %s",
            num_save_nodes,
            writer_ranks,
        )

    writer = _SafetensorShardWriter(path, enabled=is_writer_rank)
    pending_write = None

    for chunk_idx, hf_named_tensors in enumerate(
        hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights, progress_desc=progress_desc)
    ):
        if is_writer_rank and chunk_idx % num_save_nodes == save_node_rank:
            pending_write = (chunk_idx, hf_named_tensors)
            hf_named_tensors = None
        else:
            del hf_named_tensors

        if (chunk_idx + 1) % num_save_nodes == 0:
            pending_write = _write_pending_chunk(writer, pending_write)

    pending_write = _write_pending_chunk(writer, pending_write)
    _finalize_distributed_shards(path, writer.state())

    if is_save_rank:
        logger.info("Successfully saved HuggingFace model to %s", path)


def save_hf_model_bridge_to_path(args, output_dir: str | Path, model) -> None:
    """Save a Megatron model as an HF checkpoint through Megatron Bridge."""
    import torch.distributed as dist
    from megatron.bridge import AutoBridge
    from megatron.core import mpu

    from slime.utils.megatron_bridge_utils import patch_auto_bridge_hf_config, patch_megatron_model

    path = Path(output_dir)
    should_log = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )
    if should_log:
        logger.info("Saving model in HuggingFace format to %s with Megatron Bridge", path)

    path.mkdir(parents=True, exist_ok=True)
    bridge = patch_auto_bridge_hf_config(AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True))

    with patch_megatron_model(model):
        bridge.save_hf_pretrained(
            model,
            path=path,
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if should_log:
        logger.info("Successfully saved HuggingFace model to %s", path)


class _SafetensorShardWriter:
    def __init__(self, path: Path, *, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        self.total_size = 0
        self.weight_map: dict[str, str] = {}
        self.shard_files: list[str] = []

    def write(self, named_tensors, shard_idx: int) -> None:
        if not self.enabled:
            return
        assert shard_idx is not None, "shard_idx must be set when writing HF shards"

        from safetensors.torch import save_file

        state_dict = {}
        total_size = 0
        for name, tensor in named_tensors:
            if name in self.weight_map or name in state_dict:
                raise ValueError(f"Duplicate HF tensor while saving: {name}")
            total_size += tensor.numel() * tensor.element_size()
            state_dict[name] = _tensor_for_safetensors(tensor)

        if not state_dict:
            return

        filename = self._next_filename(shard_idx)
        if (self.path / filename).exists():
            raise ValueError(f"Duplicate HF shard file while saving: {filename}")

        save_file(state_dict, self.path / filename, metadata={"format": "pt"})
        self.shard_files.append(filename)
        self.total_size += total_size
        for name in state_dict:
            self.weight_map[name] = filename

    def state(self) -> dict[str, Any]:
        if not self.enabled:
            return {"total_size": 0, "weight_map": {}, "shard_files": []}
        return {
            "total_size": self.total_size,
            "weight_map": dict(self.weight_map),
            "shard_files": list(self.shard_files),
        }

    def finalize(self) -> None:
        if not self.enabled:
            return
        if not self.shard_files:
            raise ValueError("No HF tensors were produced while saving")

        total_files = len(self.shard_files)
        rename_map = {}
        for idx, old_name in enumerate(self.shard_files, start=1):
            new_name = f"model-{idx:05d}-of-{total_files:05d}.safetensors"
            os.replace(self.path / old_name, self.path / new_name)
            rename_map[old_name] = new_name

        final_weight_map = {name: rename_map[filename] for name, filename in self.weight_map.items()}
        index_data = {"metadata": {"total_size": self.total_size}, "weight_map": final_weight_map}
        with open(self.path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

    def _next_filename(self, shard_idx: int) -> str:
        assert shard_idx is not None, "shard_idx must be set when naming HF shards"
        return f"model-{shard_idx + 1:05d}.safetensors"


def _write_pending_chunk(
    writer: _SafetensorShardWriter, pending_write: tuple[int, Any] | None
) -> tuple[int, Any] | None:
    if pending_write is not None:
        shard_idx, named_tensors = pending_write
        writer.write(named_tensors, shard_idx=shard_idx)
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

    return None


def _finalize_distributed_shards(path: Path, local_state: dict[str, Any]) -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local_state)
    else:
        states = [local_state]

    if _is_global_rank_zero():
        _finalize_shard_files(path, states)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _finalize_shard_files(path: Path, shard_states: list[dict[str, Any] | None]) -> None:
    shard_files = []
    total_size = 0
    raw_weight_map = {}

    for state in shard_states:
        if not state:
            continue

        total_size += state.get("total_size", 0)
        for filename in state.get("shard_files", []):
            if filename in shard_files:
                raise ValueError(f"Duplicate HF shard file while finalizing: {filename}")
            shard_files.append(filename)

        for name, filename in state.get("weight_map", {}).items():
            if name in raw_weight_map:
                raise ValueError(f"Duplicate HF tensor while finalizing: {name}")
            raw_weight_map[name] = filename

    if not shard_files:
        raise ValueError("No HF tensors were produced while saving")

    shard_files = sorted(shard_files, key=_shard_filename_sort_key)
    total_files = len(shard_files)
    rename_map = {}
    for idx, old_name in enumerate(shard_files, start=1):
        new_name = f"model-{idx:05d}-of-{total_files:05d}.safetensors"
        os.replace(path / old_name, path / new_name)
        rename_map[old_name] = new_name

    final_weight_map = {}
    for name, filename in raw_weight_map.items():
        if filename not in rename_map:
            raise ValueError(f"HF tensor {name} points to missing shard file {filename}")
        final_weight_map[name] = rename_map[filename]

    index_data = {"metadata": {"total_size": total_size}, "weight_map": final_weight_map}
    with open(path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)


def _shard_filename_sort_key(filename: str) -> tuple[float, str]:
    prefix = "model-"
    suffix = ".safetensors"
    if filename.startswith(prefix) and filename.endswith(suffix):
        middle = filename[len(prefix) : -len(suffix)]
        if middle.isdigit():
            return int(middle), filename
    return math.inf, filename


def _tensor_for_safetensors(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor


def _clear_existing_hf_weights(path: Path) -> None:
    for item in path.iterdir():
        if item.is_file() and _is_hf_weight_file(item):
            item.unlink()


def _copy_hf_assets(origin_hf_dir: str, output_dir: Path) -> None:
    origin = Path(origin_hf_dir)
    if not origin.is_dir():
        raise ValueError(f"--hf-checkpoint must be a local directory when using raw --save-hf: {origin_hf_dir}")

    for item in origin.iterdir():
        if item.is_file():
            if _is_hf_weight_file(item):
                continue
            shutil.copy2(item, output_dir / item.name)


def _is_hf_weight_file(path: Path) -> bool:
    name = path.name
    return name in _HF_WEIGHT_FILE_NAMES or name.endswith(_HF_WEIGHT_FILE_SUFFIXES)


def _is_global_rank_zero() -> bool:
    import torch.distributed as dist

    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _get_node_save_layout(args) -> tuple[int, int, bool, list[int]]:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return 1, 0, True, [0]

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gpus_per_node = int(getattr(args, "actor_num_gpus_per_node", None) or getattr(args, "num_gpus_per_node", 1) or 1)
    gpus_per_node = max(1, gpus_per_node)
    inferred_nodes = max(1, math.ceil(world_size / gpus_per_node))
    configured_nodes = int(getattr(args, "actor_num_nodes", None) or inferred_nodes)
    num_nodes = max(1, min(configured_nodes, inferred_nodes))
    writer_ranks = [node * gpus_per_node for node in range(num_nodes) if node * gpus_per_node < world_size]
    node_rank = min(rank // gpus_per_node, num_nodes - 1)
    return len(writer_ranks), node_rank, rank in writer_ranks, writer_ranks


def _raise_if_rank_zero_failed(context: str, error: str | None) -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        payload = [error]
        dist.broadcast_object_list(payload, src=0)
        error = payload[0]

    if error is not None:
        raise RuntimeError(f"Failed to {context}: {error}")
