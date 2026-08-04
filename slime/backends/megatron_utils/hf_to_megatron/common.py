from __future__ import annotations

import functools
import json
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


class SafetensorReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        index_path = self.path / "model.safetensors.index.json"
        if index_path.is_file():
            with index_path.open() as index_file:
                self.weight_map = json.load(index_file)["weight_map"]
        else:
            files = sorted(self.path.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"No safetensors checkpoint found in {self.path}")
            self.weight_map = {}
            for file in files:
                with safe_open(file, framework="pt", device="cpu") as tensors:
                    self.weight_map.update(dict.fromkeys(tensors.keys(), file.name))
        self._files = {}

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    @functools.lru_cache(maxsize=1)  # noqa: B019 - cache belongs to this reader instance
    def get_tensor(self, name: str) -> torch.Tensor:
        try:
            filename = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"HuggingFace checkpoint does not contain {name!r}") from exc
        if filename not in self._files:
            self._files[filename] = safe_open(self.path / filename, framework="pt", device="cpu")
        tensor = self._files[filename].get_tensor(name)
        scale_name = f"{name}_scale_inv"
        if tensor.element_size() == 1 and scale_name in self:
            scale_file = self.weight_map[scale_name]
            if scale_file not in self._files:
                self._files[scale_file] = safe_open(self.path / scale_file, framework="pt", device="cpu")
            scale = self._files[scale_file].get_tensor(scale_name).to(torch.bfloat16)
            rows, columns = tensor.shape
            block_rows, block_columns = scale.shape
            tensor = F.pad(
                tensor.to(torch.bfloat16),
                (0, block_columns * 128 - columns, 0, block_rows * 128 - rows),
            )
            tensor = tensor.view(block_rows, 128, block_columns, 128)
            tensor.mul_(scale[:, None, :, None])
            tensor = tensor.reshape(block_rows * 128, block_columns * 128)[:rows, :columns]
        return tensor


def strip_mcore_wrappers(name: str) -> str:
    while name.startswith("module."):
        name = name.removeprefix("module.")
    return name.removeprefix("language_model.")


def text_config(config):
    return getattr(config, "text_config", config)


def merge_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, config) -> torch.Tensor:
    config = text_config(config)
    num_groups = config.num_key_value_heads
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // num_heads
    trailing_shape = q.shape[1:]
    q = q.reshape(num_groups, num_heads // num_groups * head_dim, *trailing_shape)
    k = k.reshape(num_groups, head_dim, *trailing_shape)
    v = v.reshape(num_groups, head_dim, *trailing_shape)
    return torch.cat((q, k, v), dim=1).reshape(-1, *trailing_shape).contiguous()


def merge_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.cat((gate, up), dim=0)


def _tensor_parallel_shard(
    name: str,
    tensor: torch.Tensor,
    *,
    parallel_size: int,
    parallel_rank: int,
    partition_dim: int,
    partition_stride: int,
) -> torch.Tensor:
    if parallel_size == 1:
        return tensor

    if "linear_fc1.weight" in name or "linear_fc1.bias" in name:
        gate, up = tensor.chunk(2, dim=partition_dim)
        gate = torch.chunk(gate, parallel_size, dim=partition_dim)[parallel_rank]
        up = torch.chunk(up, parallel_size, dim=partition_dim)[parallel_rank]
        return torch.cat((gate, up), dim=partition_dim).contiguous()

    if "linear_fc2.weight" in name and partition_dim == 0:
        partition_dim = 1

    chunks = torch.chunk(tensor, parallel_size * partition_stride, dim=partition_dim)
    return torch.cat(chunks[parallel_rank::parallel_size], dim=partition_dim).contiguous()


def shard_mcore_tensor(name: str, tensor: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
    from megatron.core import mpu

    if (
        not getattr(parameter, "tensor_model_parallel", False)
        or getattr(parameter, "parallel_mode", None) == "duplicated"
    ):
        return tensor

    if ".experts." in name:
        parallel_size = mpu.get_expert_tensor_parallel_world_size()
        parallel_rank = mpu.get_expert_tensor_parallel_rank()
    else:
        parallel_size = mpu.get_tensor_model_parallel_world_size()
        parallel_rank = mpu.get_tensor_model_parallel_rank()

    return _tensor_parallel_shard(
        name,
        tensor,
        parallel_size=parallel_size,
        parallel_rank=parallel_rank,
        partition_dim=parameter.partition_dim,
        partition_stride=parameter.partition_stride,
    )


def _pad_vocab(args, name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not (name.endswith("embedding.word_embeddings.weight") or name.endswith("output_layer.weight")):
        return tensor
    padded_size = getattr(args, "padded_vocab_size", None)
    if padded_size is None or tensor.shape[0] >= padded_size:
        return tensor
    return F.pad(tensor, (0, 0, 0, padded_size - tensor.shape[0]))


def load_model_hf_weights(
    args,
    model,
    path: str | Path,
    config,
    get_hf_tensor: Callable[[str, SafetensorReader, object], torch.Tensor],
) -> None:
    from slime.backends.megatron_utils.update_weight.common import named_params_and_buffers

    reader = SafetensorReader(path)
    with torch.no_grad():
        for name, parameter in named_params_and_buffers(args, model):
            tensor = get_hf_tensor(name, reader, config)
            if name.endswith("output_layer.weight") and parameter.shape[0] == 1 and tensor.shape[0] != 1:
                continue
            tensor = shard_mcore_tensor(name, _pad_vocab(args, name, tensor), parameter)
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"Shape mismatch loading {name}: HuggingFace {tuple(tensor.shape)}, "
                    f"Megatron {tuple(parameter.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
