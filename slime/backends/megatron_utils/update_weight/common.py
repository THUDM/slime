import inspect
import re
from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from slime.utils.types import ParamInfo


def _merge_param_partitions(name, partitions, partition_dim, partition_sections=None):
    if partition_sections is not None:
        tp_size = len(partitions)
        if any(size % tp_size for size in partition_sections):
            raise ValueError("Projection sections must be divisible by TP")
        local_sections = tuple(size // tp_size for size in partition_sections)
        split = [part.split(local_sections, dim=partition_dim) for part in partitions]
        result = torch.cat([part for section in zip(*split, strict=True) for part in section], dim=partition_dim)
        result.partition_sections = partition_sections
        return result
    if "linear_fc1.weight" in name or "linear_fc1.bias" in name:
        split = [part.chunk(2, dim=0) for part in partitions]
        partitions = [part[0] for part in split] + [part[1] for part in split]
    # Megatron grouped MoE exposes its FC2 partition dimension as zero.
    if "linear_fc2.weight" in name and partition_dim == 0:
        partition_dim = 1
    return torch.cat(partitions, dim=partition_dim)


def all_gather_param(name: str, param: torch.nn.Parameter) -> torch.Tensor:
    """
    All-gather TP-sharded param to full tensor. expert_bias→param,
    non-TP/duplicated/TP-size-1→param.data.
    Uses expert-TP for ".experts.", else regular-TP. linear_fc1 rechunked (GLU), linear_fc2 dim fix.
    """
    if "expert_bias" in name:
        return param

    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    data = param.data
    if (sections := getattr(param, "partition_sections", None)) is not None:
        data.partition_sections = sections
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        return data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()

    if tp_size == 1:
        return data

    if ".experts." in name:
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_group = mpu.get_tensor_model_parallel_group()

    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)
    partition_dim = param.partition_dim
    assert param.partition_stride == 1 or (
        param.partition_stride == 2 and "linear_fc1" in name
    ), "partition_stride != 1 is not supported"
    return _merge_param_partitions(name, param_partitions, partition_dim, getattr(param, "partition_sections", None))


def all_gather_params_async(
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
) -> list[torch.Tensor]:
    """
    Parallel TP all-gather for multiple params. Loop 1: for each TP param, allocate buffers +
    dist.all_gather(async_op=True) on expert-TP/regular-TP group
    (skip expert_bias/non-TP/duplicated/TP-size-1).
    Loop 2: wait all NCCL handles (enables overlap). Loop 3: concat partitions + apply GLU rechunk/MoE dim fix.
    """
    # Phase 1: Start all async all_gather operations
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        # Prepare async all_gather
        if "expert_bias" in info.name:
            gather_tasks.append((info, param, None, None, None))
        elif not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
            gather_tasks.append((info, param.data, None, None, None))
        else:
            # Start async all_gather
            if ".experts." in info.name:
                tp_size = mpu.get_expert_tensor_parallel_world_size()
            else:
                tp_size = mpu.get_tensor_model_parallel_world_size()

            if tp_size == 1:
                gather_tasks.append((info, param.data, None, None, None))
                continue

            if ".experts." in info.name:
                tp_group = mpu.get_expert_tensor_parallel_group()
            else:
                tp_group = mpu.get_tensor_model_parallel_group()

            param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
            handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim))
            handles.append(handle)

    # Phase 2: Wait for ALL async operations to complete at once
    # This ensures maximum parallelism by not blocking on individual operations
    for handle in handles:
        handle.wait()

    # Phase 3: Process all results after all communications are done
    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim in gather_tasks:
        if handle is None:
            # No all_gather needed
            param = direct_param
        else:
            assert partition_dim is not None, "partition_stride != 1 is not supported"
            param = _merge_param_partitions(
                info.name, param_partitions, partition_dim, info.attrs.get("partition_sections")
            )

        if (sections := info.attrs.get("partition_sections")) is not None:
            param.partition_sections = sections

        gathered_params.append(param)

    return gathered_params


def named_params_and_buffers(args: Namespace, model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Yield (global_name, param/buffer) with consistent names across PP/EP. Adjusts indices for
    virtual PP + EP offsets. Handles decoder.layers, mtp.layers (Multi-Token Prediction), expert_bias.
    """
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    if args.num_experts:
        expert_offset = ep_rank * args.num_experts // ep_size

    sig = inspect.signature(get_transformer_layer_offset)
    need_vp_stage = "vp_stage" in sig.parameters

    for vp_stage, model_module in enumerate(model):
        if need_vp_stage:
            layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        else:
            layer_offset = get_transformer_layer_offset(model_module.config)
        for name, param in model_module.named_parameters():
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name
            prefix = "module.module.language_model." if ".language_model." in name else "module.module."

            decoder_layers_pattern = r"module\.module\.(?:language_model\.)?decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                mtp_layers_pattern = r"module\.module\.(?:language_model\.)?mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    yield name, param
                    continue

                # MTP layer indices start from 0
                layer_idx, rest = match.groups()
                expert_pattern = r"transformer_layer\.mlp\.experts\.(.+)\.(weight|bias)(\d+)"
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                rest, param_type, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"{prefix}mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.{param_type}{expert_idx}", param
                continue

            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            expert_pattern = r"mlp\.experts\.(.+)\.(weight|bias)(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, param_type, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"{prefix}decoder.layers.{layer_idx}.mlp.experts.{rest}.{param_type}{expert_idx}", param
            else:
                yield f"{prefix}decoder.layers.{layer_idx}.{rest}", param

        # treat expert bias as normal parameters
        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers
            if "expert_bias" not in name:
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name
            prefix = "module.module.language_model." if ".language_model." in name else "module.module."

            decoder_layers_pattern = r"module\.module\.(?:language_model\.)?decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                layer_idx, rest = match.groups()
                layer_idx = int(layer_idx) + layer_offset
                yield f"{prefix}decoder.layers.{layer_idx}.{rest}", buffer
