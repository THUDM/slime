"""
TODO s:
- entrypoint function in `slime/slime/backends/sglang_utils/sglang_engine.py`
- entrypoint function in `UpdateWeightsFromDistributedReqInput` of sglang, "/mnt/home/xinji1/clean_sglang/sglang/python/sglang/srt/managers/io_struct.py"
- if it's better to specifiy the port directly in __init__ of  P2PTransferEngine?  rather than creating the session id over and over again?
- merge `init_p2p_transfer_engine` into `init_distributed_weight`
- replace dict with Sequence[str, ...]

Q:
- always re-connect/rebuild the communication group? why?
- zmq should be build only once or multiple times?
- mooncake `store?`
- `disconnect_rollout_engines_from_distributed`?
- destroy the transfer_engine in the end?
"""

import inspect
import re
import socket
import time
from argparse import Namespace
from collections.abc import Iterator, Mapping, Sequence
from typing import Callable, Optional

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from ray import ObjectRef
from ray.actor import ActorHandle
import zmq
try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from tqdm import tqdm

from slime.utils.distributed_utils import get_gloo_group, init_process_group
from slime.utils.types import ParamInfo

from .megatron_to_hf import convert_to_hf  # noqa: F401

try:
    try:
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
    except ImportError:
        from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

    use_flattened_tensor_bucket = True
except Exception:
    use_flattened_tensor_bucket = False

import zmq
from collections import defaultdict
from queue import Queue
from typing import Dict, Optional
import logging
logger = logging.getLogger(__name__)
try:
    from slime.backends.sglang_utils.sglang_rdma_p2p_transfer import P2PTransferEngine
    support_sglang_p2p_transfer = True
except Exception:
    support_sglang_p2p_transfer = False

def all_gather_param(name: str, param: torch.nn.Parameter) -> torch.Tensor:
    """
    All-gather TP-sharded param to full tensor. expert_bias→param, non-TP/duplicated→param.data.
    Uses expert-TP for ".experts.", else regular-TP. linear_fc1 rechunked (GLU), linear_fc2 dim fix.
    """
    if "expert_bias" in name:
        return param

    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)
    partition_dim = param.partition_dim
    assert param.partition_stride == 1, "partition_stride != 1 is not supported"
    # TODO: here we did an extra copy during concat, maybe merge this with convert_to_hf is better?
    # TODO: check only GLU is used.
    if "linear_fc1.weight" in name:
        param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
        param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
    # this is bug in megatron's grouped moe.
    if "linear_fc2.weight" in name:
        if partition_dim == 0:
            partition_dim = 1
    param = torch.cat(param_partitions, dim=partition_dim)
    return param


def all_gather_params_async(
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
) -> list[torch.Tensor]:
    """
    Parallel TP all-gather for multiple params. Loop 1: for each TP param, allocate buffers +
    dist.all_gather(async_op=True) on expert-TP/regular-TP group (skip expert_bias/non-TP/duplicated).
    Loop 2: wait all NCCL handles (enables overlap). Loop 3: concat partitions + apply GLU rechunk/MoE dim fix.
    """
    # Phase 1: Start all async all_gather operations
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        # Prepare async all_gather
        if "expert_bias" in info.name:
            gather_tasks.append((info, param, None, None, None))
            handles.append(None)
        elif not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
            gather_tasks.append((info, param.data, None, None, None))
            handles.append(None)
        else:
            # Start async all_gather
            if ".experts." in info.name:
                tp_size = mpu.get_expert_tensor_parallel_world_size()
                tp_group = mpu.get_expert_tensor_parallel_group()
            else:
                tp_size = mpu.get_tensor_model_parallel_world_size()
                tp_group = mpu.get_tensor_model_parallel_group()

            param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
            handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim))
            handles.append(handle)

    # Phase 2: Wait for ALL async operations to complete at once
    # This ensures maximum parallelism by not blocking on individual operations
    for handle in handles:
        if handle is not None:
            handle.wait()

    # Phase 3: Process all results after all communications are done
    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim in gather_tasks:
        if handle is None:
            # No all_gather needed
            param = direct_param
        else:
            # Process the gathered partitions (same logic as original all_gather_param)
            assert partition_dim is not None, "partition_stride != 1 is not supported"
            # TODO: here we did an extra copy during concat, maybe merge this with convert_to_hf is better?
            # TODO: check only GLU is used.
            if "linear_fc1.weight" in info.name:
                param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
                param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
            # this is bug in megatron's grouped moe.
            if "linear_fc2.weight" in info.name:
                if partition_dim == 0:
                    partition_dim = 1
            param = torch.cat(param_partitions, dim=partition_dim)

        gathered_params.append(param)

    return gathered_params


def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Remove vocab padding: param[:vocab_size] for embedding/output layers, else unchanged.
    """
    if name == "module.module.embedding.word_embeddings.weight" or name == "module.module.output_layer.weight":
        return param[:vocab_size]
    return param


def named_parameters(args: Namespace, model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
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

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    yield name, param
                    continue

                # MTP layer indices start from 0
                layer_idx, rest = match.groups()
                expert_pattern = r"transformer_layer.mlp.experts\.(.+)\.weight(\d+)"
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.weight{expert_idx}", param
                continue

            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.weight{expert_idx}", param
            else:
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param

        # treat expert bias as normal parameters
        for name, buffer in model_module.named_buffers():
            if "expert_bias" not in name:
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                layer_idx, rest = match.groups()
                layer_idx = int(layer_idx) + layer_offset
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", buffer


def get_param_infos(args: Namespace, model: Sequence[torch.nn.Module]) -> list[ParamInfo]:
    """
    Build global param metadata: collect → exchange PP/EP → resolve duplicates (MTP virtual PP)
    by min src_rank → validate. Returns sorted ParamInfo identical across all ranks.
    """
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    param_infos = {}
    rank = dist.get_rank()
    for name, param in named_parameters(args, model):
        param_infos[name] = ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={
                "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                "partition_dim": getattr(param, "partition_dim", -1),
                "partition_stride": getattr(param, "partition_stride", 1),
                "parallel_mode": getattr(param, "parallel_mode", None),
            },
            size=param.numel() * param.element_size(),
            src_rank=rank,
        )

    if pp_size > 1:
        param_infos_list = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_pipeline_model_parallel_group()
        )
        for src_rank, infos in param_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                if name in param_infos:
                    assert args.mtp_num_layers is not None
                    old_info = param_infos[name]
                    if old_info.src_rank > src_rank:
                        param_infos[name] = info
                else:
                    param_infos[name] = info

    if ep_size > 1:
        param_infos_list = [None] * ep_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_expert_model_parallel_group()
        )
        for src_rank, infos in param_infos_list:
            for name, info in infos.items():
                if name not in param_infos:
                    # here we need to set the src_rank to the rank within the expert model parallel group
                    info.src_rank = src_rank
                    param_infos[name] = info

    param_infos = list(param_infos.values())
    param_infos = sorted(param_infos, key=lambda info: info.name)

    # Check all ranks has the same parameter info
    all_param_info_list = [None] * dist.get_world_size()
    dist.all_gather_object(
        obj=param_infos,
        object_list=all_param_info_list,
        group=get_gloo_group(),
    )
    for i, param_info in enumerate(param_infos):
        for infos in all_param_info_list:
            assert infos[i].name == param_info.name, f"Parameter name mismatch: {infos[i].name} != {param_info.name}"
            assert (
                infos[i].shape == param_info.shape
            ), f"Parameter shape mismatch: {infos[i].shape} != {param_info.shape}"
            assert (
                infos[i].dtype == param_info.dtype
            ), f"Parameter dtype mismatch: {infos[i].dtype} != {param_info.dtype}"

    return param_infos


def get_param_info_buckets(args: Namespace, model: Sequence[torch.nn.Module]) -> list[list[ParamInfo]]:
    """
    Partition params into buckets ≤ update_weight_buffer_size (with TP replication).
    """
    param_infos = get_param_infos(args, model)
    param_info_buckets = [[]]  # Start with one empty bucket
    buffer_size = 0  # Track current bucket size in bytes

    for info in param_infos:
        # Expert params use expert-TP size, others use regular-TP size
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()

        # Full param size = shard size × TP replicas (all-gather will reconstruct full param)
        param_size = info.size * tp_size

        # If adding this param exceeds limit AND current bucket has params: start new bucket
        if buffer_size + param_size > args.update_weight_buffer_size and len(param_info_buckets[-1]) > 0:
            param_info_buckets.append([])
            buffer_size = 0

        # Add param to current bucket and update size
        param_info_buckets[-1].append(info)
        buffer_size += param_size

    return param_info_buckets


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict→GPU) → broadcast PP/EP(GPU NCCL) → gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: GPU→CPU serialize → gather_object(Gloo CPU, collects from rollout_num_gpus_per_engine ranks) → Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        vocab_size: int,
    ) -> None:
        """
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.quantization_config = quantization_config
        self.param_info_buckets = get_param_info_buckets(self.args, self.model)
        self.weight_version = 0

        # create the group within megatron.
        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = start_rank + self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

        self._model_update_groups = None

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
        colocate_engine_nums = (
            self.args.actor_num_nodes * self.args.actor_num_gpus_per_node // self.args.rollout_num_gpus_per_engine
        )
        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "slime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args, self._group_name, self.distributed_rollout_engines
                )

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            if dist.get_rank() in group_ranks:
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        weights = self.weights_getter()

        num_buckets = len(self.param_info_buckets)
        for i in tqdm(range(num_buckets), disable=rank != 0, desc="Update weights"):
            current_params, current_infos = self._gather_bucket_params(self.param_info_buckets[i], weights)
            refs = self._update_converted_params_from_tensor(current_params, current_infos)
            ray.get(refs)
            del current_params, current_infos

        dist.barrier(group=get_gloo_group())

    def _gather_bucket_params(
        self,
        param_infos: Sequence[ParamInfo],
        weights,
    ) -> tuple[Sequence[torch.Tensor], Sequence[ParamInfo]]:
        monkey_patch_torch_reductions()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        ep_size = mpu.get_expert_model_parallel_world_size()
        rank = dist.get_rank()
        # init params:
        params = []
        for info in param_infos:
            if dist.get_rank() == info.src_rank:
                params.append(
                    torch.nn.Parameter(
                        weights[info.name].to(device=torch.cuda.current_device(), non_blocking=True),
                        requires_grad=False,
                    )
                )
            else:
                params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))
        torch.cuda.synchronize()

        # broadcast params across pp ranks
        if pp_size > 1:
            handles = []
            for info, param in zip(param_infos, params):
                if info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
                    handles.append(
                        torch.distributed.broadcast(
                            param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group(), async_op=True
                        )
                    )
            for handle in handles:
                handle.wait()

        # broadcast params across ep ranks
        if ep_size > 1:
            handles = []
            for info, param in zip(param_infos, params):
                if ".experts." in info.name:
                    src_rank = (
                        info.src_rank
                        if info.src_rank in dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
                        else rank
                    )
                    handles.append(
                        torch.distributed.broadcast(
                            param, src=src_rank, group=mpu.get_expert_model_parallel_group(), async_op=True
                        )
                    )
            for handle in handles:
                handle.wait()

        # Set tp attrs for all params
        for info, param in zip(param_infos, params):
            for key, value in info.attrs.items():
                setattr(param, key, value)

        # Batch async all_gather for all parameters
        gathered_params = all_gather_params_async(list(zip(param_infos, params)))

        return gathered_params, param_infos

    def _update_converted_params_from_tensor(
        self, gathered_params: Sequence[torch.Tensor], param_infos: list[ParamInfo]
    ) -> list[ObjectRef]:

        converted_named_tensors = []
        for info, param in zip(param_infos, gathered_params):
            param = remove_padding(info.name, param, self.vocab_size)
            converted_named_tensors.extend(
                convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config)
            )

        all_refs = []

        refs_colocated = self._send_to_colocated_engine(converted_named_tensors)
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self.args,
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                converted_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs

    def _send_to_colocated_engine(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> list[ObjectRef]:
        if use_flattened_tensor_bucket:
            if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
                converted_named_tensors_by_dtypes = {"dtype": converted_named_tensors}
            else:
                converted_named_tensors_by_dtypes = {}
                for name, tensor in converted_named_tensors:
                    dtype = tensor.dtype
                    if dtype not in converted_named_tensors_by_dtypes:
                        converted_named_tensors_by_dtypes[dtype] = []
                    converted_named_tensors_by_dtypes[dtype].append((name, tensor))

            serialized_tensors = []
            for dtype, named_tensors in converted_named_tensors_by_dtypes.items():
                flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
                metadata = flattened_tensor_bucket.get_metadata()
                flattened_tensor_data = {
                    "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                    "metadata": metadata,
                }
                serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))
        else:
            serialized_tensors = MultiprocessingSerializer.serialize(converted_named_tensors, output_str=True)

        serialized_named_tensors = (
            [None] * dist.get_world_size(self._ipc_gather_group) if self._ipc_gather_src == dist.get_rank() else None
        )
        dist.gather_object(
            serialized_tensors,
            object_gather_list=serialized_named_tensors,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        if dist.get_rank() == self._ipc_gather_src:
            refs = []
            if use_flattened_tensor_bucket:
                # TODO: here we assume all ranks have the same number of dtypes, not sure if that is correct.
                num_dtypes = len(serialized_named_tensors[0])
                for i in range(num_dtypes):
                    kwargs = {
                        "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                        "load_format": "flattened_bucket",
                        "weight_version": str(self.weight_version),
                    }
                    refs.append(self._ipc_engine.update_weights_from_tensor.remote(**kwargs))
            else:
                kwargs = {
                    "serialized_named_tensors": serialized_named_tensors,
                    "weight_version": str(self.weight_version),
                }
                refs.append(self._ipc_engine.update_weights_from_tensor.remote(**kwargs))
            return refs
        return []

# TODO(xinji1): where the updates for:
# - accept the signals from rollout engines that weights from training side could be sent now.
# - collect the weights within each pp group
# - mapping ?
# - start sending weights
# - wait for the signals from rollout workers.


class UpdateWeightFromDistributedRDMA:
    """
    Update distributed engines via p2p Transfer. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        vocab_size: int,
    ) -> None:
        """
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        # The following params are for p2p rdma transfering
        assert self.args.update_weights_p2p_transfer
        master_address, master_port = get_master_address_and_port()
        self.master_address = master_address
        # In NCCL, master_port is for NCCL communication;
        # In p2p RDMA, master_port is for the zmq communication
        self.master_port = master_port
        
        self.rollout_engines_index_mapping : dict[int, Optional[list[int]]]| None = None
        self.selected_rollout_engines: Sequence[ActorHandle] | None = None
        
        self.training_p2p_transfer_engine = P2PTransferEngine(
            hostname = self.master_address,
            gpu_id=0, # training host
            ib_device= None # TODO : check
        )

        
        # TODO(xinji1): type
        # TODO(xinji1): if it's necessary that we need to destory the 
        self.context = None
        self.router_socket = None
        # index mapping from pp_rank to the ranks of rollout_engines,
        # indicating that which rollout_engines should be linked
        
        if self.args.update_weights_p2p_transfer:
            assert support_sglang_p2p_transfer, "Please check the importing of `MooncakeTransferEngine` in sglang"

    # a message loop is needed
    def 
    
    def _initial_rollout_mapping(self):
        """ build the index mapping from training ranks of different pp_src_ranks to the gpus/nodes of rollout engines"""
        # TODO: how to build the index mapping from pp_src_rank to rollout_engine ranks
        # This function should initialize `self.rollout_engines_index_mapping`
        pass
    
    def select_rollout_engines(self, rollout_engines: Sequence[ActorHandle], pp_rank: int) -> None:
        """Select rollout engins given pp_rank"""
        if self.rollout_engines_index_mapping is None:
            self.selected_rollout_engines = rollout_engines
        else:
            candidates = [engine for i, engine in enumerate(rollout_engines) if i in self.rollout_engines_index_mapping[pp_rank]]
            self.selected_rollout_engines = Sequence(candidates) 

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Create NCCL/P2P RDMA "slime-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self._initial_rollout_mapping()
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        # For TP:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # TODO: should we update the selected rollout engines every time
        self.select_rollout_engines(rollout_engines=rollout_engines, pp_rank=pp_rank)

        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines,
                )
            # initialize address and port
            # TODO(xinji1): whether we should move this part into __init__ ? 
            master_address, master_port = get_master_address_and_port()
            self.context, self.router_socket = initialize_socket_for_weight_transfering(
                master_address=master_address,
                master_port=master_port
            )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, 
                self._group_name, 
                rollout_engines, 
                rollout_engines_index_mapping=None, # TBD
                master_address=master_address,
                master_port=master_port,
            )
    @torch.no_grad()
    def update_weights(self) -> None:
        if self.args.update_weights_p2p_transfer:
            self.update_weights_rdma()
        else:
            self.update_weights_nccl()

    def update_weights_nccl(self) -> None:
        """
        Pause → flush → non-expert (TP) → expert (EP) → continue. Progress on PP source.
        """
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        converted_named_tensors = []
        # non expert params
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

        for name, param in named_parameters(self.args, self.model):
            if ".experts." in name:
                continue
            buffer_size = self._update_weight_from_distributed(
                name, param, converted_named_tensors, buffer_size, pbar=pbar
            )

        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        named_tensors = []
        for name, param in named_parameters(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_weight_from_distributed(
                name, param, named_tensors, buffer_size, pbar=pbar
            )

        if named_tensors:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def update_weights_rdma(self) -> None:
        """
        Update weights with P2P rdma
        Pause → flush → non-expert (TP) → expert (EP) → continue. Progress on PP source.
        """
        self.weight_version += 1
        
        
        # TODO(): we may choose to pause engine/flush_chache based on the pp_src_rank(self._group_name)
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            # TODO(xinji1): For rollout, whether we should decouple `prepare_tensor_buffer_and_send_sync_status`,
            # and `transfering weight from training side`? 
        dist.barrier(group=get_gloo_group())
        buffer_size = 0
        converted_named_tensors = []
        # non expert params
        pbar = tqdm(desc=f"[{self._group_name}] Update weights through P2P RDMA", total=0) if self._is_pp_src_rank else None

        for name, param in named_parameters(self.args, self.model):
            if ".experts." in name:
                continue
            buffer_size = self._update_weight_from_distributed(
                name, param, converted_named_tensors, buffer_size, pbar=pbar
            )

        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        named_tensors = []
        for name, param in named_parameters(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_weight_from_distributed(
                name, param, named_tensors, buffer_size, pbar=pbar
            )

        if named_tensors:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())
        # TODO: close the zmq socket.
        if dist.get_rank() == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _update_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int | None:
        """
        Non-expert: gather TP → rm pad → HF → buffer (flush if full). All gather, PP source buffers.
        Returns updated bytes on source, None on non-source.
        """
        param = all_gather_param(name, param)
        param = remove_padding(name, param, self.vocab_size)
        if not self._is_pp_src_rank:
            return

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
            buffer_size = 0
        converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        buffer_size += param_size
        return buffer_size

    def _update_expert_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        param = all_gather_param(name, param)
        param = remove_padding(name, param, self.vocab_size)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_distributed(
        self, named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Gather EP → HF → broadcast. Clears buffer.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_pp_src_rank:
            return

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        self._update_bucket_weights_from_distributed(converted_hf_tensors, pbar)

    def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        if self.args.update_weights_p2p_transfer:
            self._update_bucket_weights_from_distributed_rdma(converted_named_tensors=converted_named_tensors, pbar=pbar)
        else:
            self._update_bucket_weights_from_distributed_nccl(converted_named_tensors=converted_named_tensors, pbar=pbar)

    def _update_bucket_weights_from_distributed_rdma(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Lock → broadcast → clear → unlock → pbar++. 
        """
        # lock the rollout engines to prevent dead lock on broadcast.
        # TODO(xinji1): if rdma needs it too.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        refs = update_weights_from_distributed(
            self.args,
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.selected_rollout_engines, # selected ones
            converted_named_tensors,
            self.context, 
            self.router_socket,
        )

        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        pbar.update(1)

    def _update_bucket_weights_from_distributed_nccl(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Lock → broadcast → clear → unlock → pbar++. Lock prevents NCCL deadlock.
        """
        # lock the rollout engines to prevent dead lock on broadcast.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        refs = update_weights_from_distributed(
            self.args,
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            converted_named_tensors,
        )

        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        pbar.update(1)



def get_master_address_and_port() -> tuple[str, int]:
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    return master_address, master_port

def initialize_socket_for_weight_transfering(master_address: str, master_port: int):
    """
    Build context/socket for the weight transfering between training side and rollout side.
    Specifically, the socket is used for:
    - Rollout side: sending messages to training side, that the buffer is ready, and weights
        could be transfered from training side.
    - Training side: sending messages to rollout side, that the weight transfering is done.
       
    # TODO(): may be replaced with Mooncake's TransferEngine store?
    """
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{master_address}:{master_port}")
    return context, router_socket

def connect_rollout_engines_from_distributed(
    args: Namespace, group_name: str, rollout_engines: Sequence[ActorHandle],
    rollout_engines_index_mapping: list[int] | None = None,
    master_address: str | None = None,
    master_port: int | None = None,
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.
    
    Args:
        rollout_engines_index_mapping: list[int]. which rollout engines should be
            connected with TransferEngine, when p2p weight transfering
            enabled.
    """
    if master_address is None or master_port is None:
        master_address, master_port = get_master_address_and_port()
    world_size = len(rollout_engines) * args.rollout_num_gpus_per_engine + 1
    # NOTE(xinji1): where the init group of weight_transfer in traning side
    # happens. Note that currently the group is only for the pp_src_rank of 
    # pp group for training side.
    # TODO(xinji1):
    # in unittest, build other remote groups like the op here.
    # TODO(xinji1): build a new class for RDMA transfering (nccl, rdma)
    if args.update_weights_p2p_transfer:
        # TODO: new function
        # build the p2p link from the host to the file 
        # TODO: To get the best mapping, from current host to the 
        # ranks in rollout engine. 
        # TODO: when # training senders > # rollout engines.
        # 1. send weights to some of the gpus in one rollout engine.
        # 2. send weights to all gpus of one rollout engine, then split
        # the weights from rollout side.
        # Method 2 should be more strightforward?
        if rollout_engines_index_mapping is None:
            # default: connect the host to all rollout_engines
            rollout_engines_index_mapping = list(range(len(rollout_engines)))
        selected_rollout_engines = [engine for i, engine in enumerate(rollout_engines) if i in rollout_engines_index_mapping ]
        # Build p2p for the selected engines    
        refs = [
            engine.init_p2p_transfer_engine.remote(
                master_address, # TODO(xinji1): if it's right to use master_address only?
                master_port, # TODO(xinji1): p2p transfer port should be intiialized here.
                i * args.rollout_num_gpus_per_engine + 1, # TODO(xinji1): another offset in `init_p2p_transfer_engine` is necessary?
            )
            for i, engine in enumerate(selected_rollout_engines) # TODO(xinji1): seems not necessary to build p2p for all engines
        ]
        
        model_update_groups = P2PTransferEngine(
            hostname=master_address,
            gpu_id=0, # let rank of training side be 0
            ib_device=None, # TODO(xinji1): None?
        )
    else:
        # here there's one process group of pp_src_rank 
        # and all the other ranks in rollout engine.
        refs = [
            engine.init_weights_update_group.remote(
                master_address,
                master_port,
                i * args.rollout_num_gpus_per_engine + 1,
                world_size,
                group_name,
                backend="nccl",
            )
            for i, engine in enumerate(rollout_engines)
        ]
        model_update_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0,
            group_name=group_name,
        )
        ray.get(refs) # to get the result from remote
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL/p2p RDMA commucation on training and engines.
    """
    if args.update_weights_p2p_transfer:
        raise NotImplementedError("not support destorying the p2p RDMA yet")
    else:
        refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
        dist.destroy_process_group(model_update_groups)
        ray.get(refs)


def update_weights_from_distributed(
    args: Namespace,
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    context,
    router_socket,
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors.
    """
    if args.update_weights_p2p_transfer:
        return update_weights_from_distributed_rdma(
            args, group_name, group, weight_version, rollout_engines, converted_named_tensors,
            context, router_socket
        )
    else:
        return update_weights_from_distributed_nccl(
            args, group_name, group, weight_version, rollout_engines, converted_named_tensors
        )

# TODO(xinji1): it's better to involve it into a class..
def update_weights_from_distributed_rdma(
    args: Namespace,
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    context,
    router_socket,
) -> list[ObjectRef]:
    """
    Send metadata (Ray), passing tensors with P2P RDMA.

    Prerequisites:
    - there're two types of seesion id in this case that should be kept in mind:
      - p2p_session_id: for the zmq communication of training and rollout, like
          syncing the buffer status. Usually it's `host_ip:port`
      - transfer_session_id: locally generated from Mooncake's Engine, indicating
          the id of weight transfering session.

    Steps:
    - Enable the update_weights_from_distributed so that the training host could get 
      - whether the buffer of rollout engine is ready
      - session_id
    """
    # TODO(xinji1):  build thread pool. If we could use different ports to pass weights? 
    
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_engines
    ]
    
    # first round syncing: buffer ready/ transfer_session_id from zmq
    assert context is not None and router_socket is not None, "for p2p transfering, the zmq should be initialized first."
    rollout_count = len(rollout_engines)
    while rollout_count > 0:
        # TODO(xinji1): right?
        try:
            if router_socket.poll(timeout=100):  # 100ms timeout
                # ROUTER receives frames from DEALER
                frames = router_socket.recv_multipart()

                logger.debug(f"[Slime Weights Updating] Received {len(frames)} frames, "
                            f"frame lengths: {[len(f) for f in frames]}")

                if len(frames) < 2:
                    logger.warning(f"[Slime Weights Updating] Received malformed message with {len(frames)} frames")
                    continue

                # ROUTER adds identity as first frame
                # Format is typically: [identity, delimiter (empty), actual_message]
                identity = frames[0]

                # Try to find the JSON payload
                message = None
                for i in range(1, len(frames)):
                    try:
                        if len(frames[i]) > 0:  # Skip empty frames
                            message = zmq.utils.jsonapi.loads(frames[i])
                            logger.debug(f"[Slime Weights Updating] Found JSON in frame {i}")
                            break
                    except Exception:
                        continue

                if message is None:
                    logger.error(f"[Slime Weights Updating] Could not decode JSON from any frame")
                    continue

                msg_type = message.get("type", "")

                if msg_type == "sync_status":
                    # Handle sync_status from rollout worker
                    # where the weights transfering happens.
                    _handle_sync_status(identity, message)
                else:
                    logger.warning(f"[Slime Weights Updating] Unknown message type: {msg_type}")
                
                rollout_count -=1
        except zmq.ZMQError as e:
            if rollout_count > 0:
                logger.error(f"[Slime Weights Updating] ZMQ error: {e}")
            break
        except Exception as e:
            logger.error(f"[Slime Weights Updating] Error in message loop: {e}", exc_info=True)


    # handles = []
    # for _, param in converted_named_tensors:
    #     handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    # for handle in handles:
    #     handle.wait()

    return refs

def _handle_sync_status(identity: bytes, message: dict, converted_named_tensors:  Sequence[tuple[str, torch.Tensor]], engine, router_socket):
    """
    Handle sync_status message from rollout worker.
    Message format:
    {
        "type": "sync_status",
        "p2p_session_id": "training_ip:training_port",  # Training's ZMQ address
        "status": "ready",
        "ip": rollout_ip,
        "transfer_session_id": rollout_mooncake_session_id,  # Rollout's Mooncake RPC address
        "ptr": rollout_ptr,
        "length": buffer_length,
        "task_id": task_id
    }
    """
    # Log the raw message for debugging
    logger.debug(f"[Slime Weights Updating] Raw message: {message}")

    p2p_session_id = message.get("p2p_session_id", "")
    rollout_ip = message.get("ip", "")
    rollout_transfer_session_id = message.get("transfer_session_id", "")
    rollout_ptr = message.get("ptr", 0)
    rollout_length = message.get("length", 0)
    task_id = message.get("task_id", "")

    logger.info(
        f"[Slime Weights Updating] Received sync_status: "
        f"p2p_session_id={p2p_session_id}, "
        f"task_id={task_id}, "
        f"rollout_transfer_session_id={rollout_transfer_session_id}, "
        f"rollout_ip={rollout_ip}, "
        f"rollout_ptr={rollout_ptr:#x}"
    )

    try:
        # # Get the weights to send (for now, use the first registered weights)
        # if not weight_buffers:
        #     raise RuntimeError("No weights registered")        
        # TODO(xinji1): per tensor level or per bucket level? should be per tensor level
        # TODO(xinji1): build a worker pool instead? 
        for weight_name, weight in converted_named_tensors:
            
            # TODO(xinji1): make them as the dict? so that there's no need to recompute them?
            src_ptr = weight.data_ptr()
            src_length = weight.numel() * weight.element_size()

            # Verify length matches
            if src_length != rollout_length:
                raise RuntimeError(
                    f"Length mismatch: src={src_length}, dst={rollout_length}"
                )

            # Validate that we have rollout_transfer_session_id
            if not rollout_transfer_session_id:
                raise RuntimeError(
                    "Missing rollout_transfer_session_id in sync_status message. "
                    "This is required for establishing RDMA connection."
                )

            logger.info(
                f"[Slime Weights Updating] Performing RDMA write: "
                f"src_ptr={src_ptr:#x} -> dst_ptr={rollout_ptr:#x}, "
                f"length={src_length}, "
                f"target_session={rollout_transfer_session_id}"
            )

            # Perform RDMA write using Mooncake transfer engine
            # Use rollout's Mooncake transfer_session_id for RDMA connection
            status = engine.transfer_sync(
                session_id=rollout_transfer_session_id,
                buffer=src_ptr,
                peer_buffer_address=rollout_ptr,
                length=src_length,
            )

            if status != 0:
                raise RuntimeError(f"RDMA transfer failed with status {status}")

            logger.info(
                f"[Slime Weights Updating] RDMA write completed successfully for task {task_id}"
            )

            # Send success confirmation back to rollout worker
            # ROUTER to DEALER: [identity, empty_delimiter, payload]
            response_data = zmq.utils.jsonapi.dumps({
                "type": "transfer_complete",
                "status": "success",
                "task_id": task_id,
            })
            router_socket.send_multipart([identity, b"", response_data])

            logger.info(f"[TrainingWeightSender] Sent success confirmation for task {task_id}")

    except Exception as e:
        logger.error(f"[TrainingWeightSender] Error handling sync_status: {e}", exc_info=True)

        # Send failure confirmation
        response_data = zmq.utils.jsonapi.dumps({
            "type": "transfer_complete",
            "status": "failed",
            "error": str(e),
            "task_id": task_id,
        })
        router_socket.send_multipart([identity, b"", response_data])

def update_weights_from_distributed_nccl(
    args: Namespace,
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_engines
    ]

    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs
