import collections
import inspect
import re
import socket
import time
from argparse import Namespace
from collections.abc import Iterator, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from ray import ObjectRef
from ray.actor import ActorHandle

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
        weights: Mapping[str, Mapping[str, torch.Tensor]],
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
        self.weights = weights
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

        num_buckets = len(self.param_info_buckets)

        # Define the window size for the pipeline. This controls how many update tasks
        # can be in-flight simultaneously to manage memory usage and concurrency.
        pipeline_window_size = 4

        ray_refs = collections.deque()

        # Create a separate CUDA stream for pre-fetching the next bucket of parameters.
        side_stream = torch.cuda.Stream()

        current_params, current_infos = self._gather_bucket_params(self.param_info_buckets[0])

        for i in tqdm(range(num_buckets), disable=rank != 0, desc="Update weights (pipelined)"):
            if i + 1 < num_buckets:
                with torch.cuda.stream(side_stream):
                    next_params, next_infos = self._gather_bucket_params(self.param_info_buckets[i + 1])

            refs = self._update_converted_params_from_tensor(current_params, current_infos)
            ray_refs.append(refs)

            # Prevents OOM errors and ensures smooth pipeline execution
            if rank == self._ipc_gather_src:
                while len(ray_refs) >= pipeline_window_size:
                    oldest_refs_list = ray_refs.popleft()
                    if oldest_refs_list:
                        ray.get(oldest_refs_list)

            if i + 1 < num_buckets:
                torch.cuda.current_stream().wait_stream(side_stream)
                current_params = next_params
                current_infos = next_infos

        if rank == self._ipc_gather_src and ray_refs:
            all_remaining_refs = sum(ray_refs, [])
            if all_remaining_refs:
                ray.get(all_remaining_refs)

        dist.barrier(group=get_gloo_group())

    def _gather_bucket_params(
        self, param_infos: Sequence[ParamInfo]
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
                        self.weights["actor"][info.name].to(device=torch.cuda.current_device(), non_blocking=True),
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


class UpdateWeightFromDistributed:
    """
    Update distributed engines via NCCL. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights: Mapping[str, Mapping[str, torch.Tensor]],
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

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Create NCCL "slime-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        # For TP:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self._group_name, rollout_engines
            )

    @torch.no_grad()
    def update_weights(self) -> None:
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


def connect_rollout_engines_from_distributed(
    args: Namespace, group_name: str, rollout_engines: Sequence[ActorHandle]
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.
    """
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = len(rollout_engines) * args.rollout_num_gpus_per_engine + 1

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
    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
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
