from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
import math
import logging
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


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
    ) -> None:
        """
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._delta_cache: dict[str, dict[str, Any]] = {}
        self._delta_threshold = getattr(args, "delta_threshold", 0.3)
        self._delta_block_size = self._resolve_delta_block_size(args, quantization_config)
        self._delta_enabled = (
            getattr(args, "weight_update_mode", "full") == "delta"
            and quantization_config is not None
            and quantization_config.get("quant_method") == "fp8"
        )

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

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
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            del long_lived_tensors

        dist.barrier(group=get_gloo_group())

        # int4/fp4 post_process
        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        if self._delta_enabled:
            refs_colocated, long_lived_tensors = self._send_hf_params_delta(hf_named_tensors)
        else:
            refs_colocated, long_lived_tensors = _send_to_colocated_engine(
                hf_named_tensors,
                ipc_engine=self._ipc_engine,
                ipc_gather_src=self._ipc_gather_src,
                ipc_gather_group=self._ipc_gather_group,
                weight_version=self.weight_version,
            )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors

    def _resolve_delta_block_size(self, args: Namespace, quantization_config: dict[str, Any] | None):
        block_size = getattr(args, "delta_block_size", None)
        if block_size:
            if len(block_size) == 1:
                return None
            if len(block_size) >= 2:
                return (block_size[0], block_size[1])
        if quantization_config is not None:
            block_size = quantization_config.get("weight_block_size")
            if block_size:
                return (block_size[0], block_size[1])
        return None

    def _send_hf_params_delta(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        name_to_tensor = {name: tensor for name, tensor in hf_named_tensors}
        processed = set()
        full_named_tensors: list[tuple[str, torch.Tensor]] = []
        delta_items: list[dict[str, Any]] = []
        total_blocks = 0
        delta_blocks = 0
        full_update_blocks = 0
        skipped_blocks = 0
        total_bytes = 0
        delta_bytes = 0

        for name, tensor in hf_named_tensors:
            if name in processed:
                continue
            if tensor.dtype == torch.float8_e4m3fn and name.endswith(".weight"):
                scale_name = self._get_fp8_scale_name(name, name_to_tensor)
                scale_tensor = name_to_tensor.get(scale_name) if scale_name else None
                if scale_tensor is None:
                    full_named_tensors.append((name, tensor))
                    processed.add(name)
                    continue
                assert scale_name is not None

                delta_payload, curr_qweight, curr_scale = self._build_fp8_delta_payload(
                    name, tensor, scale_name, scale_tensor
                )
                self._update_delta_cache(name, curr_qweight, scale_name, curr_scale)
                processed.add(name)
                processed.add(scale_name)

                if delta_payload is None:
                    full_named_tensors.append((name, tensor))
                    full_named_tensors.append((scale_name, scale_tensor))
                elif delta_payload.get("empty"):
                    skipped_blocks += delta_payload.get("total_blocks", 0)
                    continue
                else:
                    delta_items.append(delta_payload)
                    total_blocks += delta_payload.get("total_blocks", 0)
                    delta_blocks += delta_payload.get("delta_blocks", 0)
                    if delta_payload.get("full_update"):
                        full_update_blocks += delta_payload.get("total_blocks", 0)
                    total_bytes += self._payload_total_bytes(delta_payload)
                    delta_bytes += self._payload_delta_bytes(delta_payload)
            else:
                full_named_tensors.append((name, tensor))
                processed.add(name)

        all_refs: list[ObjectRef] = []
        long_lived_tensors = []
        if full_named_tensors:
            refs_full, long_lived_tensors = _send_to_colocated_engine(
                full_named_tensors,
                ipc_engine=self._ipc_engine,
                ipc_gather_src=self._ipc_gather_src,
                ipc_gather_group=self._ipc_gather_group,
                weight_version=self.weight_version,
            )
            all_refs.extend(refs_full)

        refs_delta = _send_delta_to_colocated_engine(
            delta_items,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        all_refs.extend(refs_delta)

        if dist.get_rank() == self._ipc_gather_src:
            ratio = (delta_blocks / total_blocks) if total_blocks else 0.0
            bytes_ratio = (delta_bytes / total_bytes) if total_bytes else 0.0
            savings_bytes = total_bytes - delta_bytes
            logger.info(
                "delta update stats: delta_blocks=%d total_blocks=%d ratio=%.4f bytes_ratio=%.4f "
                "savings_mb=%.2f full_blocks=%d skipped_blocks=%d full_tensors=%d",
                delta_blocks,
                total_blocks,
                ratio,
                bytes_ratio,
                savings_bytes / (1024 * 1024),
                full_update_blocks,
                skipped_blocks,
                len(full_named_tensors),
            )

        return all_refs, long_lived_tensors

    def _get_fp8_scale_name(self, weight_name: str, name_to_tensor: dict[str, torch.Tensor]) -> str | None:
        scale_name_inv = weight_name.replace(".weight", ".weight_scale_inv")
        if scale_name_inv in name_to_tensor:
            return scale_name_inv
        scale_name = weight_name.replace(".weight", ".weight_scale")
        if scale_name in name_to_tensor:
            return scale_name
        return None

    def _build_fp8_delta_payload(
        self,
        weight_name: str,
        qweight: torch.Tensor,
        scale_name: str,
        scale: torch.Tensor,
    ) -> tuple[dict[str, Any] | None, torch.Tensor, torch.Tensor]:
        curr_qweight = qweight.detach().cpu()
        curr_scale = scale.detach().cpu()
        prev = self._delta_cache.get(weight_name)
        if prev is None or (
            prev["block_size"] != self._delta_block_size
            or prev["qweight"].shape != curr_qweight.shape
            or prev["scale"].shape != curr_scale.shape
        ):
            return self._build_full_fp8_delta_payload(weight_name, scale_name, curr_qweight, curr_scale)

        if self._delta_block_size is None:
            if torch.equal(curr_qweight, prev["qweight"]) and torch.equal(curr_scale, prev["scale"]):
                return {"empty": True, "total_blocks": 1}, curr_qweight, curr_scale
            return (
                {
                    "weight_name": weight_name,
                    "scale_name": scale_name,
                    "full_shape": tuple(curr_qweight.shape),
                    "scale_shape": tuple(curr_scale.shape),
                    "block_size": None,
                    "block_indices": [0],
                    "block_shapes": [tuple(curr_qweight.shape)],
                    "qweight_blocks": [curr_qweight],
                    "scale_blocks": curr_scale,
                    "total_blocks": 1,
                    "delta_blocks": 1,
                },
                curr_qweight,
                curr_scale,
            )

        if curr_qweight.ndim != 2 or curr_scale.ndim != 2:
            return self._build_full_fp8_delta_payload(weight_name, scale_name, curr_qweight, curr_scale)

        block_m, block_n = self._delta_block_size
        grid_m = math.ceil(curr_qweight.shape[0] / block_m)
        grid_n = math.ceil(curr_qweight.shape[1] / block_n)
        if curr_scale.shape != (grid_m, grid_n):
            return self._build_full_fp8_delta_payload(weight_name, scale_name, curr_qweight, curr_scale)

        block_indices = []
        block_shapes = []
        qweight_blocks = []
        scale_blocks = []
        total_blocks = grid_m * grid_n

        for block_row in range(grid_m):
            row_start = block_row * block_m
            row_end = min(curr_qweight.shape[0], row_start + block_m)
            for block_col in range(grid_n):
                col_start = block_col * block_n
                col_end = min(curr_qweight.shape[1], col_start + block_n)
                curr_block = curr_qweight[row_start:row_end, col_start:col_end]
                prev_block = prev["qweight"][row_start:row_end, col_start:col_end]
                if torch.equal(curr_block, prev_block) and torch.equal(
                    curr_scale[block_row, block_col], prev["scale"][block_row, block_col]
                ):
                    continue
                block_indices.append(block_row * grid_n + block_col)
                block_shapes.append((row_end - row_start, col_end - col_start))
                qweight_blocks.append(curr_block)
                scale_blocks.append(curr_scale[block_row, block_col])

        if not block_indices:
            return {"empty": True, "total_blocks": total_blocks}, curr_qweight, curr_scale

        if len(block_indices) / total_blocks > self._delta_threshold:
            return self._build_full_fp8_delta_payload(weight_name, scale_name, curr_qweight, curr_scale)

        scale_blocks_tensor = torch.stack(scale_blocks)
        return (
            {
                "weight_name": weight_name,
                "scale_name": scale_name,
                "full_shape": tuple(curr_qweight.shape),
                "scale_shape": tuple(curr_scale.shape),
                "block_size": (block_m, block_n),
                "block_indices": block_indices,
                "block_shapes": block_shapes,
                "qweight_blocks": qweight_blocks,
                "scale_blocks": scale_blocks_tensor,
                "total_blocks": total_blocks,
                "delta_blocks": len(block_indices),
            },
            curr_qweight,
            curr_scale,
        )

    def _build_full_fp8_delta_payload(
        self,
        weight_name: str,
        scale_name: str,
        curr_qweight: torch.Tensor,
        curr_scale: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
        if self._delta_block_size is None or curr_qweight.ndim != 2:
            return (
                {
                    "weight_name": weight_name,
                    "scale_name": scale_name,
                    "full_shape": tuple(curr_qweight.shape),
                    "scale_shape": tuple(curr_scale.shape),
                    "block_size": None,
                    "block_indices": [0],
                    "block_shapes": [tuple(curr_qweight.shape)],
                    "qweight_blocks": [curr_qweight],
                    "scale_blocks": curr_scale,
                    "total_blocks": 1,
                    "delta_blocks": 1,
                    "full_update": True,
                },
                curr_qweight,
                curr_scale,
            )

        block_m, block_n = self._delta_block_size
        grid_m = math.ceil(curr_qweight.shape[0] / block_m)
        grid_n = math.ceil(curr_qweight.shape[1] / block_n)
        block_indices = []
        block_shapes = []
        qweight_blocks = []
        scale_blocks = []
        for block_row in range(grid_m):
            row_start = block_row * block_m
            row_end = min(curr_qweight.shape[0], row_start + block_m)
            for block_col in range(grid_n):
                col_start = block_col * block_n
                col_end = min(curr_qweight.shape[1], col_start + block_n)
                block_indices.append(block_row * grid_n + block_col)
                block_shapes.append((row_end - row_start, col_end - col_start))
                qweight_blocks.append(curr_qweight[row_start:row_end, col_start:col_end])
                if curr_scale.ndim == 2 and block_row < curr_scale.shape[0] and block_col < curr_scale.shape[1]:
                    scale_blocks.append(curr_scale[block_row, block_col])
        scale_blocks_tensor = torch.stack(scale_blocks) if scale_blocks else curr_scale.flatten()[:1]
        return (
            {
                "weight_name": weight_name,
                "scale_name": scale_name,
                "full_shape": tuple(curr_qweight.shape),
                "scale_shape": tuple(curr_scale.shape),
                "block_size": (block_m, block_n),
                "block_indices": block_indices,
                "block_shapes": block_shapes,
                "qweight_blocks": qweight_blocks,
                "scale_blocks": scale_blocks_tensor,
                "total_blocks": len(block_indices),
                "delta_blocks": len(block_indices),
                "full_update": True,
            },
            curr_qweight,
            curr_scale,
        )

    def _update_delta_cache(
        self, weight_name: str, curr_qweight: torch.Tensor, scale_name: str, curr_scale: torch.Tensor
    ) -> None:
        self._delta_cache[weight_name] = {
            "qweight": curr_qweight.clone(),
            "scale": curr_scale.clone(),
            "scale_name": scale_name,
            "block_size": self._delta_block_size,
        }

    def _payload_total_bytes(self, payload: dict[str, Any]) -> int:
        qweight_blocks = payload.get("qweight_blocks", [])
        if not qweight_blocks:
            return 0
        full_shape = payload.get("full_shape")
        if not full_shape:
            return 0
        qweight_size = int(torch.tensor(full_shape).prod().item()) * qweight_blocks[0].element_size()
        scale_shape = payload.get("scale_shape")
        scale_blocks = payload.get("scale_blocks")
        if scale_shape is None or scale_blocks is None:
            return qweight_size
        scale_size = int(torch.tensor(scale_shape).prod().item()) * scale_blocks.element_size()
        return qweight_size + scale_size

    def _payload_delta_bytes(self, payload: dict[str, Any]) -> int:
        qweight_blocks = payload.get("qweight_blocks", [])
        qweight_bytes = sum(block.numel() * block.element_size() for block in qweight_blocks)
        scale_blocks = payload.get("scale_blocks")
        scale_bytes = scale_blocks.numel() * scale_blocks.element_size() if scale_blocks is not None else 0
        return qweight_bytes + scale_bytes


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    # TODO improve
    long_live_tensors = []

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = flattened_tensor_bucket.get_metadata()
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": metadata,
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if dist.get_rank() == ipc_gather_src:
        # TODO: here we assume all ranks have the same number of dtypes, not sure if that is correct.
        num_dtypes = len(serialized_named_tensors[0])
        for i in range(num_dtypes):
            kwargs = {
                "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                "load_format": "flattened_bucket",
                "weight_version": str(weight_version),
            }
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors


def _send_delta_to_colocated_engine(
    delta_items: list[dict[str, Any]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> list[ObjectRef]:
    serialized_delta = MultiprocessingSerializer.serialize({"deltas": delta_items}, output_str=True)
    serialized_payloads = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_delta,
        object_gather_list=serialized_payloads,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs: list[ObjectRef] = []
    if dist.get_rank() == ipc_gather_src:
        refs.append(
            ipc_engine.update_weights_from_tensor_delta.remote(
                serialized_delta_payloads=serialized_payloads,
                weight_version=str(weight_version),
            )
        )
    return refs
