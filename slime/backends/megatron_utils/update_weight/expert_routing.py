import logging
import re
from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch.distributed as dist
from megatron.core import mpu

from slime.utils.distributed_utils import get_gloo_group
from slime.utils.types import ParamInfo

from .hf_weight_iterator_direct import pack_param_info_buckets

__all__ = ["configure_expert_routing"]


logger = logging.getLogger(__name__)

_ROUTED_EXPERT = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)")


@dataclass(frozen=True)
class _ExpertParam:
    info: ParamInfo
    layer: int
    expert: int
    target_ranks: tuple[int, ...]


@dataclass(frozen=True)
class _ExpertTransfer:
    source_rank: int
    target_ranks: tuple[int, ...]
    params: tuple[_ExpertParam, ...]


_ExpertTransferBatch = tuple[_ExpertTransfer, ...]
_ExpertTransferGroup = tuple[_ExpertTransferBatch, ...]


@dataclass(frozen=True)
class _SGLangMoeTopology:
    tp_size: int
    pp_size: int
    ep_size: int
    moe_dp_size: int


def _config_value(
    parallel_config: Mapping[str, Any] | None,
    key: str,
    default: Any,
) -> Any:
    if parallel_config is None:
        return default
    return parallel_config.get(key, parallel_config.get(key.replace("_", "-"), default))


def _get_sglang_moe_topology(
    args: Namespace,
    engine_gpu_count: int,
    parallel_config: Mapping[str, Any] | None = None,
) -> _SGLangMoeTopology:
    pp_size = int(_config_value(parallel_config, "pp_size", getattr(args, "sglang_pp_size", 1)))
    default_tp_size = engine_gpu_count // pp_size
    tp_size = int(_config_value(parallel_config, "tp_size", default_tp_size))
    ep_size = int(_config_value(parallel_config, "ep_size", getattr(args, "sglang_ep_size", 1)))
    moe_dp_size = int(_config_value(parallel_config, "moe_dp_size", getattr(args, "sglang_moe_dp_size", 1)))

    return _SGLangMoeTopology(
        tp_size=tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        moe_dp_size=moe_dp_size,
    )


def _sglang_topology_signature(topology: _SGLangMoeTopology) -> tuple[int, int, int, int]:
    return (
        topology.tp_size,
        topology.pp_size,
        topology.ep_size,
        topology.moe_dp_size,
    )


def _get_homogeneous_sglang_moe_topology(
    args: Namespace,
    engine_gpu_counts: Sequence[int],
    engine_parallel_configs: Sequence[Mapping[str, Any]] | None,
) -> _SGLangMoeTopology:
    if engine_parallel_configs is None:
        return _get_sglang_moe_topology(args, engine_gpu_count=engine_gpu_counts[0])
    if len(engine_parallel_configs) != len(engine_gpu_counts):
        raise ValueError(
            f"SGLang engine parallel config count {len(engine_parallel_configs)} "
            f"!= engine count {len(engine_gpu_counts)}"
        )

    topologies = [
        _get_sglang_moe_topology(args, engine_gpu_count=gpu_count, parallel_config=parallel_config)
        for gpu_count, parallel_config in zip(engine_gpu_counts, engine_parallel_configs, strict=True)
    ]
    signatures = {_sglang_topology_signature(topology) for topology in topologies}
    if len(signatures) != 1:
        raise ValueError(f"SGLang engines have heterogeneous parallel topology: {sorted(signatures)}")
    return topologies[0]


def _can_route_experts(
    args: Namespace,
    sglang_moe_topology: _SGLangMoeTopology,
    engine_gpu_counts: Sequence[int],
) -> bool:
    return (
        getattr(args, "megatron_to_hf_mode", "raw") == "raw"
        and sglang_moe_topology.pp_size == 1
        and sglang_moe_topology.ep_size > 1
        and not getattr(args, "sglang_enable_eplb", False)
        and getattr(args, "sglang_ep_num_redundant_experts", 0) == 0
        and getattr(args, "sglang_init_expert_location", "trivial") == "trivial"
        and not getattr(args, "sglang_enable_elastic_expert_backup", False)
        and mpu.get_expert_tensor_parallel_world_size() == 1
        and _sglang_moe_tp_is_one(engine_gpu_counts, sglang_moe_topology)
    )


def _sglang_moe_tp_is_one(
    engine_gpu_counts: Sequence[int],
    topology: _SGLangMoeTopology,
) -> bool:
    """Return whether each SGLang engine has no tensor parallelism inside experts."""
    if topology.pp_size != 1:
        return False
    expected_size = topology.ep_size * topology.moe_dp_size
    return all(gpu_count == expected_size for gpu_count in engine_gpu_counts)


def _get_expert_target_ranks(
    engine_gpu_counts: Sequence[int],
    engine_gpu_offsets: Sequence[int],
    *,
    ep_size: int,
    moe_dp_size: int,
    world_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Map each EP shard to colocated ranks; engine_size=EP*MoE-DP means MoE-TP=1."""
    expected_size = ep_size * moe_dp_size
    targets = [[] for _ in range(ep_size)]
    for gpu_count, gpu_offset in zip(engine_gpu_counts, engine_gpu_offsets, strict=True):
        if gpu_count != expected_size:
            raise ValueError(
                f"SGLang MoE TP must be 1, got engine_size={gpu_count}, EP={ep_size}, MoE-DP={moe_dp_size}"
            )
        if gpu_offset < 0 or gpu_offset + gpu_count > world_size:
            raise ValueError("SGLang engine is outside the Megatron world")
        for dp_rank in range(moe_dp_size):
            for ep_rank in range(ep_size):
                targets[ep_rank].append(gpu_offset + dp_rank * ep_size + ep_rank)
    return tuple(tuple(ranks) for ranks in targets)


def _build_expert_params(
    infos: Sequence[ParamInfo],
    target_ranks: Sequence[Sequence[int]],
    *,
    num_experts: int,
) -> list[_ExpertParam]:
    ep_size = len(target_ranks)
    if num_experts % ep_size:
        raise ValueError("num_experts must be divisible by SGLang EP")
    experts_per_rank = num_experts // ep_size
    coverage: dict[int, set[tuple[int, int]]] = defaultdict(set)
    params = []
    for info in infos:
        layer, projection, expert = map(int, _ROUTED_EXPERT.fullmatch(info.name).groups())
        if not 0 <= expert < num_experts:
            raise ValueError(f"invalid expert id {expert} in {info.name}")
        ep_rank = expert // experts_per_rank
        coverage[layer].add((expert, projection))
        params.append(
            _ExpertParam(
                info=info,
                layer=layer,
                expert=expert,
                target_ranks=tuple(target_ranks[ep_rank]),
            )
        )

    expected = {(expert, projection) for expert in range(num_experts) for projection in (1, 2)}
    if not coverage or any(found != expected for found in coverage.values()):
        raise ValueError("routed-expert metadata is incomplete")
    return sorted(params, key=lambda param: (param.layer, param.info.name))


def _set_expert_source_ranks(
    infos: Sequence[ParamInfo],
    local_names_by_rank: Sequence[Sequence[str]],
) -> list[ParamInfo]:
    owners = {}
    for rank, names in enumerate(local_names_by_rank):
        for name in names:
            owners.setdefault(name, rank)
    missing = [info.name for info in infos if info.name not in owners]
    if missing:
        raise ValueError(f"no physical owner for {missing[0]}")
    return [replace(info, src_rank=owners[info.name]) for info in infos]


def _resolve_expert_source_ranks(
    infos: Sequence[ParamInfo],
    get_local_weight_names: Callable[[], Iterable[str]],
) -> list[ParamInfo]:
    local_expert_names = tuple(name for name in get_local_weight_names() if _ROUTED_EXPERT.fullmatch(name))
    local_names_by_rank = [None] * dist.get_world_size()
    dist.all_gather_object(local_names_by_rank, local_expert_names, group=get_gloo_group())
    return _set_expert_source_ranks(infos, local_names_by_rank)


def _build_expert_transfer_plan(
    params: Sequence[_ExpertParam],
    buffer_size: int,
) -> list[_ExpertTransferGroup]:
    """Build expert transfer groups with pre-packed, rank-bounded transfer batches."""
    if buffer_size <= 0:
        raise ValueError("update_weight_buffer_size must be positive")

    params_by_transfer: dict[tuple[int, tuple[int, ...], int], list[_ExpertParam]] = defaultdict(list)
    for param in params:
        params_by_transfer[(param.layer, param.target_ranks, param.info.src_rank)].append(param)

    by_layer: dict[int, list[_ExpertTransfer]] = defaultdict(list)
    for (layer, target_ranks, source_rank), transfer_params in params_by_transfer.items():
        transfer = _ExpertTransfer(
            source_rank=source_rank,
            target_ranks=target_ranks,
            params=tuple(sorted(transfer_params, key=lambda param: (param.expert, param.info.name))),
        )
        by_layer[layer].append(transfer)

    transfer_plan = []
    for layer in sorted(by_layer):
        transfer_group = tuple(
            sorted(by_layer[layer], key=lambda transfer: (transfer.target_ranks, transfer.source_rank))
        )
        transfer_plan.append(tuple(_pack_expert_transfer_batches(transfer_group, buffer_size)))
    return transfer_plan


def _expert_transfer_size(transfer: _ExpertTransfer) -> int:
    return sum(param.info.size for param in transfer.params)


def _pack_expert_transfer_batches(
    transfers: Sequence[_ExpertTransfer],
    buffer_size: int,
) -> list[_ExpertTransferBatch]:
    """First-fit transfers while capping per-rank staging bytes."""
    sized_transfers = sorted(
        ((_expert_transfer_size(transfer), transfer) for transfer in transfers),
        key=lambda item: (-item[0], item[1].target_ranks, item[1].source_rank),
    )
    if buffer_size < sized_transfers[0][0]:
        raise ValueError("one source-to-target expert transfer bundle exceeds update_weight_buffer_size")

    batches: list[list[_ExpertTransfer]] = []
    batch_costs: list[dict[int, int]] = []
    for size, transfer in sized_transfers:
        participants = set(transfer.target_ranks) | {transfer.source_rank}
        candidates = [
            index
            for index, costs in enumerate(batch_costs)
            if all(costs.get(rank, 0) + size <= buffer_size for rank in participants)
        ]
        if candidates:
            batch_index = min(candidates, key=lambda index: (sum(batch_costs[index].values()), index))
        else:
            batch_index = len(batches)
            batches.append([])
            batch_costs.append({})

        batches[batch_index].append(transfer)
        for rank in participants:
            batch_costs[batch_index][rank] = batch_costs[batch_index].get(rank, 0) + size

    return [tuple(batch) for batch in batches]


def _log_disabled_expert_routing(reason: str) -> None:
    if dist.get_rank() == 0:
        logger.info("Disable rank-local expert update: %s", reason)


def configure_expert_routing(
    *,
    args: Namespace,
    full_param_info_buckets: Sequence[Sequence[ParamInfo]] | None,
    get_local_weight_names: Callable[[], Iterable[str]],
    engine_gpu_counts: Sequence[int],
    engine_gpu_offsets: Sequence[int],
    engine_parallel_configs: Sequence[Mapping[str, Any]] | None,
    use_distribute: bool,
) -> tuple[list[list[ParamInfo]] | None, list[_ExpertTransferGroup]]:
    if full_param_info_buckets is None:
        return None, []

    if use_distribute:
        _log_disabled_expert_routing("distributed SGLang engines are present")
        return None, []
    if not engine_gpu_counts:
        _log_disabled_expert_routing("no colocated SGLang engines")
        return None, []

    try:
        sglang_moe_topology = _get_homogeneous_sglang_moe_topology(
            args,
            engine_gpu_counts,
            engine_parallel_configs,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        _log_disabled_expert_routing(str(exc))
        return None, []

    if not _can_route_experts(
        args,
        sglang_moe_topology,
        engine_gpu_counts=engine_gpu_counts,
    ):
        _log_disabled_expert_routing("SGLang/Megatron expert topology is not eligible")
        return None, []
    dense_infos = []
    expert_infos = []
    for bucket in full_param_info_buckets:
        for info in bucket:
            (expert_infos if _ROUTED_EXPERT.fullmatch(info.name) else dense_infos).append(info)
    if not expert_infos:
        return None, []

    try:
        expert_infos = _resolve_expert_source_ranks(expert_infos, get_local_weight_names)
        target_ranks = _get_expert_target_ranks(
            engine_gpu_counts,
            engine_gpu_offsets,
            ep_size=sglang_moe_topology.ep_size,
            moe_dp_size=sglang_moe_topology.moe_dp_size,
            world_size=dist.get_world_size(),
        )
        expert_params = _build_expert_params(
            expert_infos,
            target_ranks,
            num_experts=args.num_experts,
        )
        buffer_size = args.update_weight_buffer_size
        expert_transfer_plan = _build_expert_transfer_plan(expert_params, buffer_size)
        expert_transfer_batches = sum(len(group) for group in expert_transfer_plan)
        dense_buckets = pack_param_info_buckets(dense_infos, buffer_size)
    except (AttributeError, TypeError, ValueError) as exc:
        _log_disabled_expert_routing(str(exc))
        return None, []

    if dist.get_rank() == 0:
        logger.info(
            "Enabled rank-local expert update: Megatron PP=%d EP=%d, SGLang EP=%d, "
            "MoE-DP=%d, %d -> %d transfer groups (%d dense + %d expert, %d expert transfer batches)",
            mpu.get_pipeline_model_parallel_world_size(),
            mpu.get_expert_model_parallel_world_size(),
            sglang_moe_topology.ep_size,
            sglang_moe_topology.moe_dp_size,
            len(full_param_info_buckets),
            len(dense_buckets) + len(expert_transfer_plan),
            len(dense_buckets),
            len(expert_transfer_plan),
            expert_transfer_batches,
        )
    return dense_buckets, expert_transfer_plan
