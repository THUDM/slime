from __future__ import annotations

import socket
import time
from argparse import Namespace
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from slime.utils import accelerator
from slime.utils.distributed_utils import get_gloo_group, init_process_group
from slime.utils.engine_group_wave import build_engine_group_waves, run_engine_group_waves
from slime.utils.http_utils import _wrap_ipv6

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, named_params_and_buffers


@dataclass(frozen=True)
class DistributedWeightUpdateGroup:
    """One trainer process group and the rollout engines that join it."""

    engine_indices: tuple[int, ...]
    group_name: str
    process_group: dist.ProcessGroup
    rollout_engines: tuple[ActorHandle, ...]


class UpdateWeightFromDistributed:
    """
    Update distributed engines through a device process group. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 transfers. Non-expert (TP) and expert (EP) params separate.
    Subclasses override ``_send_weights`` / ``_on_chunk`` to inject per-mode behaviour.
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
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups: list[DistributedWeightUpdateGroup] = []
        self.update_weight_metrics: dict[str, float] = {}

    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Drained by the actor onto the rollout/step log.
        """
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        """
        Create "slime-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent transfers.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if self._model_update_groups:
                disconnect_rollout_engine_groups_from_distributed(self._model_update_groups)
            self._model_update_groups = connect_rollout_engine_groups_from_distributed(
                self.args,
                self._group_name,
                rollout_engines,
                engine_gpu_counts=engine_gpu_counts,
                max_inflight_engine_groups=getattr(self.args, "update_weight_max_inflight_engine_groups", 0),
            )

    def disconnect_rollout_engines(self) -> None:
        if not getattr(self, "_is_pp_src_rank", False) or not self._model_update_groups:
            return
        disconnect_rollout_engine_groups_from_distributed(self._model_update_groups)
        self._model_update_groups = []

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → flush → _send_weights → continue. Progress on PP source.
        """
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            # int4/fp4 pre_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                    max_inflight_engine_groups=getattr(self.args, "update_weight_max_inflight_engine_groups", 0),
                )
        dist.barrier(group=get_gloo_group())

        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None
        self._send_weights(pbar)

        if dist.get_rank() == 0:
            # int4/fp4 post_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                    max_inflight_engine_groups=getattr(self.args, "update_weight_max_inflight_engine_groups", 0),
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_weights(self, pbar: tqdm | None) -> None:
        """
        Non-expert (TP) pass → barrier → expert (EP) pass → barrier. Each iterator
        yields broadcast-ready chunks (bucketing happens internally); subclasses
        override ``_on_chunk`` to inject per-chunk behaviour.
        """
        for chunk_iter in (self._iter_non_expert_chunks(), self._iter_expert_chunks()):
            for hf_chunk in chunk_iter:
                self._on_chunk(hf_chunk)
                self._update_bucket_weights_from_distributed(hf_chunk, pbar=pbar)
            dist.barrier(group=get_gloo_group())

    def _on_chunk(self, hf_chunk: list[tuple[str, torch.Tensor]]) -> None:
        """
        Hook for each HF chunk in ``_send_weights`` before its broadcast. No-op by default.
        """

    def _iter_non_expert_chunks(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """
        Yield broadcast-sized HF chunks of non-expert params: TP all-gather +
        HF convert per param, then bucket up to ``--update-weight-buffer-size``.
        Empty on non-PP-src ranks (they still join all_gather_param).
        """
        buffer_size = 0
        buffer: list[tuple[str, torch.Tensor]] = []
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            param = all_gather_param(name, param)
            if not self._is_pp_src_rank:
                continue
            hf_chunk = convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
            chunk_bytes = sum(t.numel() * t.element_size() for _, t in hf_chunk)
            if buffer and buffer_size + chunk_bytes > self.args.update_weight_buffer_size:
                yield buffer
                buffer = []
                buffer_size = 0
            buffer.extend(hf_chunk)
            buffer_size += chunk_bytes
        if buffer:
            yield buffer

    def _iter_expert_chunks(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """
        Yield one HF chunk per EP-weighted batch of expert params: TP gather +
        buffer until threshold, then EP gather + HF convert.
        """
        params = ((n, p) for n, p in named_params_and_buffers(self.args, self.model) if ".experts." in n)
        buffer_size = 0
        batch: list[tuple[str, torch.Tensor]] = []
        for name, param in params:
            param = all_gather_param(name, param)
            param_size = param.numel() * param.element_size()
            if (
                buffer_size + param_size
            ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
                hf_chunk = self._ep_gather_and_convert(batch)
                if hf_chunk:
                    yield hf_chunk
                batch = []
                buffer_size = 0
            batch.append((name, param))
            buffer_size += param_size
        if batch:
            hf_chunk = self._ep_gather_and_convert(batch)
            if hf_chunk:
                yield hf_chunk

    def _ep_gather_and_convert(self, named_tensors: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
        """
        EP all-gather a buffered batch + HF convert on PP source. Returns HF tensors on
        PP source, [] elsewhere. Clears ``named_tensors``.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=accelerator.current_device())
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
            return []

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        return converted_hf_tensors

    def _update_bucket_weights_from_distributed(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
        load_format: str | None = None,
    ) -> None:
        """
        Lock → transfer → clear → unlock → pbar++. Lock prevents communication deadlock.
        """
        # Lock the rollout engines to prevent communication deadlock.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        try:
            update_weights_in_engine_group_waves(
                self._model_update_groups,
                self.weight_version,
                converted_named_tensors,
                max_inflight_engine_groups=getattr(self.args, "update_weight_max_inflight_engine_groups", 0),
                load_format=load_format,
            )
        finally:
            ray.get(self.rollout_engine_lock.release.remote())
        converted_named_tensors.clear()
        if pbar is not None:
            pbar.update(1)


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
) -> dist.ProcessGroup:
    """
    Create a device process group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the process group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)

    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1  # +1 for training rank 0

    # Compute cumulative rank offsets: engine i starts at cumulative[i] + 1.
    cumulative = [0]
    for c in engine_gpu_counts:
        cumulative.append(cumulative[-1] + c)

    backend = accelerator.weight_update_backend()
    refs = [
        engine.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=cumulative[i] + 1,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        for i, engine in enumerate(rollout_engines)
    ]
    model_update_groups = init_process_group(
        backend=backend,
        init_method=f"tcp://{_wrap_ipv6(master_address)}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups


def connect_rollout_engine_groups_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
    *,
    max_inflight_engine_groups: int = 0,
    total_engine_groups: int | None = None,
    engine_index_offset: int = 0,
) -> list[DistributedWeightUpdateGroup]:
    """Create aggregate or per-engine process groups for weight-update waves.

    The aggregate group preserves the existing data path when every engine may
    run together. A bounded policy needs one process group per logical engine;
    otherwise an aggregate NCCL broadcast would still require every engine to
    enter the collective at once and could not be admitted in waves.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)
    if len(engine_gpu_counts) != len(rollout_engines):
        raise ValueError("engine_gpu_counts must have one entry per rollout engine")
    if any(gpu_count <= 0 for gpu_count in engine_gpu_counts):
        raise ValueError("engine GPU counts must be positive")
    if max_inflight_engine_groups < 0:
        raise ValueError("max_inflight_engine_groups must be non-negative")
    if not rollout_engines:
        return []

    total = total_engine_groups if total_engine_groups is not None else len(rollout_engines)
    if total < len(rollout_engines):
        raise ValueError("total_engine_groups cannot be smaller than the distributed engine count")
    if engine_index_offset < 0 or engine_index_offset + len(rollout_engines) > total:
        raise ValueError("distributed engine indices must fit within total_engine_groups")
    use_waves = 0 < max_inflight_engine_groups < total

    if not use_waves:
        process_group = connect_rollout_engines_from_distributed(
            args,
            group_name,
            rollout_engines,
            engine_gpu_counts=engine_gpu_counts,
        )
        return [
            DistributedWeightUpdateGroup(
                engine_indices=tuple(range(engine_index_offset, engine_index_offset + len(rollout_engines))),
                group_name=group_name,
                process_group=process_group,
                rollout_engines=tuple(rollout_engines),
            )
        ]

    update_groups = []
    try:
        for local_index, (engine, gpu_count) in enumerate(zip(rollout_engines, engine_gpu_counts, strict=True)):
            engine_index = engine_index_offset + local_index
            engine_group_name = f"{group_name}-engine-{engine_index}"
            process_group = connect_rollout_engines_from_distributed(
                args,
                engine_group_name,
                [engine],
                engine_gpu_counts=[gpu_count],
            )
            update_groups.append(
                DistributedWeightUpdateGroup(
                    engine_indices=(engine_index,),
                    group_name=engine_group_name,
                    process_group=process_group,
                    rollout_engines=(engine,),
                )
            )
    except Exception:
        disconnect_rollout_engine_groups_from_distributed(update_groups)
        raise
    return update_groups


def disconnect_rollout_engines_from_distributed(group_name, model_update_groups, rollout_engines):
    """
    Destroy the weight-update process group on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    dist.destroy_process_group(model_update_groups)
    ray.get(refs)


def disconnect_rollout_engine_groups_from_distributed(
    update_groups: Sequence[DistributedWeightUpdateGroup],
) -> None:
    """Destroy all process groups created for aggregate or waved updates."""
    for update_group in update_groups:
        disconnect_rollout_engines_from_distributed(
            update_group.group_name,
            update_group.process_group,
            update_group.rollout_engines,
        )


def launch_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    load_format: str | None = None,
) -> tuple[list[ObjectRef], list[dist.Work]]:
    """Launch engine receives and broadcasts without waiting for completion."""
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
            load_format=load_format,
        )
        for engine in rollout_engines
    ]
    handles = [dist.broadcast(param.data, 0, group=group, async_op=True) for _, param in converted_named_tensors]
    return refs, handles


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    load_format: str | None = None,
) -> list[ObjectRef]:
    """
    Send metadata through Ray and tensors through the configured transport.
    """
    refs, handles = launch_weights_from_distributed(
        group_name,
        group,
        weight_version,
        rollout_engines,
        converted_named_tensors,
        load_format=load_format,
    )
    for handle in handles:
        handle.wait()

    return refs


def update_weights_in_engine_group_waves(
    update_groups: Sequence[DistributedWeightUpdateGroup],
    weight_version: int,
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    max_inflight_engine_groups: int,
    load_format: str | None = None,
) -> list[ObjectRef]:
    """Transfer one bucket to bounded waves of independent engine groups."""
    all_refs = []
    for wave in build_engine_group_waves(update_groups, max_inflight_engine_groups):
        wave_refs = []
        wave_handles = []
        for _index, update_group in wave:
            refs, handles = launch_weights_from_distributed(
                update_group.group_name,
                update_group.process_group,
                weight_version,
                update_group.rollout_engines,
                converted_named_tensors,
                load_format=load_format,
            )
            wave_refs.extend(refs)
            wave_handles.extend(handles)
        for handle in wave_handles:
            handle.wait()
        if wave_refs:
            ray.get(wave_refs)
        all_refs.extend(wave_refs)
    return all_refs


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
    max_inflight_engine_groups: int = 0,
):
    """
    Trigger post-process for int4/fp4 quantization on all rollout engines.
    """
    run_engine_group_waves(
        rollout_engines,
        max_inflight_engine_groups,
        lambda _index, engine: engine.post_process_weights.remote(
            restore_weights_before_load=restore_weights_before_load,
            post_process_quantization=post_process_quantization,
        ),
        ray.get,
    )
