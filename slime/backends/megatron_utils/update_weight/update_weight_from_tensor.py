from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from time import perf_counter
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from slime.observability.communication_timeline import communication_phase
from slime.observability.weight_sync_trace import trace_weight_update
from slime.utils import accelerator
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.engine_group_wave import build_engine_group_waves
from slime.utils.types import ParamInfo

from ..megatron_to_hf import convert_to_hf
from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .expert_routing import configure_expert_routing
from .hf_weight_iterator_direct import HfWeightIteratorDirect
from .update_weight_from_distributed import (
    connect_rollout_engine_groups_from_distributed,
    disconnect_rollout_engine_groups_from_distributed,
    launch_weights_from_distributed,
    post_process_weights,
    synchronize_weight_transfer,
)
from .weight_sync_credit import WeightBucketReservation, WeightSyncCreditController


def _build_flattened_tensor_data(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> dict[str, Any]:
    if not named_tensors:
        return {
            "flattened_tensor": torch.empty(0, dtype=torch.uint8, device=accelerator.current_device()),
            "metadata": [],
        }

    # Do not reuse the IPC-facing flattened tensor. SGLang returns from the
    # HTTP/Ray request after enqueueing GPU copies into model weights, but it
    # does not guarantee a CUDA-device sync before the response. Reusing and
    # overwriting the same producer buffer immediately after ray.get can race
    # with the consumer-side copy and corrupt weights.
    flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    return {
        "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
        "metadata": flattened_tensor_bucket.get_metadata(),
    }


def _tensor_tree_nbytes(value: Any) -> int:
    """Count tensor storage represented by a nested staging object."""
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())
    if isinstance(value, Mapping):
        return sum(_tensor_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_nbytes(item) for item in value)
    return 0


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
        Compute param buckets.  IPC Gloo groups are created later in
        ``connect_rollout_engines`` once ``engine_gpu_counts`` is known.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.rank = dist.get_rank()
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}
        self._weight_sync_credit = WeightSyncCreditController(
            max_inflight_buckets=getattr(args, "update_weight_max_inflight_buckets", 0),
            max_inflight_bytes=getattr(args, "update_weight_max_inflight_bytes", 0),
        )

        self._hf_weight_iterator = HfWeightIteratorDirect(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )
        param_info_buckets = getattr(self._hf_weight_iterator, "megatron_local_param_info_buckets", None)
        self._full_param_info_buckets = (
            tuple(tuple(bucket) for bucket in param_info_buckets) if param_info_buckets is not None else None
        )
        self._non_expert_param_info_buckets: list[list[ParamInfo]] | None = None

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._ipc_engine_index = None
        self._model_update_groups = []
        self._total_engine_groups = 0
        self._max_inflight_engine_groups = getattr(args, "update_weight_max_inflight_engine_groups", 0)
        self._wave_scheduling_enabled = False
        self._expert_transfer_plan = []

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self._all_rollout_engines = rollout_engines
        self.rollout_engines = rollout_engines
        self._total_engine_groups = len(rollout_engines)
        self._max_inflight_engine_groups = getattr(self.args, "update_weight_max_inflight_engine_groups", 0)
        self._wave_scheduling_enabled = 0 < self._max_inflight_engine_groups < self._total_engine_groups

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "slime"
            if self._is_distributed_src_rank:
                if self._model_update_groups:
                    disconnect_rollout_engine_groups_from_distributed(self._model_update_groups)

                self._model_update_groups = connect_rollout_engine_groups_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                    max_inflight_engine_groups=self._max_inflight_engine_groups,
                    total_engine_groups=self._total_engine_groups,
                    engine_index_offset=colocate_engine_nums,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]
        colocate_parallel_configs = (
            engine_parallel_configs[:colocate_engine_nums] if engine_parallel_configs is not None else None
        )

        # Create IPC Gloo gather groups (only on first call; partitioning is
        # fixed across reconnects).
        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if self.rank in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = colocate_gpu_offsets[i]

        # Map training ranks to colocated engine actors.
        self._ipc_engine_index = None
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= self.rank < end:
                self._ipc_engine = engine
                self._ipc_engine_index = i

        self._non_expert_param_info_buckets, self._expert_transfer_plan = configure_expert_routing(
            args=self.args,
            full_param_info_buckets=self._full_param_info_buckets,
            get_local_weight_names=self.weights_getter,
            engine_gpu_counts=colocate_gpu_counts,
            engine_gpu_offsets=colocate_gpu_offsets,
            engine_parallel_configs=colocate_parallel_configs,
            use_distribute=self.use_distribute,
        )

    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Empty under colocate today;
        kept symmetric with UpdateWeightFromDistributed so the actor can drain unconditionally.
        """
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    def _consumer_engines(self) -> Sequence[ActorHandle]:
        """Return colocated and distributed consumers as one lifecycle unit."""
        return getattr(self, "_all_rollout_engines", self.rollout_engines)

    def _wait_for_bucket_completion(
        self,
        reservation: WeightBucketReservation,
        handles: Sequence[Any],
        refs: Sequence[ObjectRef],
        named_tensors: Sequence[tuple[str, torch.Tensor]] = (),
    ) -> None:
        """Wait for transport/load and share the engine acknowledgement within its TP group."""
        local_error: Exception | None = None
        trace = getattr(self, "_weight_sync_trace", None)
        if trace is not None and reservation.bucket_id not in trace.pending:
            trace = None  # A waved submission already waited and recorded its consumers.
        try:
            for handle in handles:
                handle.wait()
            if handles:
                synchronize_weight_transfer(named_tensors)
            if trace is not None:
                with trace.consumer(reservation) as phase:
                    ray.get(refs)
                    phase.mark_consumer()
                # CUDA IPC has no separate transport Work; its first observed
                # completion remains the engine receive/load acknowledgement.
                trace.complete(reservation)
            else:
                ray.get(refs)
        except Exception as error:
            if trace is not None:
                trace.complete(reservation, error)
            local_error = error

        group_error: str | None = None
        if self._ipc_gather_group is not None:
            error_message = (
                f"{type(local_error).__name__}: {local_error}"
                if self.rank == self._ipc_gather_src and local_error is not None
                else None
            )
            status = [error_message]
            dist.broadcast_object_list(
                status,
                src=self._ipc_gather_src,
                group=self._ipc_gather_group,
            )
            group_error = status[0]

        if local_error is not None:
            raise local_error
        if group_error is not None:
            raise RuntimeError(f"weight consumer failed on engine source rank: {group_error}")

        # For colocated CUDA IPC, the Ray response is the first observable
        # receive/load acknowledgement. Do not claim an earlier transport
        # boundary when there is no separate Work handle.
        self._weight_sync_credit.mark_transport_complete(reservation)
        self._weight_sync_credit.mark_consumers_complete(reservation)

    def _mark_bucket_launched(
        self,
        reservation: WeightBucketReservation,
        refs: Sequence[ObjectRef],
        handles: Sequence[Any],
        long_lived_tensors: Any,
    ) -> None:
        self._weight_sync_credit.mark_launched(
            reservation,
            transport_bytes=reservation.bucket_bytes if refs or handles else 0,
            staging_bytes=_tensor_tree_nbytes(long_lived_tensors),
            consumer_objects=len(refs),
        )

    def _prepare_expert_weight_batch(
        self,
        transfers: Sequence[Any],
        megatron_local_weights: Mapping[str, torch.Tensor],
        staging_buffers: dict[tuple[torch.dtype, tuple[int, ...]], list[torch.Tensor]],
    ) -> list[tuple[str, torch.Tensor]]:
        local_params = []
        p2p_ops = []
        buffer_offsets: dict[tuple[torch.dtype, tuple[int, ...]], int] = defaultdict(int)
        for transfer in transfers:
            for expert_param in transfer.params:
                info = expert_param.info
                if self.rank != transfer.source_rank and self.rank not in transfer.target_ranks:
                    continue
                key = (info.dtype, tuple(info.shape))
                pool = staging_buffers.setdefault(key, [])
                offset = buffer_offsets[key]
                buffer_offsets[key] = offset + 1
                if offset == len(pool):
                    pool.append(torch.empty(info.shape, dtype=info.dtype, device=accelerator.device()))
                tensor = pool[offset]
                if self.rank == transfer.source_rank:
                    source = megatron_local_weights[info.name]
                    if source.shape != info.shape or source.dtype != info.dtype:
                        raise ValueError(f"expert metadata changed for {info.name}")
                    tensor.copy_(source, non_blocking=True)
                    p2p_ops.extend(
                        dist.P2POp(dist.isend, tensor, target_rank)
                        for target_rank in transfer.target_ranks
                        if target_rank != self.rank
                    )
                    if self.rank in expert_param.target_ranks:
                        local_params.append((expert_param, tensor))
                else:
                    p2p_ops.append(dist.P2POp(dist.irecv, tensor, transfer.source_rank))
                    local_params.append((expert_param, tensor))

        for request in dist.batch_isend_irecv(p2p_ops) if p2p_ops else ():
            request.wait()

        hf_named_tensors = []
        for expert_param, tensor in local_params:
            hf_named_tensors.extend(
                convert_to_hf(
                    self.args,
                    self.model_name,
                    expert_param.info.name,
                    tensor,
                    self.quantization_config,
                )
            )
        return hf_named_tensors

    def _update_expert_weights(
        self,
        megatron_local_weights: Mapping[str, torch.Tensor],
    ) -> None:
        dist.barrier(group=get_gloo_group())
        # Initialize WORLD on all ranks before subset batched P2P.
        dist.barrier()
        # Reuse staging across layers instead of fragmenting the CUDA allocator.
        staging_buffers: dict[tuple[torch.dtype, tuple[int, ...]], list[torch.Tensor]] = {}
        for transfer_group in tqdm(
            self._expert_transfer_plan,
            disable=self.rank != 0,
            desc="Update expert weights",
        ):
            for transfer_batch in transfer_group:
                trace = getattr(self, "_weight_sync_trace", None)
                if trace is None:
                    hf_named_tensors = self._prepare_expert_weight_batch(
                        transfer_batch, megatron_local_weights, staging_buffers
                    )
                else:
                    with communication_phase("weight_convert", bucket_id=trace.next_bucket_id, bucket_kind="expert"):
                        hf_named_tensors = self._prepare_expert_weight_batch(
                            transfer_batch, megatron_local_weights, staging_buffers
                        )
                    trace.next_bucket_id += 1
                self._weight_sync_credit.set_persistent_staging_bytes(_tensor_tree_nbytes(staging_buffers))
                reservation = self._reserve_weight_bucket(hf_named_tensors)
                refs, handles, long_lived_tensors = self._submit_weight_bucket(hf_named_tensors, reservation)
                self._wait_for_bucket_completion(reservation, handles, refs, hf_named_tensors)
                hf_named_tensors.clear()
                if isinstance(long_lived_tensors, list):
                    long_lived_tensors.clear()
                self._weight_sync_credit.mark_staging_released(reservation)
                self._weight_sync_credit.release(reservation)
                if trace is not None:
                    trace.released(reservation)
                dist.barrier(group=get_gloo_group())
                accelerator.synchronize()
                del refs, handles, long_lived_tensors, hf_named_tensors
                accelerator.ipc_collect()
                accelerator.empty_cache()
        del staging_buffers
        self._weight_sync_credit.set_persistent_staging_bytes(0)
        accelerator.empty_cache()

    @torch.no_grad()
    @trace_weight_update
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        sync_started = perf_counter()
        self.weight_version += 1

        if self.rank == 0:
            consumer_engines = self._consumer_engines()
            ray.get([engine.pause_generation.remote() for engine in consumer_engines])
            ray.get([engine.flush_cache.remote() for engine in consumer_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=consumer_engines,
                    max_inflight_engine_groups=self._max_inflight_engine_groups,
                )
        dist.barrier(group=get_gloo_group())

        self._weight_sync_credit.begin_version(self.weight_version)
        try:
            megatron_local_weights = self.weights_getter()

            param_info_buckets = (
                self._non_expert_param_info_buckets if self._expert_transfer_plan else self._full_param_info_buckets
            )
            hf_chunks = self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights,
                param_info_buckets=param_info_buckets,
            )
            trace = getattr(self, "_weight_sync_trace", None)
            if trace is not None:
                hf_chunks = trace.converted_chunks(hf_chunks)
            if self._weight_sync_credit.enabled:
                self._send_weight_bucket_windows(hf_chunks)
            else:
                for hf_named_tensors in hf_chunks:
                    reservation = self._reserve_weight_bucket(hf_named_tensors)
                    refs, handles, long_lived_tensors = self._submit_weight_bucket(hf_named_tensors, reservation)
                    self._wait_for_bucket_completion(reservation, handles, refs, hf_named_tensors)
                    hf_named_tensors.clear()
                    if isinstance(long_lived_tensors, list):
                        long_lived_tensors.clear()
                    self._weight_sync_credit.mark_staging_released(reservation)
                    self._weight_sync_credit.release(reservation)
                    if trace is not None:
                        trace.released(reservation)
                    # Free GPU tensors so the caching allocator can reuse the blocks,
                    # then release CUDA IPC cache entries whose consumers (sglang engines)
                    # have already closed their IPC handles.
                    del refs, handles, long_lived_tensors, hf_named_tensors
                    accelerator.ipc_collect()
                    accelerator.empty_cache()

            if self._expert_transfer_plan:
                self._update_expert_weights(megatron_local_weights)

            del megatron_local_weights
            dist.barrier(group=get_gloo_group())
            # After the barrier all engines have returned, so every rank's last-chunk
            # IPC handles are now released by the consumers.  Clean them up.
            accelerator.ipc_collect()
            accelerator.empty_cache()

            # int4/fp4 post_process
            if self.rank == 0:
                if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                    post_process_weights(
                        restore_weights_before_load=False,
                        post_process_quantization=True,
                        rollout_engines=self._consumer_engines(),
                        max_inflight_engine_groups=self._max_inflight_engine_groups,
                    )
                ray.get([engine.continue_generation.remote() for engine in self._consumer_engines()])
            dist.barrier(group=get_gloo_group())
        except BaseException as error:
            self._weight_sync_credit.fail_version(self.weight_version, error)
            raise

        # Commit is the control-plane publication boundary: all loads and
        # post-processing completed and every consumer has resumed.
        self._weight_sync_credit.commit_version(self.weight_version)
        self.update_weight_metrics.update(self._weight_sync_credit.metrics())
        self.update_weight_metrics["perf/update_weights_sync_seconds"] = perf_counter() - sync_started

    def _reserve_weight_bucket(self, hf_named_tensors: Sequence[tuple[str, torch.Tensor]]) -> WeightBucketReservation:
        bucket_bytes = self._weight_bucket_bytes(hf_named_tensors)
        reservation = self._weight_sync_credit.reserve(bucket_bytes)
        if reservation is None:
            raise RuntimeError("weight bucket credits are full")
        return reservation

    def _weight_bucket_bytes(self, hf_named_tensors: Sequence[tuple[str, torch.Tensor]]) -> int:
        bucket_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in hf_named_tensors)
        if self._weight_sync_credit.max_inflight_bytes:
            # Every rank must flush at the same bucket boundary before the next
            # gather_object. Use the largest local representation so byte-credit
            # decisions stay identical even when a rank contributes no tensors.
            global_bucket_bytes = torch.tensor(bucket_bytes, dtype=torch.int64, device="cpu")
            dist.all_reduce(global_bucket_bytes, op=dist.ReduceOp.MAX, group=get_gloo_group())
            bucket_bytes = int(global_bucket_bytes.item())
        return bucket_bytes

    def _send_weight_bucket_windows(
        self,
        hf_chunks: Iterator[list[tuple[str, torch.Tensor]]],
    ) -> None:
        pending: list[tuple[WeightBucketReservation, list[tuple[str, torch.Tensor]]]] = []
        for hf_named_tensors in hf_chunks:
            bucket_bytes = self._weight_bucket_bytes(hf_named_tensors)
            reservation = self._weight_sync_credit.reserve(bucket_bytes)
            if reservation is None:
                self._flush_weight_bucket_window(pending)
                pending = []
                reservation = self._weight_sync_credit.reserve(bucket_bytes)
                if reservation is None:
                    raise RuntimeError("weight bucket credit did not admit an empty window")
            pending.append((reservation, hf_named_tensors))
            if self._weight_sync_credit.full:
                self._flush_weight_bucket_window(pending)
                pending = []

        if pending:
            self._flush_weight_bucket_window(pending)

    def _flush_weight_bucket_window(
        self,
        pending: Sequence[tuple[WeightBucketReservation, list[tuple[str, torch.Tensor]]]],
    ) -> None:
        launched = []
        for reservation, hf_named_tensors in pending:
            refs, handles, long_lived_tensors = self._submit_weight_bucket(hf_named_tensors, reservation)
            launched.append((reservation, hf_named_tensors, refs, handles, long_lived_tensors))

        while launched:
            reservation, hf_named_tensors, refs, handles, long_lived_tensors = launched.pop(0)
            self._wait_for_bucket_completion(reservation, handles, refs, hf_named_tensors)
            hf_named_tensors.clear()
            if isinstance(long_lived_tensors, list):
                long_lived_tensors.clear()
            del hf_named_tensors, refs, handles, long_lived_tensors
            self._weight_sync_credit.mark_staging_released(reservation)
            self._weight_sync_credit.release(reservation)
            if getattr(self, "_weight_sync_trace", None) is not None:
                self._weight_sync_trace.released(reservation)

        accelerator.ipc_collect()
        accelerator.empty_cache()

    def _submit_weight_bucket(
        self, hf_named_tensors, reservation: WeightBucketReservation
    ) -> tuple[list[ObjectRef], list[Any], Any]:
        if self._wave_scheduling_enabled:
            return self._send_hf_params_in_waves(hf_named_tensors, reservation)
        trace = getattr(self, "_weight_sync_trace", None)
        if trace is not None:
            trace.start(reservation)
        refs, handles, long_lived_tensors = self._send_hf_params(hf_named_tensors)
        if trace is not None:
            trace.submitted(reservation)
        self._mark_bucket_launched(reservation, refs, handles, long_lived_tensors)
        return refs, handles, long_lived_tensors

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], list[Any], Any]:
        all_refs = []
        all_handles = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            update_group = self._model_update_groups[0]
            refs_distributed, handles_distributed = launch_weights_from_distributed(
                update_group.group_name,
                update_group.process_group,
                self.weight_version,
                update_group.rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)
            all_handles.extend(handles_distributed)

        return all_refs, all_handles, long_lived_tensors

    def _send_hf_params_in_waves(
        self, hf_named_tensors, reservation: WeightBucketReservation
    ) -> tuple[list[ObjectRef], list[Any], None]:
        """Send one bucket in globally coordinated colocated/distributed waves."""
        self._weight_sync_credit.mark_launched(reservation, transport_bytes=0, staging_bytes=0, consumer_objects=0)
        engine_indices = tuple(range(self._total_engine_groups))
        trace = getattr(self, "_weight_sync_trace", None)
        for wave_id, wave in enumerate(build_engine_group_waves(engine_indices, self._max_inflight_engine_groups)):
            active_indices = {engine_index for _position, engine_index in wave}
            wave_refs = []
            wave_handles = []
            wave_long_lived_tensors = None
            if trace is not None:
                trace.start(reservation, wave_id=wave_id, engine_group_count=len(wave))

            if self._ipc_engine_index in active_indices:
                refs_colocated, wave_long_lived_tensors = _send_to_colocated_engine(
                    hf_named_tensors,
                    ipc_engine=self._ipc_engine,
                    ipc_gather_src=self._ipc_gather_src,
                    ipc_gather_group=self._ipc_gather_group,
                    weight_version=self.weight_version,
                )
                wave_refs.extend(refs_colocated)

            if self.use_distribute and self._is_distributed_src_rank:
                for update_group in self._model_update_groups:
                    if not active_indices.intersection(update_group.engine_indices):
                        continue
                    refs, handles = launch_weights_from_distributed(
                        update_group.group_name,
                        update_group.process_group,
                        self.weight_version,
                        update_group.rollout_engines,
                        hf_named_tensors,
                    )
                    wave_refs.extend(refs)
                    wave_handles.extend(handles)

            if trace is not None:
                trace.submitted(reservation)
            self._weight_sync_credit.mark_next_wave(
                reservation,
                transport_bytes=reservation.bucket_bytes if wave_handles or wave_refs else 0,
                staging_bytes=_tensor_tree_nbytes(wave_long_lived_tensors),
                consumer_objects=len(wave_refs),
            )
            error_message = None
            try:
                self._wait_for_bucket_completion(reservation, wave_handles, wave_refs, hf_named_tensors)
            except Exception as error:
                error_message = f"{type(error).__name__}: {error}"
            # Propagate a load failure to inactive engines too, before anybody
            # enters the next wave. This also keeps all IPC producers alive.
            group = get_gloo_group()
            errors = [None] * dist.get_world_size(group=group)
            dist.all_gather_object(errors, error_message, group=group)
            if any(errors):
                self._active_wave_resources = (hf_named_tensors, wave_long_lived_tensors, wave_refs, wave_handles)
                raise RuntimeError(f"weight wave failed: {errors}")

            # Every training rank advances together, so IPC producers cannot
            # release a bucket while another rank in the engine is still using it.
            if isinstance(wave_long_lived_tensors, list):
                wave_long_lived_tensors.clear()
            del wave_long_lived_tensors
            self._weight_sync_credit.mark_staging_released(reservation)

        # Every Ray ref and IPC consumer completed at its wave boundary.
        return [], [], None


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    long_live_tensors = []

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors} if hf_named_tensors else {}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_data = _build_flattened_tensor_data(named_tensors)
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
        num_buckets = max(len(tensors) for tensors in serialized_named_tensors)
        empty_serialized_tensor = None
        for i in range(num_buckets):
            serialized_tensors_for_dtype = []
            for tensors in serialized_named_tensors:
                if i < len(tensors):
                    serialized_tensors_for_dtype.append(tensors[i])
                    continue

                if empty_serialized_tensor is None:
                    empty_tensor_data = _empty_flattened_tensor_data()
                    long_live_tensors.append(empty_tensor_data)
                    empty_serialized_tensor = MultiprocessingSerializer.serialize(empty_tensor_data, output_str=True)
                serialized_tensors_for_dtype.append(empty_serialized_tensor)

            kwargs = {
                "serialized_named_tensors": serialized_tensors_for_dtype,
                "load_format": "flattened_bucket",
                "weight_version": str(weight_version),
            }
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors


def _empty_flattened_tensor_data():
    return {
        "flattened_tensor": torch.empty(0, dtype=torch.uint8, device=accelerator.current_device()),
        "metadata": [],
    }
