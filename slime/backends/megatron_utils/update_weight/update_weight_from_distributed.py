import logging
import socket
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from slime.utils.distributed_utils import get_gloo_group, init_process_group

from ..megatron_to_hf import convert_to_hf
from .common import HFUpdate, PendingHFUpdateBucket, all_gather_param, named_params_and_buffers
from .delta_weight_update import DeltaCompressionTracker, materialize_delta_transport

logger = logging.getLogger(__name__)


class UpdateWeightFromDistributed:
    """
    Update distributed engines via NCCL. Each PP rank: group "slime-pp_{pp_rank}",
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
    ) -> None:
        """
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        self.delta_tracker = DeltaCompressionTracker(args) if args.enable_delta_compression else None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Create NCCL "slime-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
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
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args,
                self._group_name,
                rollout_engines,
                engine_gpu_counts=engine_gpu_counts,
            )

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → flush → non-expert (TP) → expert (EP) → continue. Progress on PP source.
        """
        import time as _t

        _t_update_start = _t.monotonic()
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
                )

        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        converted_named_tensors = []
        # non expert params
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

        if self.delta_tracker is None:
            for name, param in named_params_and_buffers(self.args, self.model):
                if ".experts." in name:
                    continue
                buffer_size = self._update_weight_from_distributed(
                    name, param, converted_named_tensors, buffer_size, pbar=pbar
                )

            if converted_named_tensors:
                self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
        else:
            pending_bucket = PendingHFUpdateBucket.empty()

            for name, param in named_params_and_buffers(self.args, self.model):
                if ".experts." in name:
                    continue
                param = all_gather_param(name, param)
                if not self._is_pp_src_rank:
                    continue
                hf_named_tensors = convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
                self._enqueue_delta_chunk_for_send(
                    self._prepare_hf_chunk_for_send(hf_named_tensors),
                    pending_bucket,
                    pbar=pbar,
                )

            self._flush_hf_update_bucket_from_distributed(pending_bucket, pbar=pbar)

        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        named_tensors = []
        expert_bucket = PendingHFUpdateBucket.empty() if self.delta_tracker is not None else None
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_weight_from_distributed(
                name,
                param,
                named_tensors,
                buffer_size,
                pending_bucket=expert_bucket,
                pbar=pbar,
            )

        if named_tensors:
            self._update_expert_bucket_weights_from_distributed(
                named_tensors,
                pending_bucket=expert_bucket,
                pbar=pbar,
            )

        if expert_bucket is not None:
            self._flush_hf_update_bucket_from_distributed(expert_bucket, pbar=pbar)

        dist.barrier(group=get_gloo_group())
        if self._is_pp_src_rank and self.delta_tracker is not None:
            self.delta_tracker.on_sync_succeeded()
        if self._is_pp_src_rank:
            logger.info("delta_profile: update_weights_total=%.3fs", _t.monotonic() - _t_update_start)
        if dist.get_rank() == 0:
            # int4/fp4 post_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
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
        pending_bucket: PendingHFUpdateBucket | None = None,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        param = all_gather_param(name, param)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            self._update_expert_bucket_weights_from_distributed(
                named_tensors,
                pending_bucket=pending_bucket,
                pbar=pbar,
            )
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_distributed(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        pending_bucket: PendingHFUpdateBucket | None = None,
        pbar: tqdm | None = None,
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
        for i, (_name, param) in enumerate(named_tensors):
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

        if self.delta_tracker is None or pending_bucket is None:
            self._update_bucket_weights_from_distributed(converted_hf_tensors, pbar)
        else:
            self._enqueue_delta_chunk_for_send(
                self._prepare_hf_chunk_for_send(converted_hf_tensors),
                pending_bucket,
                pbar=pbar,
            )

    def _update_weight_from_distributed_with_delta(
        self,
        name: str,
        param: torch.nn.Parameter,
        pending_bucket: PendingHFUpdateBucket,
        pbar: tqdm | None = None,
    ) -> None:
        param = all_gather_param(name, param)
        if not self._is_pp_src_rank:
            return

        hf_named_tensors = convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        self._enqueue_delta_chunk_for_send(
            self._prepare_hf_chunk_for_send(hf_named_tensors),
            pending_bucket,
            pbar=pbar,
        )

    def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Lock → broadcast → clear → unlock → pbar++. Lock prevents NCCL deadlock.
        """
        if not converted_named_tensors:
            return
        chunk_update = self._prepare_hf_chunk_for_send(converted_named_tensors)
        if chunk_update.commit_state is not None and not chunk_update.should_send:
            self._finalize_sent_chunk(chunk_update.commit_state)
            if pbar is not None:
                pbar.update(1)
            converted_named_tensors.clear()
            return
        self._send_hf_update(chunk_update.tensors, chunk_update.load_format)
        converted_named_tensors.clear()
        self._finalize_sent_chunk(chunk_update.commit_state)
        if pbar is not None:
            pbar.update(1)

    def _prepare_hf_chunk_for_send(
        self,
        hf_named_tensors: list[tuple[str, torch.Tensor]],
    ) -> HFUpdate:
        if self.delta_tracker is None:
            return HFUpdate(tensors=list(hf_named_tensors), load_format=None, commit_state=None)
        else:
            prepared = self.delta_tracker.prepare_chunk(hf_named_tensors)
            if not prepared.is_delta:
                return HFUpdate(tensors=prepared.tensors, load_format=None, commit_state=prepared.commit_state)
            else:
                # Eagerly sparse-encode: materialize now so the bucket holds
                # tiny sparse tensors instead of huge dense deltas. This frees
                # GPU memory immediately, enabling far fewer flushes without OOM.
                materialized = materialize_delta_transport(prepared.tensors, self.args.delta_compression_transport)
                sparse_bytes = sum(t.numel() * t.element_size() for _, t in materialized.tensors)
                return HFUpdate(
                    tensors=materialized.tensors,
                    load_format=materialized.load_format,
                    commit_state=prepared.commit_state,
                    transport_byte_size=sparse_bytes,
                    sparse_metadata=materialized.sparse_metadata,
                    sparse_metadata_count=len(materialized.sparse_metadata or []),
                )

    def _finalize_sent_chunk(self, commit_state) -> None:
        if self.delta_tracker is None:
            return

        assert commit_state is not None
        self.delta_tracker.commit_chunk(commit_state, weight_version=self.weight_version)

    def _enqueue_delta_chunk_for_send(
        self,
        chunk_update: HFUpdate,
        pending_bucket: PendingHFUpdateBucket,
        pbar: tqdm | None = None,
    ) -> None:
        if chunk_update.commit_state is not None and not chunk_update.should_send:
            self._finalize_sent_chunk(chunk_update.commit_state)
            return

        # With eager materialization, the bucket holds sparse-encoded data (tiny),
        # not dense deltas (huge). Safe to use a very large limit — each PP rank's
        # total sparse data is ~1.5 GB, so 5 GB fits everything in one flush.
        if chunk_update.load_format is not None:
            _DELTA_SPARSE_BYTE_LIMIT = 5 * 1024 * 1024 * 1024  # 5 GB sparse
            if pending_bucket.should_flush_before_add(chunk_update, _DELTA_SPARSE_BYTE_LIMIT):
                self._flush_hf_update_bucket_from_distributed(pending_bucket, pbar=pbar)
        elif pending_bucket.should_flush_before_add(chunk_update, self.args.update_weight_buffer_size):
            self._flush_hf_update_bucket_from_distributed(pending_bucket, pbar=pbar)
        pending_bucket.add(chunk_update)

    def _send_hf_update(
        self,
        tensors: list[tuple[str, torch.Tensor]],
        load_format: str | None,
        pre_sparse_metadata: list[dict] | None = None,
        pre_sparse_metadata_counts: list[int] | None = None,
    ) -> None:
        import time as _t

        _t0 = _t.monotonic()
        if load_format is None:
            send_tensors = tensors
            sparse_metadata = None
        elif pre_sparse_metadata is not None:
            # Eagerly materialized: tensors are already sparse-encoded packed
            # tensors. Consolidate multiple chunks' packed buffers into one pair.
            send_tensors, sparse_metadata = _consolidate_sparse_tensors(
                tensors,
                pre_sparse_metadata,
                pre_sparse_metadata_counts,
                load_format,
            )
        else:
            materialized = materialize_delta_transport(tensors, self.args.delta_compression_transport)
            send_tensors = materialized.tensors
            sparse_metadata = materialized.sparse_metadata
            load_format = materialized.load_format
        _t_materialize = _t.monotonic() - _t0

        _t0 = _t.monotonic()
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            pass
        _t_lock = _t.monotonic() - _t0

        _t0 = _t.monotonic()
        try:
            refs = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.rollout_engines,
                send_tensors,
                load_format=load_format,
                sparse_metadata=sparse_metadata,
            )
            ray.get(refs)
        finally:
            ray.get(self.rollout_engine_lock.release.remote())
        _t_broadcast = _t.monotonic() - _t0
        if load_format is not None:
            logger.info(
                "delta_profile: send_hf_update materialize=%.3fs lock=%.3fs broadcast=%.3fs format=%s",
                _t_materialize,
                _t_lock,
                _t_broadcast,
                load_format,
            )

    def _flush_hf_update_bucket_from_distributed(
        self,
        pending_bucket: PendingHFUpdateBucket,
        pbar: tqdm | None = None,
    ) -> None:
        if not pending_bucket.has_updates:
            return

        self._send_hf_update(
            pending_bucket.tensors,
            pending_bucket.load_format,
            pre_sparse_metadata=pending_bucket.sparse_metadata,
            pre_sparse_metadata_counts=pending_bucket.sparse_metadata_counts,
        )

        for commit_state in pending_bucket.commit_states:
            self._finalize_sent_chunk(commit_state)

        pending_bucket.clear()
        if pbar is not None:
            pbar.update(1)


def _consolidate_sparse_tensors(
    tensors: list[tuple[str, torch.Tensor]],
    sparse_metadata: list[dict],
    sparse_metadata_counts: list[int] | None,
    load_format: str,
) -> tuple[list[tuple[str, torch.Tensor]], list[dict]]:
    """Consolidate multiple eagerly-materialized sparse chunks into one broadcast.

    Each chunk produced its own packed buffer pair with metadata referencing
    local offsets. Concatenate all chunks' packed buffers and shift metadata
    offsets so the receiver sees one contiguous broadcast.
    """
    if "sparse_indices" in load_format:
        key_a, key_b = "__packed_indices__", "__packed_values__"
        off_a, off_b = "index_start", "value_start"
        end_a, end_b = "index_end", "value_end"
    elif "sparse_bitmask" in load_format:
        key_a, key_b = "__packed_masks__", "__packed_values__"
        off_a, off_b = "mask_start", "value_start"
        end_a, end_b = "mask_end", "value_end"
    else:
        return tensors, sparse_metadata

    bufs_a = [t for name, t in tensors if name == key_a]
    bufs_b = [t for name, t in tensors if name == key_b]

    if len(bufs_a) <= 1:
        # Single chunk — no consolidation needed
        return tensors, sparse_metadata

    if len(bufs_a) != len(bufs_b):
        raise ValueError(f"Mismatched sparse packed buffer counts: {len(bufs_a)=} {len(bufs_b)=} for {load_format=}")
    if sparse_metadata_counts is None:
        raise ValueError("Missing sparse metadata chunk boundaries for consolidation")
    if len(sparse_metadata_counts) != len(bufs_a):
        raise ValueError(
            "Sparse metadata chunk boundary count does not match packed buffer pairs: "
            f"{len(sparse_metadata_counts)=} {len(bufs_a)=}"
        )

    # Build cumulative offset table: chunk i starts at cum_a[i], cum_b[i]
    cum_a = [0]
    cum_b = [0]
    for a, b in zip(bufs_a, bufs_b, strict=True):
        cum_a.append(cum_a[-1] + a.numel())
        cum_b.append(cum_b[-1] + b.numel())

    adjusted = []
    mi = 0
    for ci, meta_count in enumerate(sparse_metadata_counts):
        shift_a = cum_a[ci]
        shift_b = cum_b[ci]
        for meta in sparse_metadata[mi : mi + meta_count]:
            adj = dict(meta)
            adj[off_a] += shift_a
            adj[end_a] += shift_a
            adj[off_b] += shift_b
            adj[end_b] += shift_b
            adjusted.append(adj)
        mi += meta_count

    if mi != len(sparse_metadata):
        raise ValueError(
            "Sparse metadata chunk boundaries did not consume the full metadata list: "
            f"{mi=} {len(sparse_metadata)=}"
        )

    packed_a = torch.cat(bufs_a, dim=0)
    packed_b = torch.cat(bufs_b, dim=0)
    return [(key_a, packed_a), (key_b, packed_b)], adjusted


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
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

    refs = [
        engine.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=cumulative[i] + 1,
            world_size=world_size,
            group_name=group_name,
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
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    load_format: str | None = None,
    sparse_metadata: list[dict] | None = None,
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
            **({"load_format": load_format} if load_format is not None else {}),
            **({"sparse_metadata": sparse_metadata} if sparse_metadata is not None else {}),
        )
        for engine in rollout_engines
    ]
    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()
    return refs


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
):
    """
    Trigger post-process for int4/fp4 quantization on all rollout engines.
    """
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )
