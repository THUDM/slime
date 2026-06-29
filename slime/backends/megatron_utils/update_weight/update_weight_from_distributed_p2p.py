"""
Shard-level P2P weight update for non-colocate Slime.

Each training TP rank converts its local shard to HF layout and sends it to the
matching inference TP rank via ``dist.send`` / ``dist.recv``, avoiding the default
all_gather + NCCL broadcast path.

Enable with ``--use-p2p-weight-update`` (``--update-weight-mode=full``,
``--megatron-to-hf-mode bridge``, non-colocate). Falls back to NCCL broadcast when
preconditions fail; see ``p2p_weight_update_supported`` in ``common.py`` and
``docs/en/advanced/p2p-weight-sync.md``.
"""

import socket
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf
from slime.backends.megatron_utils.misc_utils import strip_param_name_prefix
from slime.utils.distributed_utils import get_gloo_group, init_process_group

from ..megatron_to_hf import convert_shard_to_hf
from .common import (
    all_gather_param,
    named_params_and_buffers,
    p2p_weight_update_fallback_reason,
    p2p_weight_update_supported,
)
from .update_weight_from_distributed import UpdateWeightFromDistributed

# Vocab params that need all_gather + remove_padding due to Megatron/SGLang
# shard boundary misalignment.
_VOCAB_PARAMS = {"embedding.word_embeddings.weight", "output_layer.weight"}


class UpdateWeightFromDistributedP2P:
    """
    Shard-level P2P weight update: each TP rank sends its shard directly
    to the matching inference TP rank via dist.send/recv.

    Vocab params (embed_tokens, lm_head) are handled with a small all_gather
    on the TP group because Megatron and SGLang use different vocab partitioning.
    All other params use direct shard-level conversion without all_gather.
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
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self._weight_version = 0
        self._model_update_groups = None
        self._group_name = None
        self._first_update = True
        self._use_p2p: bool | None = None
        self._fallback: UpdateWeightFromDistributed | None = None

    def _get_fallback(self) -> UpdateWeightFromDistributed:
        if self._fallback is None:
            self._fallback = UpdateWeightFromDistributed(
                self.args,
                self.model,
                model_name=self.model_name,
                quantization_config=self.quantization_config,
            )
        return self._fallback

    def _resolve_updater(self, engine_gpu_counts: Sequence[int] | None = None) -> UpdateWeightFromDistributed | None:
        """Return broadcast fallback when P2P preconditions fail; None to use P2P."""
        if self._use_p2p is False:
            return self._get_fallback()
        if self._use_p2p is None:
            if p2p_weight_update_supported(self.args, self.model_name, engine_gpu_counts):
                self._use_p2p = True
                return None
            self._use_p2p = False
            if self.tp_rank == 0 and self._is_dp0:
                reason = p2p_weight_update_fallback_reason(self.args, self.model_name, engine_gpu_counts)
                print(f"[P2P] {reason}; using NCCL broadcast weight update instead.")
            return self._get_fallback()
        return None

    @property
    def weight_version(self) -> int:
        if self._use_p2p is False and self._fallback is not None:
            return self._fallback.weight_version
        return self._weight_version

    def disconnect_rollout_engines(self) -> None:
        if self._use_p2p is False and self._fallback is not None:
            self._fallback.disconnect_rollout_engines()

    def pop_metrics(self) -> dict[str, float]:
        if self._use_p2p is False:
            return self._get_fallback().pop_metrics()
        out, self._metrics = getattr(self, "_metrics", {}), {}
        return out

    @property
    def tp_size(self):
        return mpu.get_tensor_model_parallel_world_size()

    @property
    def tp_rank(self):
        return mpu.get_tensor_model_parallel_rank()

    @property
    def _is_dp0(self):
        return mpu.get_data_parallel_rank(with_context_parallel=True) == 0

    def _log_timing(self, label: str, elapsed: float, extra: str = ""):
        """Log per-phase latency on training TP rank 0."""
        if self.tp_rank == 0:
            warmup_tag = " [warmup]" if self._first_update else ""
            print(f"[P2P-TIMING]{warmup_tag} {label}: {elapsed:.3f}s{extra}")

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Join training and rollout ranks in a shared NCCL process group.

        TP rank 0 picks the master port, asks each rollout engine to join, and
        broadcasts the port to other training TP ranks. Every training TP rank
        then enters the group at its TP index; rollout ranks follow contiguous
        offsets derived from ``engine_gpu_counts``.
        """
        fallback = self._resolve_updater(engine_gpu_counts)
        if fallback is not None:
            return fallback.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )

        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        group_name = f"slime-p2p-pp_{pp_rank}"

        tp_size = self.tp_size
        tp_rank = self.tp_rank

        # Only DP=0 ranks participate in weight transfer
        if not self._is_dp0:
            return

        self._group_name = group_name
        world_size = tp_size + sum(engine_gpu_counts)

        # Compute cumulative rank offsets for engines
        cumulative = [0]
        for c in engine_gpu_counts:
            cumulative.append(cumulative[-1] + c)

        # Use the TP group to broadcast the port from TP rank 0 to all others.
        tp_group = mpu.get_tensor_model_parallel_group()

        if tp_rank == 0:
            # TP rank 0 coordinates group setup and teardown.

            if self._model_update_groups is not None:
                refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
                dist.destroy_process_group(self._model_update_groups)
                self._model_update_groups = None
                ray.get(refs)

            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            # Tell rollout engines to join the NCCL group.
            refs = [
                engine.init_weights_update_group.remote(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=tp_size + cumulative[i],
                    world_size=world_size,
                    group_name=group_name,
                    backend="nccl",
                )
                for i, engine in enumerate(rollout_engines)
            ]

            # Broadcast port to other training TP ranks via TP group.
            port_tensor = torch.tensor([master_port], dtype=torch.long, device="cuda")
            dist.broadcast(port_tensor, src=0, group=tp_group)

            # TP rank 0 enters the NCCL group as rank 0.
            self._model_update_groups = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=group_name,
                device_id=torch.device("cuda", torch.cuda.current_device()),
            )

            ray.get(refs)
        else:
            # Other training TP ranks receive the port from TP rank 0, then join.

            if self._model_update_groups is not None:
                dist.destroy_process_group(self._model_update_groups)
                self._model_update_groups = None

            # Receive port from TP rank 0 via TP group broadcast
            port_tensor = torch.tensor([0], dtype=torch.long, device="cuda")
            dist.broadcast(port_tensor, src=0, group=tp_group)

            master_port = int(port_tensor.item())
            master_address = ray._private.services.get_node_ip_address()

            # Each TP rank joins the NCCL group as its TP rank index.
            self._model_update_groups = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=tp_rank,
                group_name=group_name,
                device_id=torch.device("cuda", torch.cuda.current_device()),
            )

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Convert Megatron shards to HF layout and P2P-send them to rollout engines.

        Vocab tensors use a small TP all_gather to align Megatron/SGLang padding;
        all other parameters are converted and sent shard-wise without all_gather.
        """
        if self._use_p2p is False:
            return self._get_fallback().update_weights()

        t_total_start = time.perf_counter()
        self._weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

        dist.barrier(group=get_gloo_group())

        if not self._is_dp0:
            # Non-DP0 ranks still participate in Gloo barriers but skip NCCL work.
            dist.barrier(group=get_gloo_group())
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
            dist.barrier(group=get_gloo_group())
            return

        tp_rank = self.tp_rank
        tp_size = self.tp_size

        # Phase 1: Vocab params — all_gather + remove_padding + slice for SGLang alignment
        t_phase1_start = time.perf_counter()
        buffer_size = 0
        converted_named_tensors = []
        pbar = tqdm(desc=f"[p2p-tp{tp_rank}] Update weights", total=0) if dist.get_rank() == 0 else None

        vocab_count = 0
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            stripped = strip_param_name_prefix(name)
            if stripped not in _VOCAB_PARAMS:
                continue
            buffer_size = self._update_vocab_shard(
                name, param, converted_named_tensors, buffer_size, tp_rank, tp_size, pbar=pbar
            )
            vocab_count += 1

        t_phase1_end = time.perf_counter()
        self._log_timing(
            "Phase1 vocab (all_gather+convert+slice)",
            t_phase1_end - t_phase1_start,
            f" [{vocab_count} params, {len(converted_named_tensors)} tensors queued]",
        )

        # Phase 2: Non-vocab, non-expert params — shard-level conversion (no all_gather)
        t_phase2_start = time.perf_counter()
        nonvocab_count = 0
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            stripped = strip_param_name_prefix(name)
            if stripped in _VOCAB_PARAMS:
                continue
            buffer_size = self._update_shard_from_distributed(
                name, param, converted_named_tensors, buffer_size, tp_rank, tp_size, pbar=pbar
            )
            nonvocab_count += 1

        # Flush remaining non-expert tensors
        if converted_named_tensors:
            t_flush_start = time.perf_counter()
            self._send_bucket_shard(converted_named_tensors, pbar=pbar)
            t_flush_end = time.perf_counter()
            self._log_timing("Phase2 flush bucket", t_flush_end - t_flush_start)

        t_phase2_end = time.perf_counter()
        self._log_timing(
            "Phase2 non-vocab (convert_shard+send)",
            t_phase2_end - t_phase2_start,
            f" [{nonvocab_count} params processed]",
        )

        dist.barrier(group=get_gloo_group())
        t_barrier1 = time.perf_counter()
        self._log_timing("Gloo barrier after Phase2", t_barrier1 - t_phase2_end)

        # Phase 3: Expert params — send shard directly, no all_gather
        t_phase3_start = time.perf_counter()
        buffer_size = 0
        named_tensors = []
        expert_count = 0
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_shard_from_distributed(
                name, param, named_tensors, buffer_size, tp_rank, tp_size, pbar=pbar
            )
            expert_count += 1

        if named_tensors:
            self._send_bucket_shard(named_tensors, pbar=pbar, is_expert=True)

        t_phase3_end = time.perf_counter()
        self._log_timing("Phase3 expert (direct send)", t_phase3_end - t_phase3_start, f" [{expert_count} params]")

        # Wait for all SGLang load_weights to complete before resuming generation.
        dist.barrier(group=get_gloo_group())

        if dist.get_rank() == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        t_total_end = time.perf_counter()
        self._log_timing("TOTAL update_weights", t_total_end - t_total_start)

        self._first_update = False

    def _update_vocab_shard(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        tp_rank: int,
        tp_size: int,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Vocab params: all_gather on TP group → convert to HF (which removes Megatron
        padding internally) → slice SGLang-aligned shard for this TP rank.
        """
        t0 = time.perf_counter()

        # Step 1: all_gather on the TP group to reconstruct the full vocab tensor
        full_param = all_gather_param(name, param)
        t_allgather = time.perf_counter()

        # Step 2: Convert to HF format (internally calls remove_padding)
        converted = convert_to_hf(self.args, self.model_name, name, full_param, self.quantization_config)
        t_convert = time.perf_counter()

        if not converted:
            return buffer_size

        # Step 3: Slice the SGLang-aligned shard for this TP rank
        for hf_name, hf_param in converted:
            if hf_param.shape[0] % tp_size != 0:
                shard = hf_param
            else:
                shard_size = hf_param.shape[0] // tp_size
                shard = hf_param[tp_rank * shard_size : (tp_rank + 1) * shard_size]

            param_size = shard.numel() * shard.element_size()
            if buffer_size + param_size > self.args.update_weight_buffer_size:
                self._send_bucket_shard(converted_named_tensors, pbar=pbar)
                buffer_size = 0
            converted_named_tensors.append((hf_name, shard))
            buffer_size += param_size

        t_slice = time.perf_counter()
        self._log_timing(
            f"  vocab '{name}'",
            t_slice - t0,
            f" [all_gather={t_allgather-t0:.3f}s, convert={t_convert-t_allgather:.3f}s, slice+buf={t_slice-t_convert:.3f}s]",
        )

        return buffer_size

    def _update_shard_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        tp_rank: int,
        tp_size: int,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Convert a single TP shard to HF format. No all_gather.
        """
        converted = convert_shard_to_hf(self.args, self.model_name, name, param, tp_rank, tp_size)
        if not converted:
            return buffer_size

        for hf_name, hf_param in converted:
            param_size = hf_param.numel() * hf_param.element_size()
            if buffer_size + param_size > self.args.update_weight_buffer_size:
                self._send_bucket_shard(converted_named_tensors, pbar=pbar)
                buffer_size = 0
            converted_named_tensors.append((hf_name, hf_param))
            buffer_size += param_size

        return buffer_size

    def _update_expert_shard_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        tp_rank: int,
        tp_size: int,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Expert params: no all_gather, send shard directly.
        """
        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            self._send_bucket_shard(named_tensors, pbar=pbar, is_expert=True)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _send_bucket_shard(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
        is_expert: bool = False,
    ) -> None:
        """
        Send one bucket of presharded tensors to all rollout engines.

        Rank 0 posts a single HTTP request with concatenated metadata from every
        training TP rank (``tp_tensor_counts``). All training ranks then enter an
        NCCL barrier with rollout workers, ``dist.send`` their local tensors to
        matching rollout TP ranks on each engine, and wait for load completion.
        """
        t_bucket_start = time.perf_counter()
        tp_rank = self.tp_rank
        tp_size = self.tp_size
        group = self._model_update_groups
        load_format = "presharded"

        num_tensors = len(converted_named_tensors)
        refs: list[ObjectRef] | None = None

        t_lock_start = time.perf_counter()

        # Gather all TP ranks' metadata to rank 0 via gloo.
        my_meta = (
            [name for name, _ in converted_named_tensors],
            [str(param.dtype).replace("torch.", "") for _, param in converted_named_tensors],
            [list(param.shape) for _, param in converted_named_tensors],
        )

        all_meta = [None] * tp_size
        dist.all_gather_object(all_meta, my_meta, group=get_gloo_group())

        tp_tensor_counts = [len(m[0]) for m in all_meta]

        # Only TP rank 0 sends the HTTP request with concatenated metadata.
        if tp_rank == 0:
            all_names = []
            all_dtypes = []
            all_shapes = []
            for m in all_meta:
                all_names.extend(m[0])
                all_dtypes.extend(m[1])
                all_shapes.extend(m[2])

            while not ray.get(self.rollout_engine_lock.acquire.remote()):
                time.sleep(0.1)

            refs = [
                engine.update_weights_from_distributed.remote(
                    names=all_names,
                    dtypes=all_dtypes,
                    shapes=all_shapes,
                    group_name=self._group_name,
                    weight_version=str(self.weight_version),
                    load_format=load_format,
                    tp_tensor_counts=tp_tensor_counts,
                )
                for engine in self.rollout_engines
            ]

            ray.get(self.rollout_engine_lock.release.remote())

        t_lock_end = time.perf_counter()

        # NCCL barrier: rollout workers post dist.recv before training sends.
        dist.barrier(group=group)

        # Every training TP rank sends to the matching rank on each engine.
        cumulative = [0]
        for c in self._engine_gpu_counts:
            cumulative.append(cumulative[-1] + c)

        num_engines = len(self.rollout_engines)

        t_send_start = time.perf_counter()
        for _, param in converted_named_tensors:
            tensor = param.data.contiguous()
            for engine_idx in range(num_engines):
                dst_rank = tp_size + cumulative[engine_idx] + tp_rank
                dist.send(tensor, dst=dst_rank, group=group)

        t_send_end = time.perf_counter()

        if refs is not None:
            ray.get(refs)

        t_ray_end = time.perf_counter()

        converted_named_tensors.clear()

        self._log_timing(
            f"  bucket ({'expert' if is_expert else 'non-expert'}, {num_tensors} tensors)",
            t_ray_end - t_bucket_start,
            f" [lock+ray={t_lock_end - t_lock_start:.3f}s, dist.send={t_send_end - t_send_start:.3f}s]",
        )

        if pbar is not None:
            pbar.update(1)
