import logging
import os
from argparse import Namespace
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.observability import train_data_utils, train_metric_utils
from slime.observability.logging_utils import init_tracking
from slime.observability.profile_utils import TrainProfiler
from slime.observability.timer import Timer, inverse_timer, timer, with_defer
from slime.ray.train_actor import TrainRayActor
from slime.utils import accelerator
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.misc import Box
from slime.utils.reloadable_process_group import (
    destroy_process_groups,
    monkey_patch_torch_dist,
    register_default_process_group,
    reload_process_groups,
)
from slime.utils.routing_replay import RoutingReplay
from slime.utils.rs_refill import clone_rs_masks, compute_sequence_rs_masks, validate_final_rs_masks
from slime.utils.types import RolloutBatch

from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import prepare_routed_experts_for_routing_replay, slice_log_prob_with_cp
from .data import DataIterator, get_data_iterator
from .hf_checkpoint_saver import save_hf_model_to_path
from .initialize import init, is_megatron_main_rank
from .loss import (
    compute_advantages_and_returns,
    drain_captured_log_probs,
    enable_log_prob_capture,
    get_log_probs_and_entropy,
    get_values,
)
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_disk import UpdateWeightFromDisk
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        if args.debug_rollout_only:
            self.args = args
            return 0

        monkey_patch_torch_dist()
        super().init(args, role, with_ref, with_opd_teacher)
        self._rs_candidate_log_probs: dict[int, dict[int, torch.Tensor]] = {}
        # Destroying and recreating WORLD invalidates raw dist.group.WORLD references cached by external code.
        # Set SLIME_DESTROY_WORLD_PROCESS_GROUP=0 when such references may outlive a train sleep/wake cycle.
        if os.getenv("SLIME_DESTROY_WORLD_PROCESS_GROUP", "1").lower() not in {"0", "false", "no"}:
            register_default_process_group(timeout=timedelta(minutes=args.distributed_timeout_minutes))
        else:
            logger.info("Default WORLD process-group destruction is disabled")

        init(args)

        if is_megatron_main_rank():
            init_tracking(args, primary=False, role=role)

        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        dist.barrier(group=get_gloo_group())

        self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
            args, role
        )

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
        if vpp_size > 1:
            from megatron.core.utils import get_model_config

            microbatch_group_size_per_vp_stage = get_model_config(self.model[0]).microbatch_group_size_per_vp_stage
        else:
            microbatch_group_size_per_vp_stage = 1
        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
            "cp_size": mpu.get_context_parallel_world_size(),
            "vpp_size": vpp_size,
            "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
        }

        start_rollout_id = loaded_rollout_id + 1

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return start_rollout_id

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=True,
            ),
            single_tag=None,
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        # Load teacher model for Megatron-based on-policy distillation
        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            # Prefer HF config vocab_size (which may include model-native padding)
            # over tokenizer vocab_size, which may be smaller.
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        update_weight_mode = self.args.update_weight_mode
        update_weight_transport = self.args.update_weight_transport

        if update_weight_mode == "delta":
            # Delta sync is disk-transport only: each engine's /pull_weights applies the published
            # deltas into a host-local checkpoint on every host it spans, and the engines reload
            # via vanilla update_weights_from_disk.
            assert not self.args.colocate, "--update-weight-mode=delta is not supported with --colocate"
            assert (
                update_weight_transport == "disk"
            ), "--update-weight-mode=delta requires --update-weight-transport=disk"
            from .update_weight.update_weight_from_disk_delta import UpdateWeightFromDiskDelta

            update_weight_cls = UpdateWeightFromDiskDelta
        elif update_weight_transport == "disk":
            update_weight_cls = UpdateWeightFromDisk
        elif self.args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        else:
            assert update_weight_mode == "full"
            assert (
                update_weight_transport == "nccl"
            ), f"unsupported weight sync mode/transport: {update_weight_mode!r}/{update_weight_transport!r}"
            update_weight_cls = UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )
        self.weight_updater.weight_version = getattr(self.args, "update_weight_start_version", 0)

        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            self._switch_model("actor")
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        if (
            self.role == "actor"
            and self.args.use_critic
            and not self.args.colocate
            and hasattr(self.weight_updater, "disconnect_rollout_engines")
        ):
            self.weight_updater.disconnect_rollout_engines()
        destroy_process_groups()

        torch_memory_saver.pause()

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        torch_memory_saver.resume()

        clear_memory()
        reload_process_groups()

        if mpu.get_pipeline_model_parallel_world_size() > 2:
            # Megatron's patched batched pipeline P2P uses the default WORLD
            # group.  After reload, PP=4 starts with only the first two stages
            # entering batch_isend_irecv(), but PyTorch requires every rank when
            # that is the first NCCL operation on a group.  Prime WORLD here,
            # after the memory saver is resumed, so later stages cannot miss its
            # lazy initialization.  Sleep still destroys it completely.
            dist.barrier(device_ids=[accelerator.current_device()])
        if self.role == "actor":
            self._switch_model("actor")
        print_memory("after wake_up model")

    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will receive the data.
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        # TODO: this is ugly, move to somewhere else?
        # move tokens to GPU in advance
        device = accelerator.current_device()
        rollout_data["tokens"] = [
            t.to(device=device, dtype=torch.long, non_blocking=True) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            t.to(device=device, dtype=torch.int, non_blocking=True) for t in rollout_data["loss_masks"]
        ]
        if "rollout_mask_sums" in rollout_data:
            # Promote precomputed per-rollout mask totals to GPU tensors here
            # (matching loss_masks) so the loss reducer can just divide.
            rollout_data["rollout_mask_sums"] = rollout_data["rollout_mask_sums"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        if "multimodal_train_inputs" in rollout_data:
            # Move multimodal training tensors to GPU in advance
            rollout_data["multimodal_train_inputs"] = [
                (
                    {
                        key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                        for key, value in mm_dict.items()
                    }
                    if mm_dict is not None
                    else None
                )
                for mm_dict in rollout_data["multimodal_train_inputs"]
            ]

        for key in ["rollout_log_probs", "rs_preflight_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                slice_log_prob_with_cp(log_prob, total_length, response_length).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                for log_prob, total_length, response_length in zip(
                    rollout_data[key],
                    rollout_data["total_lengths"],
                    rollout_data["response_lengths"],
                    strict=False,
                )
            ]
        return rollout_data

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from slime.utils.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = prepare_routed_experts_for_routing_replay(
                batch["rollout_routed_experts"],
                batch["tokens"],
                num_experts=self.args.num_experts,
                data_pad_size_multiplier=self.args.data_pad_size_multiplier,
                sequence_parallel=self.args.sequence_parallel,
                allgather_cp=self.args.allgather_cp,
            )

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays)

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
                use_rollout_top_p_replay=True,
            )

    def score_rs_candidates(self, rollout_id: int, rollout_data_ref: Box):
        """Score candidates with the proximal policy and retain logprobs locally."""

        rollout_data = self._get_rollout_data(rollout_data_ref)
        data_iterator = get_data_iterator(rollout_data)
        target_tag = "old_actor" if self.args.keep_old_actor else "actor"
        previous_tag = self._active_model_tag
        if previous_tag != target_tag:
            self._switch_model(target_tag)
        try:
            rollout_data.update(
                self.compute_log_prob(
                    data_iterator,
                    rollout_data["num_microbatches"],
                    store_prefix="",
                )
            )
        finally:
            if previous_tag is not None and self._active_model_tag != previous_tag:
                self._switch_model(previous_tag)

        if not mpu.is_pipeline_last_stage():
            return []

        train_log_probs = rollout_data.get("log_probs")
        if not train_log_probs:
            raise RuntimeError("RS refill preflight did not produce training log probabilities")
        if len(train_log_probs) != len(rollout_data["sample_indices"]):
            raise RuntimeError(
                "RS refill preflight logprob/sample count mismatch: "
                f"{len(train_log_probs)} != {len(rollout_data['sample_indices'])}"
            )
        if len(train_log_probs) != len(rollout_data["group_indices"]):
            raise RuntimeError(
                "RS refill preflight logprob/group count mismatch: "
                f"{len(train_log_probs)} != {len(rollout_data['group_indices'])}"
            )

        original_loss_masks = clone_rs_masks(rollout_data["loss_masks"])
        modified_masks = compute_sequence_rs_masks(
            args=self.args,
            train_log_probs=train_log_probs,
            rollout_log_probs=rollout_data["rollout_log_probs"],
            loss_masks=rollout_data["loss_masks"],
        )

        # Every TP rank executes the gate; only TP rank 0 owns the replicated
        # report and actor-local cache transferred back through Ray.
        if mpu.get_tensor_model_parallel_rank() != 0:
            return []
        if rollout_id in self._rs_candidate_log_probs:
            raise RuntimeError(f"RS candidate logprob cache already exists for rollout_id={rollout_id}")
        if len(original_loss_masks) != len(modified_masks):
            raise RuntimeError(
                "RS preflight gate returned a different sample count: "
                f"original={len(original_loss_masks)}, modified={len(modified_masks)}"
            )
        if len(original_loss_masks) != len(rollout_data["loss_masks"]):
            raise RuntimeError("RS preflight gate mutated the input loss-mask list length")
        validate_final_rs_masks(original_loss_masks, rollout_data["loss_masks"])

        cache_bytes_by_sample = [train_log_prob.numel() * 4 for train_log_prob in train_log_probs]
        candidate_cache_bytes = sum(cache_bytes_by_sample)
        if candidate_cache_bytes > self.args.rs_refill_max_candidate_cache_bytes:
            raise RuntimeError(
                "RS candidate proximal-logprob cache exceeds the per-actor limit before pinned-memory allocation: "
                f"required={candidate_cache_bytes}, limit={self.args.rs_refill_max_candidate_cache_bytes}. "
                "Increase --rs-refill-max-candidate-cache-bytes or reduce the candidate batch/response length."
            )

        stats = []
        for original_mask, modified_mask in zip(original_loss_masks, modified_masks, strict=True):
            valid_tokens = original_mask.sum().to(torch.long)
            if modified_mask.shape == original_mask.shape:
                gate_passed = torch.all(modified_mask.float() == original_mask.float()).to(torch.long)
            else:
                gate_passed = original_mask.new_tensor(0, dtype=torch.long)
            stats.append(torch.stack((valid_tokens, gate_passed)))

        stats = torch.stack(stats)
        if stats.is_cuda:
            stats_cpu = torch.empty_like(stats, device="cpu", pin_memory=True)
            stats_cpu.copy_(stats, non_blocking=True)
            proximal_log_probs = []
            for train_log_prob in train_log_probs:
                cpu_log_prob = torch.empty(
                    train_log_prob.shape,
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=True,
                )
                cpu_log_prob.copy_(train_log_prob.detach(), non_blocking=True)
                proximal_log_probs.append(cpu_log_prob)
            torch.cuda.current_stream(device=stats.device).synchronize()
        else:
            stats_cpu = stats.cpu()
            proximal_log_probs = [
                train_log_prob.detach().float().cpu().contiguous() for train_log_prob in train_log_probs
            ]

        stats_rows = stats_cpu.tolist()
        sample_indices = [int(sample_index) for sample_index in rollout_data["sample_indices"]]
        if len(sample_indices) != len(set(sample_indices)):
            raise RuntimeError("RS preflight received duplicate sample indices on one actor rank")
        self._rs_candidate_log_probs[rollout_id] = dict(zip(sample_indices, proximal_log_probs, strict=True))

        reports = []
        for sample_index, group_index, stats_row, cache_bytes in zip(
            sample_indices,
            rollout_data["group_indices"],
            stats_rows,
            cache_bytes_by_sample,
            strict=True,
        ):
            valid_tokens, gate_passed = stats_row
            reports.append(
                {
                    "sample_index": sample_index,
                    "group_index": int(group_index),
                    "valid_tokens": valid_tokens,
                    "gate_passed": bool(gate_passed),
                    "policy_version": str(self.weight_updater.weight_version),
                    "candidate_cache_bytes": cache_bytes,
                }
            )
        return reports

    def take_rs_candidate_log_probs(self, rollout_id: int, selected_sample_indices: list[int]):
        """Return selected local caches once and discard all other candidates."""

        if not mpu.is_pipeline_last_stage() or mpu.get_tensor_model_parallel_rank() != 0:
            return {}

        selected = [int(sample_index) for sample_index in selected_sample_indices]
        if len(selected) != len(set(selected)):
            raise ValueError("Selected RS sample indices must be unique")
        cache = self._rs_candidate_log_probs.pop(rollout_id, None)
        if cache is None:
            raise RuntimeError(f"No RS candidate logprob cache for rollout_id={rollout_id}")
        return {sample_index: cache[sample_index] for sample_index in selected if sample_index in cache}

    def discard_rs_candidate_log_probs(self, rollout_id: int) -> None:
        """Idempotently discard a candidate cache after coordination fails."""

        self._rs_candidate_log_probs.pop(rollout_id, None)

    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
        if self.args.debug_rollout_only:
            return None

        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        if self.role == "critic":
            result = self.train_critic(rollout_id, rollout_data)
        else:
            self.train_actor(rollout_id, rollout_data, external_data=external_data)
            result = None

        if self.args.offload_train:
            del rollout_data
            self.sleep()

        return result

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch):
        """Train critic and return CPU values (used as old-values for the next actor train)."""
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        # Compute current critic values (used as old_values for value loss and for actor advantages).
        rollout_data.update(forward_only(get_values, self.args, self.model, data_iterator, num_microbatches))

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
            global_batch_sizes,
        )

        if mpu.is_pipeline_last_stage() and "values" in rollout_data:
            from slime.backends.megatron_utils.data import tensors_to_cpu

            return {"values": tensors_to_cpu(rollout_data["values"])}
        return {}

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch, external_data=None) -> None:
        rs_preflight_log_probs = rollout_data.pop("rs_preflight_log_probs", None)
        has_rs_preflight_log_probs = rs_preflight_log_probs is not None
        rs_preflight_response_lengths = None
        rs_preflight_loss_masks = None
        if getattr(self.args, "rs_batch_refill", False):
            if not has_rs_preflight_log_probs:
                raise RuntimeError("RS-refilled training batch is missing its proximal logprob cache")
            if "log_probs" in rollout_data:
                raise RuntimeError("RS-refilled training batch contains conflicting proximal logprob fields")
            sample_count = len(rollout_data["response_lengths"])
            if not (len(rs_preflight_log_probs) == len(rollout_data["loss_masks"]) == sample_count):
                raise RuntimeError(
                    "RS-refilled proximal cache/sample count mismatch: "
                    f"log_probs={len(rs_preflight_log_probs)}, masks={len(rollout_data['loss_masks'])}, "
                    f"response_lengths={sample_count}"
                )
            for sample_position, (log_probs, loss_mask, response_length) in enumerate(
                zip(
                    rs_preflight_log_probs,
                    rollout_data["loss_masks"],
                    rollout_data["response_lengths"],
                    strict=True,
                )
            ):
                if log_probs.shape != loss_mask.shape or log_probs.numel() != int(response_length):
                    raise RuntimeError(
                        "RS-refilled proximal cache shape mismatch: "
                        f"sample={sample_position}, log_probs={tuple(log_probs.shape)}, "
                        f"mask={tuple(loss_mask.shape)}, response_length={response_length}"
                    )
            rollout_data["log_probs"] = rs_preflight_log_probs
            rs_preflight_response_lengths = tuple(rollout_data["response_lengths"])
            rs_preflight_loss_masks = clone_rs_masks(rollout_data["loss_masks"])
        elif has_rs_preflight_log_probs:
            raise RuntimeError("Received an RS preflight cache while --rs-batch-refill is disabled")

        # Create data iterator for log_probs and train.
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )

                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="teacher_",
                        )
                    )

                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                can_reuse_log_probs_in_loss = (
                    len(num_microbatches) == 1
                    and self.args.loss_type == "policy_loss"
                    and self.args.kl_coef == 0
                    and not self.args.use_rollout_logprobs
                    and not self.args.get_mismatch_metrics
                    and not self.args.use_critic
                    and not self.args.keep_old_actor
                    and not self.args.use_opd
                    and (not self.args.use_routing_replay or self.args.use_rollout_routing_replay)
                    and self.args.advantage_estimator != "gspo"
                )
                if (
                    (not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics)
                    and not can_reuse_log_probs_in_loss
                    and not has_rs_preflight_log_probs
                ):
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    if external_data is not None and mpu.is_pipeline_last_stage():
                        values = external_data.get("values")
                        if values is not None:
                            from slime.backends.megatron_utils.data import tensors_to_gpu

                            rollout_data["values"] = tensors_to_gpu(values)
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

            train_metric_utils.log_rollout_data(
                rollout_id,
                self.args,
                rollout_data,
            )

            if has_rs_preflight_log_probs:
                if tuple(rollout_data["response_lengths"]) != rs_preflight_response_lengths:
                    raise RuntimeError("RS-refilled response lengths changed after proximal preflight")
                validate_final_rs_masks(rs_preflight_loss_masks, rollout_data["loss_masks"])

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            # When dumping train debug data but the actor log_probs were not
            # recomputed separately (can_reuse_log_probs_in_loss / use_rollout_logprobs),
            # snapshot them from the training forward so the dump still carries
            # per-sample log_probs — at no extra forward pass.
            capture_log_probs = self.args.save_debug_train_data is not None and "log_probs" not in rollout_data
            if capture_log_probs:
                enable_log_prob_capture()
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    global_batch_sizes,
                )
            if capture_log_probs:
                captured = drain_captured_log_probs()
                # `captured` is non-empty only on the last PP stage running a loss
                # that snapshots log_probs (policy_loss), and then covers every
                # local sample. Key it by this rank's `partition` to land in local
                # sample order; skip otherwise (nothing to place).
                if captured:
                    rollout_data["log_probs"] = [captured[pos] for pos in rollout_data["partition"]]

            self.prof.step(rollout_id=rollout_id)

        train_data_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        train_metric_utils.log_perf_data(
            rollout_id,
            self.args,
            extra_metrics=self.weight_updater.pop_metrics(),
        )

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            self.wake_up()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            save_hf_model_to_path(self.args, Path(self.args.save_hf.format(rollout_id=rollout_id)), self.model)

        if self.args.offload_train:
            self.sleep()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        (
            rollout_engines,
            rollout_engine_lock,
            num_new_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            engine_parallel_configs,
        ) = ray.get(self.rollout_manager.get_updatable_engines_and_lock.remote())

        reconnect_rollout_engines = self.args.offload_train and self.args.use_critic and not self.args.colocate

        if not rollout_engines and not reconnect_rollout_engines:
            if dist.get_rank() == 0:
                logger.info("No updatable SGLang engines are running; skip weight update.")
            return

        if reconnect_rollout_engines:
            self.wake_up()
        elif self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0 or reconnect_rollout_engines:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
                engine_parallel_configs=engine_parallel_configs,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if reconnect_rollout_engines:
            self.sleep()
        elif self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = (
            self.args.load,
            self.args.no_load_optim,
            self.args.no_load_rng,
            self.args.finetune,
            self.args.ckpt_step,
        )
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        (
            self.args.load,
            self.args.no_load_optim,
            self.args.no_load_rng,
            self.args.finetune,
            self.args.ckpt_step,
        ) = old_args

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag
