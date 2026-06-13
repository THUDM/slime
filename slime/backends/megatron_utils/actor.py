import json
import logging
import os
import random
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.misc import Box
from slime.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from slime.utils.routing_replay import RoutingReplay
from slime.utils.timer import Timer, inverse_timer, timer, with_defer
from slime.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import slice_log_prob_with_cp, slice_with_cp
from .data import DataIterator, get_data_iterator, log_perf_data, log_rollout_data
from .hf_checkpoint_saver import save_hf_model_to_path
from .initialize import init, is_megatron_main_rank
from .loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_logits_for_distill, get_values
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

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

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
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
            ),
            single_tag=None,
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        # Load teacher model for Megatron-based on-policy distillation
        if with_opd_teacher and not getattr(args, "use_mopd", False):
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        # Load multiple teacher models for Megatron-based MOPD
        self._mopd_teacher_domains: list[str] = []
        mopd_teacher_mode = getattr(args, "mopd_teacher_mode", "megatron")
        if getattr(args, "use_mopd", False):
            if mopd_teacher_mode == "megatron" and getattr(args, "mopd_teacher_loads", None):
                mopd_teachers = (
                    json.loads(args.mopd_teachers) if isinstance(args.mopd_teachers, str) else args.mopd_teachers
                )
                for i, teacher_cfg in enumerate(mopd_teachers):
                    domain = teacher_cfg["domain"]
                    tag = f"mopd_teacher_{domain}"
                    self._mopd_teacher_domains.append(domain)
                    self.load_other_checkpoint(tag, args.mopd_teacher_loads[i])
                    logger.info(f"Loaded MOPD teacher model for domain '{domain}' from {args.mopd_teacher_loads[i]}")
            elif mopd_teacher_mode == "sglang":
                logger.info(
                    "MOPD SGLang teacher mode: skipping Megatron teacher model loading. "
                    "Teacher data will be collected from SGLang remote servers during rollout."
                )

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            # Prefer HF config vocab_size (which may include model-native padding)
            # over tokenizer vocab_size (which may be smaller, e.g. GPT-OSS).
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        if self.args.colocate:
            assert (
                self.args.update_weight_mode == "full"
            ), "--update-weight-mode=delta is not supported with --colocate"
            update_weight_cls = UpdateWeightFromTensor
        elif self.args.update_weight_mode == "delta":
            # Lazy import: the delta module pulls DeltaEncoding/DeltaParam/DeltaSpec from
            # sglang, which only exist on newer images. Importing eagerly would break old
            # images even when delta mode is unused.
            from .update_weight.update_weight_from_distributed_delta import UpdateWeightFromDistributedDelta

            update_weight_cls = UpdateWeightFromDistributedDelta
        else:
            assert self.args.update_weight_mode == "full"
            if self.args.update_weight_transport == "disk":
                update_weight_cls = UpdateWeightFromDisk
            else:
                assert (
                    self.args.update_weight_mode == "full" and self.args.update_weight_transport == "nccl"
                ), f"unsupported weight sync mode/transport: {self.args.update_weight_mode!r}/{self.args.update_weight_transport!r}"
                update_weight_cls = UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )

        # Ensure actor weights are on GPU and _active_model_tag is correct
        # after loading ref/teacher/mopd_teacher/old_actor checkpoints.
        if self._active_model_tag != "actor":
            self._switch_model("actor")

        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
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
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in rollout_data["loss_masks"]
        ]
        if "rollout_mask_sums" in rollout_data:
            # Promote precomputed per-rollout mask totals to GPU tensors here
            # (matching loss_masks) so the loss reducer can just divide.
            rollout_data["rollout_mask_sums"] = torch.tensor(
                rollout_data["rollout_mask_sums"], dtype=torch.float32, device=torch.cuda.current_device()
            )
        if "multimodal_train_inputs" in rollout_data:
            # Move multimodal training tensors to GPU in advance
            rollout_data["multimodal_train_inputs"] = [
                (
                    {
                        key: (
                            torch.from_numpy(v.copy()).to(device=torch.cuda.current_device())
                            if isinstance(v, np.ndarray)
                            else v.to(device=torch.cuda.current_device())
                        )
                        for key, v in mm_dict.items()
                    }
                    if mm_dict is not None
                    else None
                )
                for mm_dict in rollout_data["multimodal_train_inputs"]
            ]

        if self.args.qkv_format == "bshd":
            # TODO: micro-batch wise dynamic, possibly move to @data.py:get_data_iterator
            max_seq_len = max(rollout_data["total_lengths"])

            # pad to reduce memory fragmentation and maybe make the computation faster
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size

            rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])

        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                torch.tensor(
                    slice_log_prob_with_cp(
                        log_prob,
                        total_length,
                        response_length,
                        self.args.qkv_format,
                        rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    ),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for i, (log_prob, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]

        # Process MOPD teacher log_probs (dict: domain -> list)
        # When teacher data is unavailable (e.g., HTTP request failure), entries
        # may be None. We replace None with -inf tensors so all DP ranks execute
        # the same backward operations, preventing NCCL deadlocks from
        # inconsistent collective calls.
        if "mopd_teacher_log_probs" in rollout_data:
            mopd_lp_dict = rollout_data["mopd_teacher_log_probs"]
            processed = {}
            for domain, lp_list in mopd_lp_dict.items():
                domain_processed = []
                for i, (log_prob, total_length, response_length) in enumerate(
                    zip(
                        lp_list,
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                ):
                    if log_prob is None:
                        # Create a -inf tensor of the correct size as fallback.
                        # -inf log-probs produce zero KL contribution, so this
                        # domain has no effect on the loss for this sample.
                        sliced_len = len(
                            slice_log_prob_with_cp(
                                torch.zeros(response_length),
                                total_length,
                                response_length,
                                self.args.qkv_format,
                                rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                            )
                        )
                        domain_processed.append(
                            torch.full(
                                (sliced_len,), float("-inf"), device=torch.cuda.current_device(), dtype=torch.float32
                            )
                        )
                    else:
                        domain_processed.append(
                            torch.tensor(
                                slice_log_prob_with_cp(
                                    log_prob,
                                    total_length,
                                    response_length,
                                    self.args.qkv_format,
                                    rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                                ),
                                device=torch.cuda.current_device(),
                                dtype=torch.float32,
                            )
                        )
                processed[domain] = domain_processed
            rollout_data["mopd_teacher_log_probs"] = processed
        if "rollout_routed_experts" in rollout_data:
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
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

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        def pad_func(experts, pad):
            _, num_layers, topk = experts.shape
            pad = (
                torch.arange(
                    pad * num_layers * topk,
                    device=experts.device,
                    dtype=experts.dtype,
                ).reshape((pad, num_layers, topk))
                % self.args.num_experts
            )
            return torch.cat([experts, pad], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = batch["rollout_routed_experts"]
            tokens = batch["tokens"]
            assert len(rollout_routed_experts) == len(tokens)
            for a, b in zip(rollout_routed_experts, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            rollout_routed_experts = [pad_func(r, 1) for r in rollout_routed_experts]
            # TODO: maybe extract a common process function for here and get_batch?
            rollout_routed_experts = [slice_with_cp(r, pad_func) for r in rollout_routed_experts]
            rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            pad = (pad_size - rollout_routed_experts.size(0) % pad_size) % pad_size
            if pad != 0:
                rollout_routed_experts = pad_func(rollout_routed_experts, pad)

            if self.args.sequence_parallel:
                seqlen = rollout_routed_experts.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                rollout_routed_experts = rollout_routed_experts[start:end]

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
        return_logits: bool = False,
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            result = forward_only(
                get_logits_for_distill if return_logits else get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )
            return result

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

                # Forward each MOPD teacher model for Megatron-based MOPD
                # Only applies when mopd_teacher_mode == "megatron". In SGLang mode,
                # teacher data is collected during rollout and arrives in rollout_data.
                mopd_teacher_mode = getattr(self.args, "mopd_teacher_mode", "megatron")
                if (
                    getattr(self.args, "use_mopd", False)
                    and mopd_teacher_mode == "megatron"
                    and hasattr(self, "_mopd_teacher_domains")
                    and self._mopd_teacher_domains
                ):
                    mopd_teacher_log_probs = {}
                    mopd_distill_type = getattr(self.args, "mopd_distill_type", "token_level")
                    use_full_vocab = mopd_distill_type == "full_vocab"
                    use_top_k = mopd_distill_type == "top_k"
                    for domain in self._mopd_teacher_domains:
                        tag = f"mopd_teacher_{domain}"
                        if tag in self.weights_backuper.backup_tags:
                            if self.args.use_routing_replay:
                                os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                            self._switch_model(tag)
                            if use_full_vocab or use_top_k:
                                # Full-vocab / top-k mode: get full logits from teacher
                                teacher_result = self.compute_log_prob(
                                    data_iterator,
                                    num_microbatches,
                                    store_prefix=f"mopd_teacher_{domain}_fv_",
                                    return_logits=True,
                                )
                                if use_full_vocab:
                                    # Full-vocab: store all logits [R_i, V_local]
                                    logits_key = f"mopd_teacher_{domain}_fv_logits"
                                    if logits_key in teacher_result:
                                        rollout_data[logits_key] = teacher_result[logits_key]
                                elif use_top_k:
                                    # Top-k: store top-k logits + indices [R_i, k] per sample
                                    topk_k = self.args.mopd_topk_k
                                    logits_list = teacher_result.get(f"mopd_teacher_{domain}_fv_logits", [])
                                    if logits_list:
                                        topk_logits_key = f"mopd_teacher_{domain}_topk_logits"
                                        topk_indices_key = f"mopd_teacher_{domain}_topk_indices"
                                        topk_log_sum_exp_key = f"mopd_teacher_{domain}_topk_log_sum_exp"
                                        topk_logits_list = []
                                        topk_indices_list = []
                                        topk_log_sum_exp_list = []
                                        tp_group = mpu.get_tensor_model_parallel_group()
                                        for sample_logits in logits_list:
                                            # sample_logits: [R_i, V_local]
                                            # Compute log_sum_exp for exact tail mass estimation.
                                            # This avoids the inaccurate uniform tail assumption
                                            # (V - V_eff) / V which over-estimates tail mass
                                            # when k << V, causing KL inflation of ~5+ nats.
                                            local_max = sample_logits.max(dim=-1).values
                                            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)
                                            # Numerically stable log_sum_exp
                                            shifted = sample_logits - local_max.unsqueeze(-1)
                                            local_sum_exp = shifted.exp().sum(dim=-1)
                                            dist.all_reduce(local_sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
                                            log_sum_exp = (local_sum_exp + 1e-20).log() + local_max
                                            topk_log_sum_exp_list.append(log_sum_exp.detach().float())

                                            topk_vals, topk_idx = sample_logits.topk(topk_k, dim=-1)
                                            topk_logits_list.append(topk_vals.detach().float())
                                            topk_indices_list.append(topk_idx.detach().int())
                                        rollout_data[topk_logits_key] = topk_logits_list
                                        rollout_data[topk_indices_key] = topk_indices_list
                                        rollout_data[topk_log_sum_exp_key] = topk_log_sum_exp_list
                            else:
                                # Token-level mode: only need log_probs
                                teacher_result = self.compute_log_prob(
                                    data_iterator,
                                    num_microbatches,
                                    store_prefix=f"mopd_teacher_{domain}_",
                                )
                                lp_key = f"mopd_teacher_{domain}_log_probs"
                                if lp_key in teacher_result:
                                    mopd_teacher_log_probs[domain] = teacher_result[lp_key]
                    if mopd_teacher_log_probs:
                        rollout_data["mopd_teacher_log_probs"] = mopd_teacher_log_probs

                # SGLang MOPD mode: convert rollout-collected top-k data to per-domain batch format
                if getattr(self.args, "use_mopd", False) and mopd_teacher_mode == "sglang":
                    mopd_distill_type = getattr(self.args, "mopd_distill_type", "token_level")
                    if mopd_distill_type == "top_k":
                        # Convert SGLang-sourced top-k data (nested dict format from rollout)
                        # to per-domain batch keys matching the Megatron loss function's expected format.
                        sglang_topk_logits = rollout_data.pop("mopd_teacher_topk_logits", None)
                        sglang_topk_indices = rollout_data.pop("mopd_teacher_topk_indices", None)
                        if sglang_topk_logits and sglang_topk_indices:
                            tp_rank = mpu.get_tensor_model_parallel_rank()
                            tp_size = mpu.get_tensor_model_parallel_world_size()
                            # Use the ORIGINAL vocab_size (not padded_vocab_size) for
                            # TP shard calculations. Megatron's ColumnParallelLinear
                            # output layer dimensions are based on the actual
                            # vocab_size / tp_size (loaded from HF checkpoint),
                            # NOT padded_vocab_size / tp_size. Using padded values
                            # causes local indices to exceed the model's actual
                            # vocab dimension, leading to gather-index-out-of-bounds
                            # errors in the downstream KL computation.
                            #   padded_vocab_size = 249856 → per-shard = 15616
                            #   vocab_size          = 248320 → per-shard = 15520 (actual)
                            #   Overflow range: [15520, 15615] (96 phantom indices)
                            vocab_size = self.args.vocab_size
                            padded_vocab_size = self.args.padded_vocab_size
                            vocab_local_size = vocab_size // tp_size
                            vocab_offset = tp_rank * vocab_local_size
                            topk_k = self.args.mopd_topk_k

                            # Check that SGLang teacher's vocab size is consistent
                            # with the student's vocab_size. If teacher token IDs
                            # exceed the student vocab range, the global→local TP
                            # index conversion will produce silently incorrect results.
                            _vocab_checked = False

                            for domain in sglang_topk_logits:
                                topk_logits_key = f"mopd_teacher_{domain}_topk_logits"
                                topk_indices_key = f"mopd_teacher_{domain}_topk_indices"
                                # Convert each sample's [seq_len][k] Python lists to tensors on GPU
                                topk_logits_list = []
                                topk_indices_list = []
                                for i, (logits_per_sample, indices_per_sample) in enumerate(
                                    zip(sglang_topk_logits[domain], sglang_topk_indices[domain], strict=False)
                                ):
                                    if logits_per_sample is None or indices_per_sample is None:
                                        # Fallback: create zero-contribution tensors so all DP
                                        # ranks execute the same backward operations, preventing
                                        # NCCL deadlocks from inconsistent collective calls.
                                        # Use -inf logits → zero KL divergence contribution.
                                        seq_len = rollout_data["response_lengths"][i]
                                        topk_logits_list.append(
                                            torch.full(
                                                (seq_len, topk_k),
                                                float("-inf"),
                                                device=torch.cuda.current_device(),
                                                dtype=torch.float32,
                                            )
                                        )
                                        topk_indices_list.append(
                                            torch.zeros(
                                                (seq_len, topk_k),
                                                device=torch.cuda.current_device(),
                                                dtype=torch.int64,
                                            )
                                        )
                                    else:
                                        # SGLang returns GLOBAL token IDs, but the Megatron loss
                                        # function (vocab_parallel_topk_reverse_kl) expects LOCAL
                                        # indices within each TP shard's vocab range, with each
                                        # shard having exactly k entries per position.
                                        #
                                        # Strategy: For each position, scatter the global top-k
                                        # entries to the appropriate shard. Each TP rank keeps
                                        # entries whose global token ID falls in its range
                                        # [vocab_offset, vocab_offset + vocab_local_size),
                                        # converts to local index, and pads to k entries with
                                        # local_idx=0, logit=-inf (contributing nothing to KL).
                                        global_indices = torch.tensor(
                                            indices_per_sample, device=torch.cuda.current_device(), dtype=torch.int64
                                        )  # [seq_len, k_global]
                                        global_logits = torch.tensor(
                                            logits_per_sample, device=torch.cuda.current_device(), dtype=torch.float32
                                        )  # [seq_len, k_global]

                                        # Vocab consistency check (once per actor step)
                                        if not _vocab_checked:
                                            _vocab_checked = True
                                            max_token_id = global_indices.max().item()
                                            min_token_id = global_indices.min().item()
                                            logger.info(
                                                f"[MOPD] Vocab sharding: tp_rank={tp_rank}, "
                                                f"tp_size={tp_size}, vocab_size={vocab_size}, "
                                                f"padded_vocab_size={padded_vocab_size}, "
                                                f"vocab_local_size={vocab_local_size}, "
                                                f"vocab_offset={vocab_offset}, topk_k={topk_k}"
                                            )
                                            logger.info(
                                                f"[MOPD] global_indices range=[{min_token_id}, "
                                                f"{max_token_id}], shape={global_indices.shape}"
                                            )
                                            if max_token_id >= vocab_size:
                                                logger.error(
                                                    f"[MOPD] TOKEN ID OVERFLOW! "
                                                    f"max_token_id={max_token_id} >= "
                                                    f"vocab_size={vocab_size}"
                                                )

                                        seq_len = global_indices.size(0)
                                        # Mask for which entries are in this shard
                                        in_shard = (global_indices >= vocab_offset) & (
                                            global_indices < vocab_offset + vocab_local_size
                                        )
                                        # Convert to local indices
                                        local_indices = global_indices - vocab_offset
                                        # Clamp out-of-range indices to 0 (will be overridden by -inf logits)
                                        local_indices = local_indices.clamp(min=0, max=vocab_local_size - 1)

                                        # Build per-shard top-k: assign in-shard entries, pad rest with -inf
                                        # For each position, we need exactly k entries
                                        local_topk_logits = torch.full(
                                            (seq_len, topk_k),
                                            float("-inf"),
                                            device=torch.cuda.current_device(),
                                            dtype=torch.float32,
                                        )
                                        local_topk_indices = torch.zeros(
                                            (seq_len, topk_k), device=torch.cuda.current_device(), dtype=torch.int64
                                        )

                                        # Scatter: for each position, place the in-shard entries into
                                        # the first available slots. We do this row-by-row for clarity.
                                        for row in range(seq_len):
                                            shard_mask = in_shard[row]  # [k_global]
                                            shard_logits = global_logits[row][shard_mask]
                                            shard_local_idx = local_indices[row][shard_mask]
                                            n_in_shard = min(shard_logits.size(0), topk_k)
                                            if n_in_shard > 0:
                                                local_topk_logits[row, :n_in_shard] = shard_logits[:n_in_shard]
                                                local_topk_indices[row, :n_in_shard] = shard_local_idx[:n_in_shard]

                                        # [MOPD] Check local_topk_indices range after conversion
                                        _local_max = local_topk_indices.max().item()
                                        if _local_max >= vocab_local_size:
                                            logger.error(
                                                f"[MOPD] LOCAL INDEX OVERFLOW! sample={i} "
                                                f"max_local={_local_max} >= "
                                                f"vocab_local_size={vocab_local_size}"
                                            )

                                        topk_logits_list.append(local_topk_logits)
                                        topk_indices_list.append(local_topk_indices)
                                rollout_data[topk_logits_key] = topk_logits_list
                                rollout_data[topk_indices_key] = topk_indices_list

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
                    and not getattr(self.args, "use_mopd", False)
                    and not self.args.use_routing_replay
                    and self.args.advantage_estimator != "gspo"
                )
                if (
                    not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics
                ) and not can_reuse_log_probs_in_loss:
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

            log_rollout_data(
                rollout_id,
                self.args,
                rollout_data,
            )

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
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

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

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

        log_perf_data(rollout_id, self.args, extra_metrics=self.weight_updater.pop_metrics())

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

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )

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
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0 and self.weight_updater.weight_version > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

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
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        old_ckpt_step = None
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step
        elif model_tag.startswith("mopd_teacher_"):
            # MOPD teacher checkpoint step: look up from mopd_teacher_ckpt_steps by domain
            domain = model_tag[len("mopd_teacher_") :]
            if getattr(self.args, "mopd_teacher_ckpt_steps", None) is not None:
                mopd_teachers = (
                    json.loads(self.args.mopd_teachers)
                    if isinstance(self.args.mopd_teachers, str)
                    else self.args.mopd_teachers
                )
                for i, t in enumerate(mopd_teachers):
                    if t["domain"] == domain and i < len(self.args.mopd_teacher_ckpt_steps):
                        old_ckpt_step = self.args.ckpt_step
                        self.args.ckpt_step = self.args.mopd_teacher_ckpt_steps[i]
                        break

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if old_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag
