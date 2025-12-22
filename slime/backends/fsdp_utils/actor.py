import logging
import os
import random
from argparse import Namespace
from itertools import accumulate

import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ring_flash_attn import substitute_hf_flash_attn, update_ring_flash_attn_params
from tqdm import tqdm
from transformers import AutoConfig

from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils, train_metric_utils
from slime.utils.context_utils import with_defer
from slime.utils.data import get_minimum_num_micro_batch_size, process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    vanilla_tis_function,
)
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.ray_utils import Box
from slime.utils.timer import Timer, inverse_timer, timer
from slime.utils.tracking_utils import init_tracking

from ...utils import tracking_utils
from ...utils.profile_utils import TrainProfiler
from . import checkpoint
from .data_packing import pack_sequences, pad_packed_sequence_with_cp, unpack_sequences
from .lr_scheduler import get_lr_scheduler
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor

logger = logging.getLogger(__name__)


class FSDPTrainRayActor(TrainRayActor):
    """Simplified TrainRayActor for pure HF+FSDP training.

    Responsibilities:
      * Initialize model/tokenizer on rank0 sequentially to avoid race on cache
      * Wrap model with FSDP
      * Provide minimal train / save / update_weights hooks compatible with existing RayTrainGroup

    Weight update strategy:
      * Rank0 gathers state_dict (full) and broadcasts tensor-by-tensor.
      * For small models this is fine; for larger models consider sharded state_dict type.
    """

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        super().init(args, role, with_ref)

        # Setup device mesh for parallelism (handles both CP and non-CP cases)
        self._setup_device_mesh()
        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": self.dp_size,
        }

        if self.args.debug_rollout_only:
            return 0

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        # Offload train and fsdp cpu offload cannot be used together, fsdp_cpu_offload is more aggressive
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        self._enable_true_on_policy_optimizations(args)
        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True)
                # Vision models have `vision_config` in the config
                if hasattr(self.hf_config, "vision_config"):
                    self.processor = load_processor(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        init_context = self._get_init_weight_context_manager()

        with init_context():
            model = self.get_model_cls().from_pretrained(
                self.args.hf_checkpoint,
                trust_remote_code=True,
                attn_implementation=self.args.attn_implementation,
            )

        model.train()

        full_state = model.state_dict()

        model = apply_fsdp2(model, mesh=self.dp_mesh, cpu_offload=self.fsdp_cpu_offload, args=self.args)

        model = self._fsdp2_load_full_state_dict(
            model, full_state, self.dp_mesh, cpu_offload=True if self.fsdp_cpu_offload else None
        )

        self.model = model

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam'")

        # Initialize LR scheduler
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)

        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        # Create separate ref model if needed (kept in CPU until needed)
        self.ref_model = None
        if with_ref:
            self.ref_model = self._create_ref_model(args.ref_load)

        self.weight_updater = (
            UpdateWeightFromTensor(self.args, self.model)
            if self.args.colocate
            else UpdateWeightFromDistributed(self.args, self.model)
        )

        checkpoint.finalize_load(self, checkpoint_payload)

        # Initialize data packing parameters
        self.max_tokens_per_gpu = args.max_tokens_per_gpu  # From main arguments

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return int(getattr(self.args, "start_rollout_id", 0))

    def get_model_cls(self):
        # Vision models have `vision_config` in the config
        if hasattr(self.hf_config, "vision_config"):
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        else:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM

    def _enable_true_on_policy_optimizations(self, args):
        if args.true_on_policy_mode:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            from .models.qwen3_moe import apply_true_on_policy_patch_for_qwen3_moe

            logger.info("FSDPTrainRayActor call enable_batch_invariant_mode for true-on-policy")
            enable_batch_invariant_mode(
                # In Qwen3, rope `inv_freq_expanded.float() @ position_ids_expanded.float()` uses bmm
                # and disabling it will make it aligned
                enable_bmm=False,
            )

            apply_true_on_policy_patch_for_qwen3_moe()
        else:
            from .models.qwen3_moe_hf import apply_fsdp_moe_patch

            apply_fsdp_moe_patch()

    def _setup_device_mesh(self) -> None:
        """Setup device mesh for parallelism (always called, handles both CP and non-CP cases).

        Creates 2D mesh (dp_size, cp_size) for all cases:
        - When context_parallel_size > 1: hybrid CP + DP
        - When context_parallel_size = 1: pure DP (equivalent to 1D mesh)

        This ensures consistent group management across all parallelism modes.
        """
        from torch.distributed.device_mesh import init_device_mesh

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Use context_parallel_size directly (defaults to 1 for pure DP)
        self.cp_size = self.args.context_parallel_size
        self.dp_size = world_size // self.cp_size

        # Create 2D device mesh: (dp_size, cp_size)
        # Ranks laid out in row-major: mesh[dp_idx, cp_idx] = dp_idx * cp_size + cp_idx
        # - CP groups: consecutive ranks along dim 1, e.g., [0,1], [2,3], [4,5], [6,7]
        # - DP groups: striped ranks along dim 0, e.g., [0,2,4,6], [1,3,5,7]
        # When cp_size=1, this degenerates to pure DP
        self.mesh = init_device_mesh("cuda", mesh_shape=(self.dp_size, self.cp_size), mesh_dim_names=("dp", "cp"))

        # Extract process groups from mesh
        self.dp_group = self.mesh.get_group("dp")  # For FSDP gradient sync, metric reduction
        self.cp_group = self.mesh.get_group("cp")  # For Ring Flash Attention, logit gathering
        self.dp_mesh = self.mesh["dp"]  # For FSDP

        # Compute local ranks within each dimension
        self.dp_rank = rank // self.cp_size
        self.cp_rank = rank % self.cp_size

        logger.info(
            f"[Rank {rank}] Device mesh (2D): world_size={world_size}, "
            f"cp_size={self.cp_size}, dp_size={self.dp_size}"
        )
        logger.info(f"[Rank {rank}] Mesh shape: {self.mesh.shape}, " f"dp_rank={self.dp_rank}, cp_rank={self.cp_rank}")

        # Setup Ring Flash Attention with CP group from mesh (only when cp_size > 1)
        if self.cp_size > 1:
            substitute_hf_flash_attn(self.cp_group, heads_k_stride=1)
            logger.info(f"[Rank {rank}] CP initialized via device mesh")
        else:
            logger.info(f"[Rank {rank}] Pure DP mode (cp_size=1)")

    def _get_init_weight_context_manager(self):
        """Get context manager for model initialization.

        Returns a callable that creates a context manager.
        Uses meta device (no memory allocation) for non-rank-0 processes,
        UNLESS tie_word_embeddings=True (which causes hangs with meta tensors).

        Ref: verl/utils/fsdp_utils.py::get_init_weight_context_manager
        NOTE: tie_word_embedding causes meta_tensor init to hang
        """
        from accelerate import init_empty_weights

        # Check if model uses tied word embeddings (which doesn't work with meta tensors)
        use_meta_tensor = not self.hf_config.tie_word_embeddings

        def cpu_init_weights():
            return torch.device("cpu")

        if use_meta_tensor:
            # Rank 0: CPU, others: meta device (memory efficient for large models)
            return init_empty_weights if dist.get_rank() != 0 else cpu_init_weights
        else:
            logger.info(f"[Rank {dist.get_rank()}] tie_word_embeddings=True, loading full model to CPU on all ranks")
            return cpu_init_weights

    def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
        """Load full state dict into FSDP2 model with efficient broadcast from rank 0.

        This function loads weights from rank 0 and broadcasts to all other ranks,
        avoiding the need for each rank to load the full model from disk.

        Args:
            model: FSDP2-wrapped model
            full_state: State dict (only rank 0 has real weights, others have empty dict)
            device_mesh: Device mesh for FSDP
            cpu_offload: If not None, enables StateDictOptions cpu_offload

        Ref:verl/utils/fsdp_utils.py::fsdp2_load_full_state_dict
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        # Rank 0: move with weights, others: allocate empty tensors on device
        if dist.get_rank() == 0:
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)
        else:
            # to_empty creates tensors on device without initializing memory
            model = model.to_empty(device=torch.cuda.current_device())

        is_cpu_offload = cpu_offload is not None
        options = StateDictOptions(full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True)

        set_model_state_dict(model, full_state, options=options)

        # set_model_state_dict will not broadcast buffers, so we need to broadcast them manually.
        for _name, buf in model.named_buffers():
            dist.broadcast(buf, src=0)

        if is_cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())

        return model

    @timer
    def sleep(self) -> None:
        """Pause CUDA memory for all tracked tensors."""
        if not self.args.offload_train:
            return

        print_memory("before offload model")

        self.model.cpu()
        move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        """Resume CUDA memory for all tracked tensors."""
        if not self.args.offload_train:
            return

        self.model.cuda()
        move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up model")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        """Delegate checkpoint saving to the shared checkpoint utilities."""
        if self.args.debug_rollout_only or self.args.save is None:
            return

        assert not self.args.async_save, "FSDPTrainRayActor does not support async_save yet."
        checkpoint.save(self, rollout_id)

    def _compute_log_prob(
        self,
        model_tag: str,
        packed_batches: list[dict[str, torch.Tensor]],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        """Compute token log-probabilities for a list of packed batches.

        Parameters:
            model_tag: Which parameters to use, e.g. "actor" or "ref".
            packed_batches: A list of packed batch dictionaries produced by
                `pack_sequences`, each containing at least `tokens` and
                `position_ids`; may also include multimodal keys like `pixel_values`.
            store_prefix: Prefix to use for keys in outputs (e.g., "ref_").

        Returns:
            A lightweight dictionary keyed by f"{store_prefix}log_probs". The
            actual per-sequence results are written in-place into each element of
            `packed_batches` under the same key and can be read back by callers.

        Note:
            Uses separate ref model when model_tag == "ref". The ref model is
            loaded from CPU to GPU on-demand and offloaded back after use.
        """
        # Select which model to use
        if model_tag == "ref" and self.ref_model is not None:
            if not self.fsdp_cpu_offload:
                self.model.cpu()
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

            active_model = self.ref_model
            active_model.eval()
        else:
            active_model = self.model

        try:
            rollout_data = {f"{store_prefix}log_probs": []}
            with timer(f"{store_prefix}log_probs"), torch.no_grad():
                for batch in self.prof.iterate_train_log_probs(
                    tqdm(packed_batches, desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0)
                ):
                    model_args = self._get_model_inputs_args(batch)
                    logits = active_model(**model_args).logits.squeeze(0).float()
                    log_probs_result, entropy_result = get_logprob_and_entropy_with_cp(
                        logits=logits,
                        target_tokens=batch["tokens"],
                        cp_rank=self.cp_rank,
                        cp_size=self.cp_size,
                        cp_group=self.cp_group,
                        model_input_ids=model_args["input_ids"],
                        allow_compile=not self.args.true_on_policy_mode,
                        temperature=self.args.rollout_temperature,
                    )
                    batch[f"{store_prefix}log_probs"] = log_probs_result
                    if store_prefix == "":
                        batch["entropy"] = entropy_result
            return rollout_data

        finally:
            # Restore actor model if it was offloaded
            if model_tag == "ref" and self.ref_model is not None:
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

                if not self.fsdp_cpu_offload:
                    self.model.cuda()
                    dist.barrier(group=get_gloo_group())

    def _packed_data(
        self, rollout_data: dict[str, list[torch.Tensor]]
    ) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
        """Pack variable-length sequences for efficient processing.

        Parameters:
            rollout_data: Dictionary of lists containing sequence-level tensors
                such as `tokens`, `loss_masks`, `rewards`, `response_lengths`,
                `advantages`, `returns`, and optional `rollout_log_probs`.

        Returns:
            A pair `(packed_batches, grad_accum)` where `packed_batches` is a list
            of packed batch dictionaries and `grad_accum` lists the micro-batch
            indices at which to perform optimizer steps.
        """
        # Pack sequences efficiently
        tokens = rollout_data["tokens"]

        packed_batches = []
        mbs_size_list = []
        local_batch_size = self.args.global_batch_size // self.dp_size
        assert (
            self.args.global_batch_size % self.dp_size == 0
        ), f"global_batch_size {self.args.global_batch_size} is not divisible by dp_world_size {self.dp_size}"
        # Use global_batch_size for splitting when max_tokens_per_gpu is enabled
        if self.args.use_dynamic_batch_size:
            # In CP mode, CP group shares sequences, so total capacity is max_tokens_per_gpu * cp_size
            max_tokens = self.args.max_tokens_per_gpu
            if self.cp_size > 1:
                max_tokens = max_tokens * self.cp_size

            for i in range(0, len(tokens), local_batch_size):
                mbs_size_list.append(
                    get_minimum_num_micro_batch_size(
                        [len(t) for t in rollout_data["tokens"][i : i + local_batch_size]],
                        max_tokens,
                    )
                )
            num_microbatches = torch.tensor(mbs_size_list, dtype=torch.int, device=torch.cuda.current_device())
            dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=self.dp_group)
            num_microbatches = num_microbatches.tolist()
        else:
            num_microbatches = [self.args.global_batch_size // (self.args.micro_batch_size * self.dp_size)] * (
                len(tokens) // local_batch_size
            )

        start = 0
        for mbs_size in num_microbatches:
            end = start + local_batch_size
            packed_batches.extend(
                pack_sequences(
                    rollout_data["tokens"][start:end],
                    rollout_data["loss_masks"][start:end],
                    rollout_data["rewards"][start:end],
                    rollout_data["raw_reward"][start:end],
                    rollout_data["response_lengths"][start:end],
                    rollout_data["advantages"][start:end],
                    rollout_data["returns"][start:end],
                    rollout_log_probs=(
                        rollout_data["rollout_log_probs"][start:end] if "rollout_log_probs" in rollout_data else None
                    ),
                    multimodal_inputs=(
                        rollout_data["multimodal_inputs"][start:end] if "multimodal_inputs" in rollout_data else None
                    ),
                    num_packs=mbs_size,
                )
            )
            start = end
        grad_accum = list(accumulate(num_microbatches))

        return packed_batches, grad_accum

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        """Run one training update over a rollout batch.

        Parameters:
            rollout_id: Monotonic id for logging.
            rollout_data_ref: A Box handle wrapping a Ray object reference to a
                dictionary with rollout tensors and metadata (e.g., `tokens`,
                `loss_masks`, `rewards`, `response_lengths`, optional
                `rollout_log_probs`, etc.). It will be fetched and partitioned
                by `process_rollout_data` based on data-parallel rank/size.
        """
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            rollout_data = process_rollout_data(self.args, rollout_data_ref, self.dp_rank, self.dp_size)
            if self.args.debug_rollout_only:
                return
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )

    def _log_rollout_data(self, rollout_id: int, rollout_data, packed_batches):
        log_dict = {}
        if "raw_reward" in rollout_data and dist.get_rank() == 0:
            raw_reward_list = rollout_data["raw_reward"]
            if raw_reward_list:
                log_dict["rollout/raw_reward"] = sum(raw_reward_list) / len(raw_reward_list)

        for metric_key in ["log_probs", "rollout_log_probs", "ref_log_probs", "advantages", "returns"]:
            if metric_key not in packed_batches[0]:
                continue
            val = torch.tensor([0.0], device=torch.cuda.current_device())
            for _mbs_id, batches in enumerate(packed_batches):
                unpacked_batches = unpack_sequences(batches)
                for unpacked_batch in unpacked_batches:
                    if isinstance(unpacked_batch[metric_key], torch.Tensor):
                        loss_masks_tensor = unpacked_batch["loss_masks"].to(device=torch.cuda.current_device())
                        metric_tensor = unpacked_batch[metric_key].to(device=torch.cuda.current_device())
                        val += (metric_tensor * loss_masks_tensor).sum() / loss_masks_tensor.sum().clamp_min(1)
                    else:
                        val += unpacked_batch[metric_key]
            dist.all_reduce(val, op=dist.ReduceOp.SUM, group=self.dp_group)
            log_dict[f"rollout/{metric_key}"] = (
                val / (self.args.n_samples_per_prompt * self.args.rollout_batch_size)
            ).item()
        if dist.get_rank() == 0:
            logger.info(f"rollout {rollout_id}: {log_dict}")
            log_dict["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            tracking_utils.log(self.args, log_dict, step_key="rollout/step")

        if self.args.ci_test and self.args.true_on_policy_mode:
            assert log_dict["rollout/log_probs"] == log_dict["rollout/rollout_log_probs"], (
                f"CI check failed: true_on_policy_mode is enabled, but log_probs "
                f"({log_dict['rollout/log_probs']}) != rollout_log_probs "
                f"({log_dict['rollout/rollout_log_probs']})"
            )

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        if self.args.advantage_estimator in ["grpo", "gspo"]:
            rollout_data["advantages"] = rollout_data["returns"] = [
                torch.tensor([rollout_data["rewards"][i]] * rollout_data["response_lengths"][i])
                for i in range(len(rollout_data["rewards"]))
            ]
        else:
            raise NotImplementedError(f"Unsupported advantage_estimator {self.args.advantage_estimator}")

        packed_batches, grad_accum = self._packed_data(rollout_data)

        assert (
            len(grad_accum) > 0
        ), f"Invalid grad_accum {grad_accum} for micro_batch_size {self.args.micro_batch_size} and global_batch_size {self.args.global_batch_size}"

        if self.ref_model is not None:
            self._compute_log_prob("ref", packed_batches, store_prefix="ref_")

        self._compute_log_prob("actor", packed_batches)
        self._log_rollout_data(rollout_id, rollout_data, packed_batches)

        with timer("actor_train"):
            reported_accum: dict[str, list[torch.Tensor]] = {}
            self.optimizer.zero_grad(set_to_none=True)
            for mbs_id, packed_batch in self.prof.iterate_train_actor(
                enumerate(tqdm(packed_batches, desc="actor_train", disable=dist.get_rank() != 0))
            ):
                self._train_step(
                    packed_batch=packed_batch,
                    reported_accum=reported_accum,
                    mbs_id=mbs_id,
                    grad_accum=grad_accum,
                )

        self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        # Update ref model if needed (copy actor weights to ref)
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and self.ref_model is not None
        ):
            if dist.get_rank() == 0:
                logger.info(f"Updating ref model at rollout_id {rollout_id}")
            # Copy actor model state to ref model
            actor_state = self.model.state_dict()
            self.ref_model.load_state_dict(actor_state)
            self.ref_model.cpu()

    def _train_step(self, packed_batch, reported_accum, mbs_id, grad_accum):
        # Prepare model inputs
        model_args = self._get_model_inputs_args(packed_batch)
        logits = self.model(**model_args).logits.squeeze(0).float()

        # Compute log probs and entropy (unified for both CP and non-CP modes)
        log_probs, entropy_result = get_logprob_and_entropy_with_cp(
            logits=logits,
            target_tokens=packed_batch["tokens"],
            cp_rank=self.cp_rank,
            cp_size=self.cp_size,
            cp_group=self.cp_group,
            model_input_ids=model_args["input_ids"],
            allow_compile=not self.args.true_on_policy_mode,
            temperature=self.args.rollout_temperature,
        )
        packed_batch["cur_log_probs"] = log_probs
        packed_batch["entropy"] = entropy_result

        unpacked_batches = unpack_sequences(packed_batch)

        old_log_prob_key = "rollout_log_probs" if self.args.use_rollout_logprobs else "log_probs"
        missing_old_log_probs = [
            idx
            for idx, batch in enumerate(unpacked_batches)
            if old_log_prob_key not in batch or not isinstance(batch[old_log_prob_key], torch.Tensor)
        ]
        if missing_old_log_probs:
            raise KeyError(
                f"{old_log_prob_key} must be provided as torch.Tensor for all microbatches when "
                f"use_rollout_logprobs is set to {self.args.use_rollout_logprobs}. Missing in batches: {missing_old_log_probs}"
            )
        old_log_probs = torch.cat([batch[old_log_prob_key] for batch in unpacked_batches], dim=0)
        log_probs = torch.cat([batch["cur_log_probs"] for batch in unpacked_batches], dim=0)
        advantages = torch.cat([batch["advantages"] for batch in unpacked_batches], dim=0)
        loss_masks = [batch["loss_masks"].to(device=log_probs.device) for batch in unpacked_batches]
        response_lengths = [batch["response_lengths"] for batch in unpacked_batches]

        advantages = advantages.to(device=log_probs.device)
        old_log_probs = old_log_probs.to(device=log_probs.device)
        ppo_kl = old_log_probs - log_probs

        if self.args.use_opsm:
            opsm_mask, opsm_clipfrac = compute_opsm_mask(
                args=self.args,
                full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
                advantages=[batch["advantages"] for batch in unpacked_batches],
                loss_masks=loss_masks,
            )

        if self.args.advantage_estimator == "gspo":
            ppo_kl = compute_gspo_kl(
                full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
                local_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                loss_masks=loss_masks,
            )

        pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, self.args.eps_clip, self.args.eps_clip_high)

        if self.args.use_opsm:
            pg_loss = pg_loss * opsm_mask

        def _has_rollout_log_probs(batch) -> bool:
            rollout_tensor = batch.get("rollout_log_probs")
            return isinstance(rollout_tensor, torch.Tensor) and rollout_tensor.numel() > 0

        has_rollout_log_probs = all(_has_rollout_log_probs(batch) for batch in unpacked_batches)
        rollout_log_probs = (
            torch.cat([batch["rollout_log_probs"] for batch in unpacked_batches], dim=0)
            if has_rollout_log_probs
            else None
        )

        # Apply off-policy correction using importance sampling if enabled
        if self.args.use_tis:
            assert (
                has_rollout_log_probs and rollout_log_probs is not None
            ), "rollout_log_probs must be provided as non-empty torch.Tensor for TIS/MIS"

            train_log_probs_list = list(log_probs.split(response_lengths, dim=0))
            rollout_log_probs_list = list(rollout_log_probs.split(response_lengths, dim=0))
            ois = (-ppo_kl).exp()
            tis_kwargs = {
                "args": self.args,
                "pg_loss": pg_loss,
                "train_log_probs": train_log_probs_list,
                "rollout_log_probs": rollout_log_probs_list,
                "loss_masks": loss_masks,
                "response_lengths": response_lengths,
                "cp_rank": self.cp_rank,
                "cp_size": self.cp_size,
                "cp_group": self.cp_group,
            }

            if self.args.custom_tis_function_path is not None:
                tis_func = load_function(self.args.custom_tis_function_path)
            else:
                tis_func = vanilla_tis_function
            pg_loss, loss_masks, tis_metrics = tis_func(**tis_kwargs)

        if self.args.calculate_per_token_loss:
            pg_loss = sum_of_token(pg_loss, response_lengths, loss_masks)
            pg_clipfrac = sum_of_token(pg_clipfrac, response_lengths, loss_masks)
            ppo_kl = sum_of_token(ppo_kl.abs(), response_lengths, loss_masks)
        else:
            pg_loss = sum_of_sample_mean(pg_loss, response_lengths, loss_masks)
            pg_clipfrac = sum_of_sample_mean(pg_clipfrac, response_lengths, loss_masks)
            ppo_kl = sum_of_sample_mean(ppo_kl.abs(), response_lengths, loss_masks)

        # Only compare rollout vs. train log probs when they originate from different stages.
        train_rollout_logprob_abs_diff = None
        if not self.args.use_rollout_logprobs and rollout_log_probs is not None:
            train_rollout_logprob_abs_diff = (old_log_probs - rollout_log_probs).abs()
            train_rollout_logprob_abs_diff = sum_of_sample_mean(
                train_rollout_logprob_abs_diff, response_lengths, loss_masks
            ).detach()

        entropy = torch.cat([batch["entropy"] for batch in unpacked_batches], dim=0)
        entropy_loss = sum_of_sample_mean(entropy, response_lengths, loss_masks)

        loss = pg_loss - self.args.entropy_coef * entropy_loss

        if self.args.use_kl_loss:
            ref_log_probs = torch.cat([batch["ref_log_probs"] for batch in unpacked_batches], dim=0)
            importance_ratio = None
            if self.args.use_unbiased_kl:
                importance_ratio = torch.exp(log_probs - old_log_probs)
            kl = compute_approx_kl(
                log_probs,
                ref_log_probs,
                kl_loss_type=self.args.kl_loss_type,
                importance_ratio=importance_ratio,
            )
            kl_loss = sum_of_sample_mean(kl, response_lengths, loss_masks)

            loss = loss + self.args.kl_loss_coef * kl_loss

        reported = {
            "loss": loss.detach(),
            "pg_loss": pg_loss.detach(),
            "pg_clipfrac": pg_clipfrac.detach(),
            "ppo_kl": ppo_kl.detach(),
            "entropy_loss": entropy_loss.detach(),
        }

        if train_rollout_logprob_abs_diff is not None:
            reported["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff

        if self.args.use_kl_loss:
            reported["kl_loss"] = kl_loss.detach()

        if self.args.use_opsm:
            reported["opsm_clipfrac"] = opsm_clipfrac

        if self.args.use_tis and tis_metrics:
            reported["ois"] = sum_of_sample_mean(ois, response_lengths, loss_masks).detach()
            for k, v in tis_metrics.items():
                if self.args.calculate_per_token_loss:
                    reported[k] = sum_of_token(v, response_lengths, loss_masks).detach()
                else:
                    reported[k] = sum_of_sample_mean(v, response_lengths, loss_masks).detach()

        # Scale loss for gradient accumulation
        loss = loss * self.dp_size / self.args.global_batch_size
        loss.backward()

        # Accumulate reported metrics (store tensors for later mean)
        for k, v in reported.items():
            reported_accum.setdefault(k, []).append(v)

        if (mbs_id + 1) in grad_accum:
            # TODO: check if the grad norm is global grad norm.
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            # the grad norm used to be of DTensor
            grad_norm = float(grad_norm)

            self.optimizer.step()
            # Update learning rate
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            # Aggregate logs
            aggregated = {k: torch.stack(v).sum().item() for k, v in reported_accum.items()}
            # TODO: change this, this is slow.
            reduced_aggregated = [None] * self.dp_size
            dist.all_gather_object(reduced_aggregated, aggregated, group=self.dp_group)
            aggregated = {}
            for k in reported_accum.keys():
                aggregated[k] = sum([r[k] for r in reduced_aggregated]) / (self.args.global_batch_size)
            reported_accum.clear()
            if dist.get_rank() == 0:
                log_dict = {
                    f"train/{k}": (val.item() if torch.is_tensor(val) else val) for k, val in aggregated.items()
                }
                log_dict["train/grad_norm"] = grad_norm

                # Log learning rate per parameter group; use scheduler's last computed LRs
                lr_values = self.lr_scheduler.get_last_lr()
                for gid, _group in enumerate(self.optimizer.param_groups):
                    log_dict[f"train/lr-pg_{gid}"] = lr_values[gid]

                kl_info = ""
                if self.args.use_kl_loss and "kl_loss" in aggregated:
                    kl_info = f", kl_loss: {aggregated['kl_loss']:.4f}, kl_penalty: {aggregated['kl_loss'] * self.args.kl_loss_coef:.4f}"
                    logger.info(kl_info)
                logger.info(f"step {self.global_step}: {log_dict}")

                log_dict["train/step"] = self.global_step
                tracking_utils.log(self.args, log_dict, step_key="train/step")
            self.global_step += 1

    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        """Synchronize actor weights to rollout engines.

        Handles both colocated and distributed update modes. In offload mode,
        wakes up parameters as needed to perform the update.
        """
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())

        self.weight_updater.update_weights()

        if self.args.ci_test and len(rollout_engines) > 0:
            engine = random.choice(rollout_engines)
            engine_version = ray.get(engine.get_weight_version.remote())
            if str(engine_version) != str(self.weight_updater.weight_version):
                raise RuntimeError(
                    f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                )

        clear_memory()

    def _create_ref_model(self, ref_load_path: str | None):
        """Create and initialize a separate reference model with FSDP2 CPUOffloadPolicy.

        Parameters:
            ref_load_path: Path to a directory containing a HF checkpoint. If
                None, a ValueError is raised.

        Returns:
            FSDP2-wrapped ref model with CPU offload enabled

        Note:
            Creates a separate FSDP2 model instance for the reference model.
            ALWAYS uses CPUOffloadPolicy for the reference model to save memory,
            regardless of the actor model's CPU offload setting.
        """
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")

        if os.path.isdir(ref_load_path):
            logger.info(f"[Rank {dist.get_rank()}] Creating separate ref model from {ref_load_path}")

            init_context = self._get_init_weight_context_manager()

            with init_context():
                ref_model = self.get_model_cls().from_pretrained(
                    ref_load_path,
                    trust_remote_code=True,
                    attn_implementation=self.args.attn_implementation,
                )

            full_state = ref_model.state_dict()

            # Always use CPUOffloadPolicy for reference, let FSDP2 handle the offload. It is faster than model.cpu().
            ref_model = apply_fsdp2(ref_model, mesh=self.dp_mesh, cpu_offload=True, args=self.args)
            ref_model = self._fsdp2_load_full_state_dict(ref_model, full_state, self.dp_mesh, cpu_offload=True)

            logger.info(f"[Rank {dist.get_rank()}] Reference model created with FSDP2 CPUOffloadPolicy")
            return ref_model
        else:
            raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

    def _get_model_inputs_args(self, packed_sequence: dict) -> dict:
        input_ids = packed_sequence["tokens"].unsqueeze(0)
        position_ids = packed_sequence["position_ids"].unsqueeze(0)
        if self.cp_size > 1:

            packed_sequence = pad_packed_sequence_with_cp(packed_sequence, self.cp_size)

            if not packed_sequence["cu_seqlens"].is_cuda:
                packed_sequence["cu_seqlens"] = packed_sequence["cu_seqlens"].cuda()
            cu_seqlens = packed_sequence["cu_seqlens"]
            update_ring_flash_attn_params(cu_seqlens, self.cp_group)

            input_ids = torch.chunk(packed_sequence["tokens"].unsqueeze(0), self.cp_size, dim=1)[self.cp_rank]
            position_ids = torch.chunk(packed_sequence["position_ids"].unsqueeze(0), self.cp_size, dim=1)[self.cp_rank]

        model_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
        }
        if packed_sequence.get("multimodal_inputs"):
            model_args.update(packed_sequence["multimodal_inputs"])
        return model_args


def selective_log_softmax_raw(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Fused version of the common `log_softmax -> gather` operation.

    The fused version of this operation avoids the (potentially large) memory overhead
    of allocating a new tensor to store the full logprobs.

    Parameters:
        logits: Tensor of shape [..., V] containing model logits.
        input_ids: Tensor of shape [...] of token indices whose log-probabilities are gathered.

    Returns:
        Tensor of shape [...] containing the log-probabilities corresponding to `input_ids`.
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


selective_log_softmax_compiled = torch.compile(dynamic=True)(selective_log_softmax_raw)


def gather_log_probs_packed(
    shifted_logits: torch.Tensor,
    input_ids: torch.Tensor,
    allow_compile: bool,
    cu_seqlens: torch.Tensor | float | None = None,
    temperature: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather next-token log probabilities for packed sequences.

    Parameters:
        logits: Model logits of shape [B, T, V] or [T, V].
        input_ids: Token ids of shape [B, T] or [T].
        cu_seqlens: Optional cumulative sequence lengths (unused here). Present
            for API compatibility with callers.

    Returns:
        A tensor of shape [T-1] (or [B, T-1]) with log-probabilities of targets.
    """
    # Handle batch dimension - logits should be [batch_size, seq_len, vocab_size]
    if shifted_logits.dim() == 3:
        # Remove batch dimension for packed sequences
        shifted_logits = shifted_logits.squeeze(0)
        input_ids = input_ids.squeeze(0)

    if temperature is not None:
        shifted_logits = shifted_logits.div(temperature)

    targets = input_ids[1:].to(device=shifted_logits.device)

    # Gather log probs for targets
    selective_log_softmax = selective_log_softmax_compiled if allow_compile else selective_log_softmax_raw
    return selective_log_softmax(shifted_logits, targets)


def get_logprob_and_entropy_with_cp(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    cp_group,
    model_input_ids: torch.Tensor,
    allow_compile: bool,
    temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities and entropy in Context Parallel mode.

    Parameters:
        logits: Model output logits with shape [chunk_size, vocab_size]
        target_tokens: Target tokens with shape [total_seq_len]
        cp_rank: Current CP rank
        cp_size: CP world size
        cp_group: CP communication group
        model_input_ids: Model input_ids (used for the last rank)
        allow_compile: Whether to allow compilation
        temperature: Temperature parameter (optional)

    Returns:
        log_probs: Aggregated log probabilities with shape [total_seq_len - 1]
        entropy: Aggregated entropy with shape [total_seq_len - 1]
    """
    # Fast path for non-CP mode (cp_size=1): avoid unnecessary communication
    if cp_size == 1:
        shifted_logits = logits[:-1, :]
        local_log_probs = gather_log_probs_packed(
            shifted_logits, target_tokens, allow_compile=allow_compile, temperature=temperature
        )
        log_probs_full = torch.log_softmax(shifted_logits, dim=-1)
        probs = torch.softmax(shifted_logits, dim=-1)
        entropy = -(probs * log_probs_full).sum(dim=-1)
        return local_log_probs, entropy

    chunk_size = logits.shape[0]
    tokens_start_index = chunk_size * cp_rank
    tokens_end_index = (
        tokens_start_index + chunk_size + 1 if cp_rank < cp_size - 1 else tokens_start_index + chunk_size
    )

    # For the last rank, remove the last logit
    logits = logits if cp_rank < cp_size - 1 else logits[:-1, :]

    # Get local tokens for current rank
    local_tokens = (
        target_tokens[tokens_start_index:tokens_end_index] if cp_rank < cp_size - 1 else model_input_ids.squeeze(0)
    )

    # Compute local log probs
    local_log_probs = gather_log_probs_packed(
        logits, local_tokens, allow_compile=allow_compile, temperature=temperature
    )

    # Pad for the last rank
    if cp_rank == cp_size - 1:
        local_log_probs = F.pad(local_log_probs, (0, chunk_size - local_log_probs.shape[0]), value=0)

    # Compute entropy
    shifted_logits = logits[:-1, :] if cp_rank == cp_size - 1 else logits
    log_probs_full = torch.log_softmax(shifted_logits, dim=-1)
    probs = torch.softmax(shifted_logits, dim=-1)
    entropy = -(probs * log_probs_full).sum(dim=-1)

    # Pad entropy for the last rank
    if cp_rank == cp_size - 1:
        entropy = F.pad(entropy, (0, chunk_size - entropy.shape[0]), value=0)

    # Merge with a single all_gather: stack as [2, chunk_size]
    stacked_local = torch.stack([local_log_probs, entropy], dim=0)
    gathered_stacked = torch.distributed.nn.functional.all_gather(stacked_local, group=cp_group)

    # Concatenate by effective length (non-last rank=chunk_size, last rank=chunk_size-1)
    lp_parts, ent_parts = [], []
    for r in range(cp_size):
        eff_len = chunk_size if r < cp_size - 1 else max(0, chunk_size - 1)
        if eff_len > 0:
            lp_parts.append(gathered_stacked[r][0][:eff_len])
            ent_parts.append(gathered_stacked[r][1][:eff_len])

    log_probs = torch.cat(lp_parts, dim=0) if lp_parts else local_log_probs.new_zeros((0,))
    entropy_result = torch.cat(ent_parts, dim=0) if ent_parts else entropy.new_zeros((0,))

    # Truncate to global effective length T-1 (packed tokens length is T)
    log_probs = log_probs[: len(target_tokens) - 1]
    entropy_result = entropy_result[: len(target_tokens) - 1]

    return log_probs, entropy_result


def sum_of_sample_mean(x: torch.Tensor, response_lengths: list[int], loss_masks: list[torch.Tensor]) -> torch.Tensor:
    """Compute sum of per-sample means across variable-length responses.

    Parameters:
        x: Flat tensor containing concatenated per-token values across samples.
        response_lengths: Lengths of each sample's response segment in `x`.
        loss_masks: Per-sample masks aligned with `response_lengths`.

    Returns:
        A scalar tensor equal to the sum over samples of the mean value within
        each sample's response segment.
    """
    return sum(
        [
            (x_i * loss_mask_i).sum() / torch.clamp_min(loss_mask_i.sum(), 1)
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
        ]
    )


@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

    torch.cuda.synchronize()


def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)
        args: Arguments containing precision settings (fp16/bf16)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    # Determine precision policy based on args
    param_dtype = torch.bfloat16  # Default to bf16 as before
    reduce_dtype = torch.float32

    if args.fp16:
        param_dtype = torch.float16

    logger.info(f"FSDP MixedPrecision Policy: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}")

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # Apply FSDP to the top-level model
    fully_shard(model, **fsdp_kwargs)

    return model


def sum_of_token(x: torch.Tensor, response_lengths: list[int], loss_masks: list[torch.Tensor]) -> torch.Tensor:
    return sum(
        [
            (x_i * loss_mask_i).sum()
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
        ]
    )
