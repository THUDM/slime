import logging
import os
import random
from argparse import Namespace

import ray
import torch
import torch.distributed as dist
from ring_flash_attn import update_ring_flash_attn_params
from tqdm import tqdm
from transformers import AutoConfig

from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils, train_metric_utils
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.misc import Box
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.timer import Timer, inverse_timer, timer, with_defer

from ...utils.profile_utils import TrainProfiler
from ..training_utils.ci_utils import check_grad_norm
from ..training_utils.data import DataIterator, get_batch, get_data_iterator, get_rollout_data
from ..training_utils.log_utils import (
    aggregate_forward_results,
    aggregate_train_losses,
    log_rollout_data,
    log_train_step,
)
from ..training_utils.loss import compute_advantages_and_returns, get_log_probs_and_entropy, loss_function
from . import checkpoint
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state
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

        # Setup ParallelState for both CP and non-CP cases
        self.parallel_state = create_fsdp_parallel_state(args)

        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": self.parallel_state.dp_size,
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

        model = apply_fsdp2(model, mesh=self.parallel_state.dp_mesh, cpu_offload=self.fsdp_cpu_offload, args=self.args)

        model = self._fsdp2_load_full_state_dict(
            model, full_state, self.parallel_state.dp_mesh, cpu_offload=True if self.fsdp_cpu_offload else None
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
        data_iterator: DataIterator,
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        """Compute token log-probabilities using data iterator.

        Parameters:
            model_tag: Which parameters to use, e.g. "actor" or "ref".
            data_iterator: DataIterator providing micro-batches.
            num_microbatches: List of number of microbatches per step.
            store_prefix: Prefix to use for keys in outputs (e.g., "ref_").

        Returns:
            A lightweight dictionary keyed by f"{store_prefix}log_probs".

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
            forward_data_store = []
            data_iterator.reset()

            with timer(f"{store_prefix}log_probs"), torch.no_grad():
                num_steps_per_rollout = len(num_microbatches)
                for step_id in range(num_steps_per_rollout):
                    for _ in self.prof.iterate_train_log_probs(
                        tqdm(
                            range(num_microbatches[step_id]),
                            desc=f"{store_prefix}log_probs",
                            disable=dist.get_rank() != 0,
                        )
                    ):
                        forward_only_keys = [
                            "tokens",
                            "loss_masks",
                            "multimodal_train_inputs",
                            "total_lengths",
                            "response_lengths",
                            "max_seq_lens",
                        ]
                        batch = get_batch(
                            data_iterator,
                            forward_only_keys,
                            self.parallel_state,
                            self.args.data_pad_size_multiplier,
                            self.args.qkv_format,
                            get_position_ids=True,
                        )

                        model_args = self._get_model_inputs_args(batch)
                        logits = active_model(**model_args).logits.float()

                        result = get_log_probs_and_entropy(
                            logits=logits,
                            args=self.args,
                            parallel_state=self.parallel_state,
                            unconcat_tokens=batch["unconcat_tokens"],
                            total_lengths=batch["total_lengths"],
                            response_lengths=batch["response_lengths"],
                            with_entropy=(store_prefix == ""),
                            max_seq_lens=batch.get("max_seq_lens", None),
                        )

                        batch_result = {
                            f"{store_prefix}log_probs": result["log_probs"],
                        }
                        if store_prefix == "" and "entropy" in result:
                            batch_result["entropy"] = result["entropy"]
                        forward_data_store.append(batch_result)

            rollout_data = aggregate_forward_results(forward_data_store, data_iterator, self.args, store_prefix)

            return rollout_data

        finally:
            # Restore actor model if it was offloaded
            if model_tag == "ref" and self.ref_model is not None:
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

                if not self.fsdp_cpu_offload:
                    self.model.cuda()
                    dist.barrier(group=get_gloo_group())

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
            rollout_data = get_rollout_data(self.args, rollout_data_ref, self.parallel_state)
            if self.args.debug_rollout_only:
                return
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, self.parallel_state, rollout_data)
        data_iterator = data_iterator[0]

        assert (
            len(num_microbatches) > 0
        ), f"Invalid num_microbatches {num_microbatches} for micro_batch_size {self.args.micro_batch_size} and global_batch_size {self.args.global_batch_size}"

        if self.ref_model is not None:
            ref_results = self._compute_log_prob("ref", data_iterator, num_microbatches, store_prefix="ref_")
            rollout_data.update(ref_results)

        actor_results = self._compute_log_prob("actor", data_iterator, num_microbatches)
        rollout_data.update(actor_results)

        compute_advantages_and_returns(self.args, self.parallel_state, rollout_data)

        log_rollout_data(rollout_id, self.args, rollout_data, self.parallel_state)

        with timer("actor_train"):
            data_iterator.reset()
            num_steps_per_rollout = len(num_microbatches)

            for step_id in range(num_steps_per_rollout):
                self.optimizer.zero_grad(set_to_none=True)

                losses_reduced = []
                for _ in self.prof.iterate_train_actor(
                    tqdm(range(num_microbatches[step_id]), desc="actor_train", disable=dist.get_rank() != 0)
                ):
                    batch = get_batch(
                        data_iterator,
                        [
                            "tokens",
                            "loss_masks",
                            "multimodal_train_inputs",
                            "total_lengths",
                            "response_lengths",
                            "max_seq_lens",
                            "log_probs",
                            "advantages",
                            "returns",
                            "ref_log_probs",
                            "rollout_log_probs",
                        ],
                        self.parallel_state,
                        self.args.data_pad_size_multiplier,
                        self.args.qkv_format,
                        get_position_ids=True,
                    )

                    log_dict = self._train_step(
                        batch=batch,
                        step_id=step_id,
                        num_microbatches=num_microbatches[step_id],
                    )
                    losses_reduced.append(log_dict)

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                grad_norm = grad_norm.full_tensor().item()

                self.optimizer.step()
                self.lr_scheduler.step()

                if self.args.ci_test:
                    check_grad_norm(
                        args=self.args,
                        grad_norm=grad_norm,
                        rollout_id=rollout_id,
                        step_id=step_id,
                        role="actor",
                        rank=self.parallel_state.dp_cp_rank,
                    )

                loss_dict = aggregate_train_losses(losses_reduced, self.parallel_state)

                extra_metrics = {}
                for param_group_id, param_group in enumerate(self.optimizer.param_groups):
                    extra_metrics[f"lr-pg_{param_group_id}"] = param_group["lr"]

                log_train_step(
                    args=self.args,
                    loss_dict=loss_dict,
                    grad_norm=grad_norm,
                    rollout_id=rollout_id,
                    step_id=step_id,
                    num_steps_per_rollout=num_steps_per_rollout,
                    role="actor",
                    extra_metrics=extra_metrics,
                )

        self.prof.step(rollout_id=rollout_id)

        if self.args.save_debug_train_data is not None:
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

    def _train_step(self, batch, step_id, num_microbatches):
        # Prepare model inputs
        model_args = self._get_model_inputs_args(batch)
        logits = self.model(**model_args).logits.float()

        loss, normalizer, log_dict = loss_function(
            args=self.args,
            parallel_state=self.parallel_state,
            batch=batch,
            num_microbatches=num_microbatches,
            logits=logits,
            apply_megatron_loss_scaling=False,
        )

        loss.backward()

        return log_dict

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
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

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
            ref_model = apply_fsdp2(ref_model, mesh=self.parallel_state.dp_mesh, cpu_offload=True, args=self.args)
            ref_model = self._fsdp2_load_full_state_dict(
                ref_model, full_state, self.parallel_state.dp_mesh, cpu_offload=True
            )

            logger.info(f"[Rank {dist.get_rank()}] Reference model created with FSDP2 CPUOffloadPolicy")
            return ref_model
        else:
            raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

    def _get_model_inputs_args(self, batch: dict) -> dict:
        input_ids = batch["tokens"]
        position_ids = batch["position_ids"]

        if self.parallel_state.cp_size > 1:
            if "cu_seqlens" in batch:
                cu_seqlens = batch["cu_seqlens"]
                if not cu_seqlens.is_cuda:
                    cu_seqlens = cu_seqlens.cuda()
                update_ring_flash_attn_params(cu_seqlens, self.cp_group)

            input_ids = torch.chunk(input_ids, self.parallel_state.cp_size, dim=1)[self.parallel_state.cp_rank]
            position_ids = torch.chunk(position_ids, self.parallel_state.cp_size, dim=1)[self.parallel_state.cp_rank]

        model_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
        }

        if batch.get("multimodal_train_inputs"):
            model_args.update(batch["multimodal_train_inputs"])

        return model_args


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
