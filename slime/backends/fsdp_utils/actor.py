from argparse import Namespace
from contextlib import nullcontext
from itertools import accumulate

import ray
import torch
import torch.distributed as dist
import wandb
from packaging import version
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.tensor import DTensor, distribute_tensor
from torch_memory_saver import torch_memory_saver
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils, train_metric_utils
from slime.utils.context_utils import with_defer
from slime.utils.data import get_minimum_num_micro_batch_size, process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss
from slime.utils.ray_utils import Box
from slime.utils.timer import Timer, inverse_timer, timer
from slime.utils.wandb_utils import init_wandb_secondary

from ...utils.profile_utils import TrainProfiler
from . import checkpoint
from .data_packing import pack_sequences, unpack_sequences
from .fsdp_cpu_adam_wrapper import FSDPCPUAdamWrapper
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor


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
    def init(self, args: Namespace, role: str, wandb_run_id: str, with_ref: bool = False) -> int:  # type: ignore[override]
        super().init(args, role, wandb_run_id, with_ref)

        # TODO extract to function
        if args.true_on_policy_mode:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
            from transformers.models.qwen3 import modeling_qwen3

            print("FSDPTrainRayActor call enable_batch_invariant_mode for true-on-policy")
            enable_batch_invariant_mode(
                # In Qwen3, rope `inv_freq_expanded.float() @ position_ids_expanded.float()` uses bmm
                # and disabling it will make it aligned
                enable_bmm=False,
            )

            modeling_qwen3.apply_rotary_pos_emb = torch.compile(dynamic=True)(modeling_qwen3.apply_rotary_pos_emb)

        # Update rank and world_size for wandb secondary initialization (using actual distributed values)
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        if dist.get_rank() == 0:
            init_wandb_secondary(args, wandb_run_id)

        self.args = args
        self.fsdp_full_state_dict_opts = StateDictOptions(
            full_state_dict=True, cpu_offload=getattr(self.args, "fsdp_state_dict_cpu_offload", False)
        )
        torch.manual_seed(args.seed)

        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        if self.args.multimodal_keys:
            self.vlm_processor = AutoProcessor.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

        # Load model
        with torch.autocast(device_type=f"cuda:{torch.cuda.current_device()}"):
            model = AutoModelForCausalLM.from_pretrained(
                self.args.hf_checkpoint,
                trust_remote_code=True,
                attn_implementation=self.args.attn_implementation,
            )
        model.train()

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Create FSDP v2 model using FSDP
        self.model = apply_fsdp2(model)

        if args.optimizer == "deepspeed_cpu_adam":
            optimizer_config = {
                "lr": args.lr,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay,
                "adamw_mode": True,  # Use AdamW mode (decoupled weight decay)
                "fp32_optimizer_states": True,  # Keep optimizer states in FP32
            }

            self.optimizer = FSDPCPUAdamWrapper(optimizer_config, self.model)

        elif args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )

        else:
            raise ValueError(
                f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam', 'deepspeed_cpu_adam'"
            )

        self.global_step = 0
        self.micro_step = 0
        self.weights = {"actor": {}}

        checkpoint_payload = checkpoint.load(self)

        self.ref_model = None
        if with_ref:
            self.load_ref_model(args.ref_load)

        self.update_cpu_params_dict(self.weights["actor"])

        self.weight_updater = (
            UpdateWeightFromTensor(self.args, self.model, self.weights)
            if self.args.colocate
            else UpdateWeightFromDistributed(self.args, self.model, self.weights)
        )

        checkpoint.finalize_load(self, checkpoint_payload)

        # Initialize data packing parameters
        self.max_tokens_per_gpu = args.max_tokens_per_gpu  # From main arguments

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return int(getattr(self.args, "start_rollout_id", 0))

    @timer
    def sleep(self) -> None:
        """Pause CUDA memory for all tracked tensors."""
        if not self.args.offload_train:
            return

        print_memory("before offload model")

        match self.args.offload_train_mode:
            case "tms":
                # Try to avoid this case:
                # * FSDP contains a lot of cached memory and sleep
                # * SGLang resumes and allocate some memory
                # * FSDP resumes but realize there is no enough memory, thus OOM currently, but indeed the cache can be (partially) freed to fulfill requirements
                # TODO: improve it later
                clear_memory()

                torch_memory_saver.pause()
            case "move":
                self.model.cpu()
                move_torch_optimizer(self.optimizer, "cpu")
                clear_memory()
            case _:
                raise NotImplementedError

        torch.cuda.synchronize()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        """Resume CUDA memory for all tracked tensors."""
        if not self.args.offload_train:
            return

        match self.args.offload_train_mode:
            case "tms":
                torch_memory_saver.resume()
            case "move":
                self.model.cuda()
                move_torch_optimizer(self.optimizer, "cuda")
            case _:
                raise NotImplementedError

        torch.cuda.synchronize()
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up model")

    def save_model(self, iteration: int) -> None:
        """Delegate checkpoint saving to the shared checkpoint utilities."""
        if self.args.debug_rollout_only or self.args.save is None:
            return

        checkpoint.save(self, iteration)

    def compute_log_prob(
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
            This method temporarily switches model weights when `model_tag != "actor"`
            and restores the original weights and train mode afterwards.
        """
        need_restore = False
        if model_tag != "actor" and model_tag in self.weights:
            self.update_cpu_params_dict(self.weights["actor"])
            self.update_gpu_params_dict(self.weights[model_tag])
            self.model.eval()
            need_restore = True

        try:
            rollout_data = {f"{store_prefix}log_probs": []}
            with timer(f"{store_prefix}log_probs"), torch.no_grad():
                for batch in self.prof.iterate_train_log_probs(
                    tqdm(packed_batches, desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0)
                ):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        model_args = {
                            "input_ids": batch["tokens"].unsqueeze(0),
                            "position_ids": batch["position_ids"].unsqueeze(0),
                            "attention_mask": None,
                        }
                        if "pixel_values" in batch:
                            model_args["pixel_values"] = batch["pixel_values"]
                        logits = self.model(**model_args).logits
                    batch[f"{store_prefix}log_probs"] = gather_log_probs_packed(
                        logits,
                        batch["tokens"],
                        allow_compile=not self.args.true_on_policy_mode,
                        temperature=self.args.rollout_temperature,
                    )
                    if store_prefix == "":
                        shifted_logits = logits.squeeze(0)[:-1]
                        log_probs_full = torch.log_softmax(shifted_logits, dim=-1)
                        probs = torch.softmax(shifted_logits, dim=-1)
                        entropy = -(probs * log_probs_full).sum(dim=-1)
                        batch["entropy"] = entropy
            return rollout_data

        finally:
            if need_restore:
                self.update_gpu_params_dict(self.weights["actor"])
                self.model.train()
                torch.cuda.synchronize()

    def packed_data(
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
        dp_size = dist.get_world_size()
        local_batch_size = self.args.global_batch_size // dp_size
        assert (
            self.args.global_batch_size % dp_size == 0
        ), f"global_batch_size {self.args.global_batch_size} is not divisible by dp_world_size {dp_size}"
        # Use global_batch_size for splitting when max_tokens_per_gpu is enabled
        if self.args.use_dynamic_batch_size:
            for i in range(0, len(tokens), local_batch_size):
                mbs_size_list.append(
                    get_minimum_num_micro_batch_size(
                        [len(t) for t in rollout_data["tokens"][i : i + local_batch_size]],
                        self.args.max_tokens_per_gpu,
                    )
                )
            num_microbatches = torch.tensor(mbs_size_list, dtype=torch.int, device=torch.cuda.current_device())
            dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX)
            num_microbatches = num_microbatches.tolist()
        else:
            num_microbatches = [
                self.args.global_batch_size // (self.args.micro_batch_size * dist.get_world_size())
            ] * (len(tokens) // local_batch_size)

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
            self._train_core(rollout_id=rollout_id, rollout_data_ref=rollout_data_ref)

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )

    def _train_core(self, rollout_id: int, rollout_data_ref: Box) -> None:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        rollout_data = process_rollout_data(self.args, rollout_data_ref, rank, world_size)
        if self.args.advantage_estimator in ["grpo", "gspo"]:
            rollout_data["advantages"] = rollout_data["returns"] = [
                torch.tensor([rollout_data["rewards"][i]] * rollout_data["response_lengths"][i])
                for i in range(len(rollout_data["rewards"]))
            ]
        else:
            raise NotImplementedError(f"Unsupported advantage_estimator {self.args.advantage_estimator}")

        packed_batches, grad_accum = self.packed_data(rollout_data)
        log_dict = {}

        assert (
            len(grad_accum) > 0
        ), f"Invalid grad_accum {grad_accum} for micro_batch_size {self.args.micro_batch_size} and global_batch_size {self.args.global_batch_size}"

        if "ref" in self.weights:
            self.compute_log_prob("ref", packed_batches, store_prefix="ref_")

        self.compute_log_prob("actor", packed_batches)

        for metric_key in ["log_probs", "ref_log_probs", "advantages", "returns", "raw_reward"]:
            if metric_key not in packed_batches[0]:
                continue
            val = torch.tensor([0.0], device=torch.cuda.current_device())
            for mbs_id, batches in enumerate(packed_batches):
                unpacked_batches = unpack_sequences(batches)
                for unpacked_batch in unpacked_batches:
                    if isinstance(unpacked_batch[metric_key], torch.Tensor):
                        loss_masks_tensor = unpacked_batch["loss_masks"].to(device=torch.cuda.current_device())
                        metric_tensor = unpacked_batch[metric_key].to(device=torch.cuda.current_device())
                        val += (metric_tensor * loss_masks_tensor).sum() / loss_masks_tensor.sum().clamp_min(1)
                    else:
                        val += unpacked_batch[metric_key]
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            log_dict[f"rollout/{metric_key}"] = (
                val / (self.args.n_samples_per_prompt * self.args.rollout_batch_size)
            ).item()
        if dist.get_rank() == 0:
            print(f"rollout {rollout_id}: {log_dict}")
            if self.args.use_wandb:
                log_dict["rollout/step"] = (
                    rollout_id
                    if not self.args.wandb_always_use_train_step
                    else rollout_id
                    * self.args.rollout_batch_size
                    * self.args.n_samples_per_prompt
                    // self.args.global_batch_size
                )
                wandb.log(log_dict)

        with timer("actor_train"):
            reported_accum: dict[str, list[torch.Tensor]] = {}
            self.optimizer.zero_grad(set_to_none=True)
            for mbs_id, packed_batch in self.prof.iterate_train_actor(
                enumerate(tqdm(packed_batches, desc="actor_train", disable=dist.get_rank() != 0))
            ):
                self._train_step(
                    packed_batch=packed_batch,
                    world_size=world_size,
                    reported_accum=reported_accum,
                    mbs_id=mbs_id,
                    grad_accum=grad_accum,
                )

        self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        self.update_cpu_params_dict(self.weights["actor"])

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights
        ):
            if dist.get_rank() == 0:
                print(f"Updating ref model at rollout_id {rollout_id}")
            self.update_cpu_params_dict(self.weights["ref"])

    def _train_step(self, packed_batch, world_size, reported_accum, mbs_id, grad_accum):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.model(
                input_ids=packed_batch["tokens"].unsqueeze(0),
                attention_mask=None,
                position_ids=packed_batch["position_ids"].unsqueeze(0),
            ).logits

        # Handle packed sequences
        log_probs = gather_log_probs_packed(
            logits,
            packed_batch["tokens"],
            allow_compile=not self.args.true_on_policy_mode,
            cu_seqlens=packed_batch["cu_seqlens"],
            temperature=self.args.rollout_temperature,
        )
        packed_batch["cur_log_probs"] = log_probs

        shifted_logits = logits.squeeze(0)[:-1]
        log_probs_full = torch.log_softmax(shifted_logits, dim=-1)
        probs = torch.softmax(shifted_logits, dim=-1)
        entropy = -(probs * log_probs_full).sum(dim=-1)
        packed_batch["entropy"] = entropy
        unpacked_batches = unpack_sequences(packed_batch)

        old_log_probs = torch.cat([batch["log_probs"] for batch in unpacked_batches], dim=0)
        log_probs = torch.cat([batch["cur_log_probs"] for batch in unpacked_batches], dim=0)
        advantages = torch.cat([batch["advantages"] for batch in unpacked_batches], dim=0)
        loss_masks = [batch["loss_masks"].to(device=log_probs.device) for batch in unpacked_batches]
        response_lengths = [batch["response_lengths"] for batch in unpacked_batches]

        advantages = advantages.to(device=log_probs.device)
        ppo_kl = old_log_probs.to(device=log_probs.device) - log_probs

        if self.args.advantage_estimator == "gspo":
            log_ratio_splits = torch.split(ppo_kl, response_lengths, dim=0)

            seq_kls = [
                ((log_ratio_i * mask_i).sum() / mask_i.sum().clamp_min(1))
                for log_ratio_i, mask_i in zip(log_ratio_splits, loss_masks)
            ]

            ppo_kl_list = []
            for seq_kl, length in zip(seq_kls, response_lengths):
                ppo_kl_list.append(seq_kl.expand(length))

            ppo_kl = torch.cat(ppo_kl_list)

        pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, self.args.eps_clip, self.args.eps_clip_high)

        rollout_log_probs = torch.cat([batch["rollout_log_probs"] for batch in unpacked_batches], dim=0)
        rollout_log_probs = rollout_log_probs.to(device=log_probs.device)

        # Apply TIS before sample mean calculation
        if self.args.use_tis:
            # Initialize TIS variables
            tis = None
            tis_clipfrac = None
            ois = None
            # Apply TIS off-policy correction using importance sampling
            assert all(
                "rollout_log_probs" in batch
                and isinstance(batch["rollout_log_probs"], torch.Tensor)
                and batch["rollout_log_probs"].numel() > 0
                for batch in unpacked_batches
            ), "rollout_log_probs must be provided as non-empty torch.Tensor for TIS"

            tis = torch.exp(old_log_probs - rollout_log_probs)
            ois = (-ppo_kl).exp()
            tis_clip = torch.clamp(
                tis, min=getattr(self.args, "tis_clip_low", 0.1), max=getattr(self.args, "tis_clip", 2.0)
            )
            tis_clipfrac = tis_clip != tis

            pg_loss = pg_loss * tis_clip

        assert not self.args.calculate_per_token_loss, "calculate_per_token_loss not yet implemented"
        pg_loss = sum_of_sample_mean(pg_loss, response_lengths, loss_masks)
        pg_clipfrac = sum_of_sample_mean(pg_clipfrac, response_lengths, loss_masks)
        ppo_kl = sum_of_sample_mean(ppo_kl.abs(), response_lengths, loss_masks)

        train_rollout_logprob_abs_diff = (old_log_probs - rollout_log_probs).abs()
        train_rollout_logprob_abs_diff = sum_of_sample_mean(
            train_rollout_logprob_abs_diff, response_lengths, loss_masks
        ).detach()

        entropy = torch.cat([batch["entropy"] for batch in unpacked_batches], dim=0)
        entropy_loss = sum_of_sample_mean(entropy, response_lengths, loss_masks)

        loss = pg_loss - self.args.entropy_coef * entropy_loss

        if self.args.use_kl_loss:
            ref_log_probs = torch.cat([batch["ref_log_probs"] for batch in unpacked_batches], dim=0)
            kl = compute_approx_kl(
                log_probs,
                ref_log_probs,
                kl_loss_type=self.args.kl_loss_type,
            )
            kl_loss = sum_of_sample_mean(kl, response_lengths, loss_masks)

            loss = loss + self.args.kl_loss_coef * kl_loss

        reported = {
            "loss": loss.detach(),
            "pg_loss": pg_loss.detach(),
            "pg_clipfrac": pg_clipfrac.detach(),
            "ppo_kl": ppo_kl.detach(),
            "entropy_loss": entropy_loss.detach(),
            "train_rollout_logprob_abs_diff": train_rollout_logprob_abs_diff,
        }

        if self.args.use_kl_loss:
            reported["kl_loss"] = kl_loss.detach()

        if self.args.use_tis and tis is not None:
            reported["tis"] = sum_of_sample_mean(tis, response_lengths, loss_masks).detach()
            reported["ois"] = sum_of_sample_mean(ois, response_lengths, loss_masks).detach()
            reported["tis_clipfrac"] = sum_of_sample_mean(tis_clipfrac.float(), response_lengths, loss_masks).detach()

        # Scale loss for gradient accumulation
        loss = loss * dist.get_world_size() / self.args.global_batch_size
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
            self.optimizer.zero_grad(set_to_none=True)
            # Aggregate logs
            aggregated = {k: torch.stack(v).sum().item() for k, v in reported_accum.items()}
            # TODO: change this, this is slow.
            reduced_aggregated = [None] * world_size
            dist.all_gather_object(reduced_aggregated, aggregated)
            aggregated = {}
            for k in reported_accum.keys():
                aggregated[k] = sum([r[k] for r in reduced_aggregated]) / (self.args.global_batch_size)
            reported_accum.clear()
            if dist.get_rank() == 0:
                log_dict = {
                    f"train/{k}": (val.item() if torch.is_tensor(val) else val) for k, val in aggregated.items()
                }
                log_dict["train/grad_norm"] = grad_norm

                for gid, group in enumerate(self.optimizer.param_groups):
                    if "lr" in group:
                        log_dict[f"train/lr-pg_{gid}"] = group["lr"]

                kl_info = ""
                if self.args.use_kl_loss and "kl_loss" in aggregated:
                    kl_info = f", kl_loss: {aggregated['kl_loss']:.4f}, kl_penalty: {aggregated['kl_loss'] * self.args.kl_loss_coef:.4f}"
                    print(kl_info)
                print(f"step {self.global_step}: {log_dict}")

                if self.args.use_wandb and wandb is not None:
                    log_dict["train/step"] = self.global_step
                    wandb.log(log_dict)
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

        with (
            torch_memory_saver.disable()
            if self.args.offload_train and self.args.offload_train_mode == "tms" and not torch.version.hip
            else nullcontext()
        ):
            self.weight_updater.update_weights()

    @torch.no_grad()
    def update_cpu_params_dict(self, params_dict: dict[str, torch.Tensor]) -> None:
        """Copy model parameters from GPU to a pinned CPU dictionary.

        Parameters:
            params_dict: Destination mapping from parameter names to CPU tensors.
                Missing entries are allocated with matching shapes and dtypes.
        """

        state_dict = get_model_state_dict(self.model, options=self.fsdp_full_state_dict_opts)

        for name, param in state_dict.items():
            if not torch.is_tensor(param):
                continue

            if name not in params_dict:
                params_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            params_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def update_gpu_params_dict(self, params_dict: dict[str, torch.Tensor]) -> None:
        """Load parameters from a CPU dictionary into the GPU model.

        Parameters:
            params_dict: Source mapping from parameter names to CPU tensors.

        Note:
            This method handles both regular Tensors and DTensors. For DTensors,
            it properly distributes the full tensor according to FSDP sharding.
        """
        # Cache parameter and buffer maps for efficiency
        if not hasattr(self, "_fsdp_param_map"):
            self._fsdp_param_map = dict(self.model.named_parameters())
            self._fsdp_buffer_map = dict(self.model.named_buffers())

        param_map = self._fsdp_param_map
        buffer_map = self._fsdp_buffer_map

        for name, src in params_dict.items():
            if not torch.is_tensor(src):
                continue

            target_param = param_map.get(name)
            if target_param is None:
                target_param = buffer_map.get(name)
                if target_param is None:
                    continue

            dst_tensor = target_param.data

            src_tensor = src.detach()
            if src_tensor.device.type != "cpu":
                src_tensor = src_tensor.to(device=torch.device("cpu"))
            if src_tensor.dtype != dst_tensor.dtype:
                src_tensor = src_tensor.to(dtype=dst_tensor.dtype)

            if isinstance(dst_tensor, DTensor):
                distributed = distribute_tensor(
                    src_tensor.contiguous(),
                    device_mesh=dst_tensor.device_mesh,
                    placements=dst_tensor.placements,
                )
                dst_tensor.copy_(distributed)
            else:
                # Regular tensor: just move to GPU
                dst_tensor.copy_(src_tensor.to(device=dst_tensor.device, non_blocking=True))

        torch.cuda.synchronize()

    def load_ref_model(self, ref_load_path: str | None) -> None:
        """Load reference model weights once and cache them on CPU.

        Parameters:
            ref_load_path: Path to a directory containing a HF checkpoint. If
                None, a ValueError is raised.
        """
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")

        import os

        if os.path.isdir(ref_load_path):
            # Get actor weights for dtype matching
            actor_weights = self.weights["actor"]

            temp_ref_model = AutoModelForCausalLM.from_pretrained(
                ref_load_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
            ref_state_dict = temp_ref_model.state_dict()
            self.weights["ref"] = {}

            for name, tensor in ref_state_dict.items():
                actor_tensor = actor_weights.get(name)
                target_dtype = actor_tensor.dtype if actor_tensor is not None else tensor.dtype
                cpu_tensor = tensor.detach().to(device="cpu", dtype=target_dtype, copy=True)
                self.weights["ref"][name] = cpu_tensor.pin_memory()

            del temp_ref_model
            torch.cuda.empty_cache()
        else:
            raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

        print("Reference model parameters loaded and stored in CPU memory")


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
    logits: torch.Tensor,
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
    if logits.dim() == 3:
        # Remove batch dimension for packed sequences
        logits = logits.squeeze(0)
        input_ids = input_ids.squeeze(0)

    if temperature is not None:
        logits = logits.div(temperature)

    # Shift for next-token prediction: logits[:-1] predicts input_ids[1:]
    shifted_logits = logits[:-1]
    targets = input_ids[1:].to(device=shifted_logits.device)

    # Gather log probs for targets
    selective_log_softmax = selective_log_softmax_compiled if allow_compile else selective_log_softmax_raw
    return selective_log_softmax(shifted_logits, targets)


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
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks)
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


def apply_fsdp2(model):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""

    # Import FSDP v2 components based on PyTorch version
    if version.parse(torch.__version__) >= version.parse("2.6"):
        from torch.distributed.fsdp import fully_shard
    elif version.parse(torch.__version__) >= version.parse("2.4"):
        from torch.distributed._composable.fsdp import fully_shard
    else:
        raise ImportError("FSDP v2 not available")

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    for idx, module in enumerate(modules):
        fully_shard(module)
    fully_shard(model)

    return model
