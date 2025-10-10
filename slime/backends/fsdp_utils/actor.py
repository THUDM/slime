from contextlib import nullcontext
from itertools import accumulate

import ray
import torch
import torch.distributed as dist
from packaging import version
from torch.distributed.tensor import DTensor
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# Import FSDP v2 components based on PyTorch version
if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import fully_shard as FSDP
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import fully_shard as FSDP
else:
    raise ImportError("FSDP v2 not available")

import wandb
from slime.ray.train_actor import TrainRayActor
from slime.utils.data import get_minimum_num_micro_batch_size, process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss
from slime.utils.timer import Timer, timer
from slime.utils.wandb_utils import init_wandb_secondary

from .data_packing import pack_sequences, unpack_sequences
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

    def init(self, args, role, wandb_run_id, with_ref: bool = False):  # type: ignore[override]
        super().init(args, role, wandb_run_id, with_ref)

        # Update rank and world_size for wandb secondary initialization (using actual distributed values)
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        if dist.get_rank() == 0:
            init_wandb_secondary(args, wandb_run_id)

        self.args = args
        torch.manual_seed(args.seed)

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
        self.model = FSDP(model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )

        # TODO: load

        self.weights = {"actor": {}}

        self.ref_model = None
        if with_ref:
            self.load_ref_model(args.ref_load)

        self.update_cpu_params_dict(self.weights["actor"])

        self.weight_updater = (
            UpdateWeightFromTensor(self.args, self.model, self.weights)
            if self.args.colocate
            else UpdateWeightFromDistributed(self.args, self.model)
        )

        # Initialize data packing parameters
        self.max_tokens_per_gpu = args.max_tokens_per_gpu  # From main arguments

        if self.args.offload:
            self.sleep(("model"))

        Timer().start("train_wait")
        self.global_step = 0
        self.micro_step = 0
        return 0

    def sleep(self, tags):
        if not getattr(self.args, "offload", False):
            return
        if torch_memory_saver is not None:
            torch_memory_saver.pause()

    def wake_up(self, tags):
        if not getattr(self.args, "offload", False):
            return
        if torch_memory_saver is not None:
            torch_memory_saver.resume()

    def save_model(self, iteration):
        if self.args.debug_rollout_only:
            return

        raise NotImplementedError()

    def compute_log_prob(
        self,
        model_tag,
        packed_batches,
        store_prefix="",
    ):
        """
        Compute log probabilities using specified model.

        Args:
            model_tag: "actor" for current model, "ref" for reference model
            padded_batches: Input batches
            store_prefix: Prefix for storing results (e.g., "ref_")
        """
        need_restore = False
        if model_tag != "actor" and model_tag in self.weights:
            self.update_cpu_params_dict(self.weights["actor"])
            self.update_gpu_params_dict(self.weights[model_tag])
            self.model.eval()
            need_restore = True

        try:
            rollout_data = {f"{store_prefix}log_probs": []}
            with timer(f"{store_prefix}log_probs") and torch.no_grad():
                for batch in packed_batches:
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
                        logits, batch["tokens"], self.args.rollout_temperature
                    )
            return rollout_data

        finally:
            if need_restore:
                self.update_gpu_params_dict(self.weights["actor"])
                self.model.train()
                torch.cuda.synchronize()

    def packed_data(self, rollout_data):
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

    def train(self, rollout_id, rollout_data_ref):
        Timer().end("train_wait")

        if self.args.offload:
            self.wake_up(("model"))

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        rollout_data = process_rollout_data(self.args, rollout_data_ref, rank, world_size)
        if self.args.advantage_estimator in ["grpo"]:
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

        for metric_key in ["log_probs", "ref_log_probs", "advantages", "returns", "raw_rewards"]:
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

        reported_accum: dict[str, list[torch.Tensor]] = {}
        self.optimizer.zero_grad(set_to_none=True)
        for mbs_id, packed_batch in enumerate(packed_batches):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(
                    input_ids=packed_batch["tokens"].unsqueeze(0),
                    attention_mask=None,
                    position_ids=packed_batch["position_ids"].unsqueeze(0),
                ).logits

            # Handle packed sequences
            log_probs = gather_log_probs_packed(logits, packed_batch["tokens"], packed_batch["cu_seqlens"])
            packed_batch["cur_log_probs"] = log_probs
            unpacked_batches = unpack_sequences(packed_batch)

            if self.args.advantage_estimator == "gspo":
                raise NotImplementedError("implement gspo")
            else:
                old_log_probs = torch.cat([batch["log_probs"] for batch in unpacked_batches], dim=0)
                log_probs = torch.cat([batch["cur_log_probs"] for batch in unpacked_batches], dim=0)
                advantages = torch.cat([batch["advantages"] for batch in unpacked_batches], dim=0)

            loss_masks = [batch["loss_masks"].to(device=log_probs.device) for batch in unpacked_batches]
            response_lengths = [batch["response_lengths"] for batch in unpacked_batches]

            # Ensure device consistency
            ppo_kl = old_log_probs.to(device=log_probs.device) - log_probs
            advantages = advantages.to(device=ppo_kl.device)
            pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, self.args.eps_clip, self.args.eps_clip_high)

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

                rollout_log_probs = torch.cat([batch["rollout_log_probs"] for batch in unpacked_batches], dim=0)
                rollout_log_probs = rollout_log_probs.to(device=log_probs.device)

                tis = torch.exp(old_log_probs - rollout_log_probs)
                ois = (-ppo_kl).exp()
                tis_clip = torch.clamp(
                    tis, min=getattr(self.args, "tis_clip_low", 0.1), max=getattr(self.args, "tis_clip", 2.0)
                )
                tis_clipfrac = tis_clip != tis

                pg_loss = pg_loss * tis_clip

            pg_loss = sum_of_sample_mean(pg_loss, response_lengths, loss_masks)
            pg_clipfrac = sum_of_sample_mean(pg_clipfrac, response_lengths, loss_masks)
            ppo_kl = sum_of_sample_mean(ppo_kl.abs(), response_lengths, loss_masks)

            loss = pg_loss

            if self.args.entropy_coef != 0:
                raise NotImplementedError("implement entropy bonus")

            if self.args.use_kl_loss:
                ref_log_probs = torch.cat([batch["ref_log_probs"] for batch in unpacked_batches], dim=0)
                kl = compute_approx_kl(
                    log_probs,
                    ref_log_probs,
                    kl_loss_type=self.args.kl_loss_type,
                )
                kl_loss = sum_of_sample_mean(kl, response_lengths, loss_masks)

                loss = loss + self.args.kl_loss_coef * kl_loss

            # TODO: report entropy

            reported = {
                "loss": loss.detach(),
                "pg_loss": pg_loss.detach(),
                "pg_clipfrac": pg_clipfrac.detach(),
                "ppo_kl": ppo_kl.detach(),
            }

            if self.args.use_kl_loss:
                reported["kl_loss"] = kl_loss.detach()

            if self.args.use_tis and tis is not None:
                reported["tis"] = sum_of_sample_mean(tis, response_lengths, loss_masks).detach()
                reported["ois"] = sum_of_sample_mean(ois, response_lengths, loss_masks).detach()
                reported["tis_clipfrac"] = sum_of_sample_mean(
                    tis_clipfrac.float(), response_lengths, loss_masks
                ).detach()

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
                reported_accum = {}
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

                    print(f"step {self.global_step}: {log_dict}")

                    if self.args.use_wandb:
                        log_dict["train/step"] = self.global_step
                        wandb.log(log_dict)
                self.global_step += 1

        self.update_cpu_params_dict(self.weights["actor"])

        Timer().start("train_wait")
        return

    def update_weights(self):  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())

        # For colocated mode with sharded updates (full_params=False),
        # we don't need to wake up the entire model
        # The bucket-based approach will load parameters selectively from CPU storage
        # TODO:  Add bucket optimization for from distributed mode
        use_bucket_optimization = self.args.colocate and not getattr(self.weight_updater, "full_params", False)

        if self.args.offload and not use_bucket_optimization:
            # Wake up for distributed mode or full_params mode
            self.wake_up(("model"))

        with torch_memory_saver.disable() if self.args.offload and not torch.version.hip else nullcontext():
            self.weight_updater.update_weights()

        if self.args.offload and not use_bucket_optimization:
            # Sleep for distributed mode or full_params mode
            self.sleep(("model"))

    @torch.no_grad()
    def update_cpu_params_dict(self, params_dict):
        """Copy model parameters from GPU to CPU storage dictionary"""

        state_dict = self.model.state_dict()

        for name, param in state_dict.items():
            if isinstance(param, DTensor):
                param = param.full_tensor()

            if name not in params_dict:
                params_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            params_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def update_gpu_params_dict(self, params_dict):
        """Load parameters from CPU storage dictionary to GPU model"""
        # FSDP v2 doesn't need context managers - load state dict directly
        gpu_state_dict = {name: param.cuda(non_blocking=True) for name, param in params_dict.items()}
        self.model.load_state_dict(gpu_state_dict, strict=True)
        torch.cuda.synchronize()

    def load_ref_model(self, ref_load_path):
        """Load reference model parameters once and store in CPU memory (like Megatron backend)"""
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")

        current_weights = {}
        self.update_cpu_params_dict(current_weights)

        try:
            import os

            if os.path.isdir(ref_load_path):
                temp_ref_model = AutoModelForCausalLM.from_pretrained(
                    ref_load_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )

                # FSDP v2 doesn't need context managers - load state dict directly
                self.model.load_state_dict(temp_ref_model.state_dict(), strict=True)

                del temp_ref_model
                torch.cuda.empty_cache()
            else:
                raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

            self.weights["ref"] = {}
            self.update_cpu_params_dict(self.weights["ref"])

            print(f"Reference model parameters loaded and stored in CPU memory")

        finally:
            self.update_gpu_params_dict(current_weights)


def gather_log_probs(logits: torch.Tensor, input_ids: torch.Tensor, rollout_temperature: float = 1.0) -> torch.Tensor:
    # log_probs: [B, T-1, V]; input_ids: [B, T]
    pred_logits = logits[:, :-1]
    # haoran: whether to apply temperature shifting here?
    if rollout_temperature != 1.0:
        pred_logits = pred_logits / rollout_temperature
    log_probs_all = torch.log_softmax(pred_logits, dim=-1)
    tgt = input_ids[:, 1:].contiguous()
    log_probs = log_probs_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return log_probs


def gather_log_probs_packed(logits: torch.Tensor, input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Gather log probabilities for packed sequences."""
    # Handle batch dimension - logits should be [batch_size, seq_len, vocab_size]
    if logits.dim() == 3:
        # Remove batch dimension for packed sequences
        logits = logits.squeeze(0)
        input_ids = input_ids.squeeze(0)

    # Shift for next-token prediction: logits[:-1] predicts input_ids[1:]
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    targets = input_ids[1:].to(device=log_probs.device)

    # Gather log probs for targets
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Apply mask to exclude first tokens
    return gathered


def sum_of_sample_mean(x: torch.Tensor, response_lengths: list[int], loss_masks: list[torch.Tensor]):

    return sum(
        [
            (x_i * loss_mask_i).sum() / torch.clamp_min(loss_mask_i.sum(), 1)
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks)
        ]
    )
