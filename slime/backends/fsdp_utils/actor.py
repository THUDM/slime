from contextlib import nullcontext

import torch
import torch.distributed as dist
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

import wandb
from slime.ray.train_actor import TrainRayActor
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss
from slime.utils.wandb_utils import init_wandb_secondary
from slime.utils.timer import Timer, timer

from .update_weight_utils import UpdateWeightFromTensor


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
        with torch.device(f"cuda:{torch.cuda.current_device()}"):
            model = AutoModelForCausalLM.from_pretrained(
                self.args.hf_checkpoint,
                trust_remote_code=True,
            )
        model.train()

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # TODO: set correct auto_wrap_policy
        auto_wrap_policy = None

        self.model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy[self.args.fsdp_sharding_strategy],
            cpu_offload=self.args.fsdp_cpu_offload,
            forward_prefetch=self.args.fsdp_forward_prefetch,
            backward_prefetch=self.args.fsdp_backward_prefetch,
            limit_all_gathers=self.args.fsdp_limit_all_gathers,
        )

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

        self.weight_updator = UpdateWeightFromTensor(self.args, self.model)

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

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines

        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        self.weight_updator.connect_rollout_engines(rollout_engines, rollout_engine_lock)
        dist.barrier(group=get_gloo_group())

    def compute_log_prob(
        self,
        model_tag,
        padded_batches,
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
                for batch in padded_batches:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        model_args = {"input_ids": batch["tokens"]}
                        if "pixel_values" in batch:
                            model_args["pixel_values"] = batch["pixel_values"]
                    logits = self.model(**model_args).logits
                    batch[f"{store_prefix}log_probs"] = gather_log_probs(logits, batch["tokens"], self.args.rollout_temperature)
            return rollout_data
            
        finally:
            if need_restore:
                self.update_gpu_params_dict(self.weights["actor"])
                self.model.train()
                torch.cuda.synchronize()

    def pad_and_move_to_device(self, rollout_data):
        tokens = rollout_data["tokens"]
        loss_masks = rollout_data["loss_masks"]
        prompts = rollout_data.get("prompt", [[] for _ in range(len(tokens))])

        padded_batches = []
        for i in range(0, len(tokens), self.args.micro_batch_size):
            batch_tokens = tokens[i : i + self.args.micro_batch_size]
            batch_loss_masks = loss_masks[i : i + self.args.micro_batch_size]
            batch_prompts = prompts[i : i + self.args.micro_batch_size]
            max_len = max(len(t) for t in batch_tokens)
            padded_tokens = [t + [self.tokenizer.pad_token_id] * (max_len - len(t)) for t in batch_tokens]
            padded_loss_masks = [
                # -1 because its the loss mask for logprob
                [0] * (len(t) - len(l) - 1) + l + [0] * (max_len - len(t))
                for l, t in zip(batch_loss_masks, batch_tokens)
            ]
            batch = {
                "tokens": torch.tensor(padded_tokens, dtype=torch.long, device=torch.cuda.current_device()),
                "loss_masks": torch.tensor(padded_loss_masks, dtype=torch.int, device=torch.cuda.current_device()),
                "rewards": torch.tensor(
                    rollout_data["rewards"][i : i + self.args.micro_batch_size],
                    dtype=torch.float,
                    device=torch.cuda.current_device(),
                ),
                "raw_reward": rollout_data["raw_reward"][i : i + self.args.micro_batch_size],
            }

            if self.args.multimodal_keys:
                processed_media = {}
                for sample_prompt in batch_prompts:
                    for media_part in sample_prompt:
                        media_type = media_part.get("type")

                        if media_type == "image":
                            path = media_part.get("path")
                            if path:
                                if "pixel_values" not in processed_media:
                                    processed_media["pixel_values"] = []
                                image = Image.open(path).convert("RGB")
                                inputs = self.vlm_processor(images=image, return_tensors="pt")
                                processed_media["pixel_values"].append(inputs.pixel_values)

                # Stack and move all processed media to the GPU for the batch
                for key, tensor_list in processed_media.items():
                    if tensor_list:
                        batch[key] = torch.cat(tensor_list).to(
                            device=torch.cuda.current_device(), dtype=torch.bfloat16
                        )

            padded_batches.append(batch)
        return padded_batches

    def train(self, rollout_id, rollout_data_ref):  # type: ignore[override]
        Timer().end("train_wait")

        if self.args.offload:
            self.wake_up(("model"))

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        rollout_data = process_rollout_data(self.args, rollout_data_ref, rank, world_size)
        padded_batches = self.pad_and_move_to_device(rollout_data)

        grad_accum = self.args.global_batch_size // (self.args.micro_batch_size * world_size)
        assert (
            grad_accum > 0
        ), f"Invalid grad_accum {grad_accum} for micro_batch_size {self.args.micro_batch_size} and global_batch_size {self.args.global_batch_size}"

        if "ref" in self.weights:
            self.compute_log_prob("ref", padded_batches, store_prefix="ref_")

        self.compute_log_prob("actor", padded_batches)

        # TODO: compute rewards and adv for t
        for batch in padded_batches:
            if self.args.advantage_estimator in ["grpo", "gspo"]:
                batch["advantages"] = batch["returns"] = batch["rewards"].expand_as(batch["log_probs"])
            else:
                raise NotImplementedError(f"Unsupported advantage_estimator {self.args.advantage_estimator}")

        log_dict = {}
        
        for key in ["log_probs", "ref_log_probs", "advantages", "returns", "raw_reward"]:
            if key not in padded_batches[0]:
                continue
            val = torch.tensor([0.0], device=torch.cuda.current_device())
            for batch in padded_batches:
                if isinstance(batch[key], torch.Tensor):
                    val += per_sample_mean(batch[key], batch["loss_masks"]).item()
                else:
                    val += sum(batch[key])
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            log_dict[f"rollout/{key}"] = (val / len(padded_batches) / world_size).item()

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
        for mbs_id, batch in enumerate(padded_batches):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids=batch["tokens"]).logits
            log_probs = gather_log_probs(logits, batch["tokens"], self.args.rollout_temperature)

            if self.args.advantage_estimator == "gspo":
                raise NotImplementedError("implement GSPO")

            ppo_kl = batch["log_probs"] - log_probs
            pg_loss, pg_clipfrac = compute_policy_loss(
                ppo_kl, batch["advantages"], self.args.eps_clip, self.args.eps_clip_high
            )

            pg_loss = per_sample_mean(pg_loss, batch["loss_masks"])
            pg_clipfrac = per_sample_mean(pg_clipfrac, batch["loss_masks"])
            ppo_kl = per_sample_mean(ppo_kl.abs(), batch["loss_masks"])

            loss = pg_loss

            if self.args.use_tis:
                raise NotImplementedError("implement TIS")

            if self.args.entropy_coef != 0:
                raise NotImplementedError("implement entropy bonus")

            if self.args.use_kl_loss:
                kl = compute_approx_kl(
                    log_probs,
                    batch["ref_log_probs"],
                    kl_loss_type=self.args.kl_loss_type,
                )
                kl_loss = per_sample_mean(kl, batch["loss_masks"])

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

            loss = loss / grad_accum
            loss.backward()

            for k, v in reported.items():
                reported_accum.setdefault(k, []).append(v)

            if (mbs_id + 1) % grad_accum == 0:
                # TODO: check if the grad norm is global grad norm.
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                aggregated = {}
                for k, v in reported_accum.items():
                    if k in ["kl_loss"]:  
                        kl_values = torch.stack(v)
                        aggregated[k] = (kl_values * self.args.micro_batch_size).sum().item()
                    else:
                        aggregated[k] = torch.stack(v).mean().item()
                # TODO: change this, this is slow.
                reduced_aggregated = [None] * world_size
                dist.all_gather_object(reduced_aggregated, aggregated)
                aggregated = {}
                for k in reported_accum.keys():
                    if k in ["kl_loss"]:
                        total_kl = sum([r[k] for r in reduced_aggregated])
                        aggregated[k] = total_kl / self.args.global_batch_size
                    else:
                        aggregated[k] = sum([r[k] for r in reduced_aggregated]) / world_size
                reported_accum = {}
                if dist.get_rank() == 0:
                    log_dict = {
                        f"train/{k}": (val.item() if torch.is_tensor(val) else val) for k, val in aggregated.items()
                    }
                    log_dict["train/grad_norm"] = grad_norm.item() if not isinstance(grad_norm, float) else grad_norm

                    for gid, group in enumerate(self.optimizer.param_groups):
                        if "lr" in group:
                            log_dict[f"train/lr-pg_{gid}"] = group["lr"]
                    
                    kl_info = ""
                    if self.args.use_kl_loss and "kl_loss" in aggregated:
                        kl_info = f", kl_loss: {aggregated['kl_loss']:.4f}, kl_penalty: {aggregated['kl_loss'] * self.args.kl_loss_coef:.4f}"
                    
                    print(f"step {self.global_step}: loss: {aggregated.get('loss', 0):.4f}, pg_loss: {aggregated.get('pg_loss', 0):.4f}{kl_info}")
                    print(f"step {self.global_step} full: {log_dict}")
                    
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

        if self.args.offload:
            # TODO: don't wake up here
            self.wake_up(("model"))

        with torch_memory_saver.disable() if self.args.offload and not torch.version.hip else nullcontext():
            self.weight_updator.update_weights()

        if self.args.offload:
            # TODO: don't wake up here
            self.sleep(("model"))

    @torch.no_grad()
    def update_cpu_params_dict(self, params_dict):
        """Copy model parameters from GPU to CPU storage dictionary"""
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            state_dict = self.model.state_dict()
            
        for name, param in state_dict.items():
            if name not in params_dict:
                params_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            params_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def update_gpu_params_dict(self, params_dict):
        """Load parameters from CPU storage dictionary to GPU model"""
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            gpu_state_dict = {name: param.cuda(non_blocking=True) for name, param in params_dict.items()}
            self.model.load_state_dict(gpu_state_dict, strict=True)
        torch.cuda.synchronize()

    def load_ref_model(self, ref_load_path):
        """Load reference model parameters once and store in CPU memory (like Megatron backend)"""
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")
        
        print(f"Loading reference model from {ref_load_path}")
        
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
                
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
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


def per_sample_mean(x, loss_mask):
    return ((x * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1)).mean()