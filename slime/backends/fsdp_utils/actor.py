from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import wandb
from slime.backends.utils.data import process_rollout_data
from slime.ray.train_actor import TrainRayActor
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss
from slime.utils.timer import Timer, timer

from .update_weight_utils import UpdateWeightFromTensor
from .data_packing import pack_sequences


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
        self.args = args
        torch.manual_seed(args.seed)
        

        # Serialize tokenizer/config loading across ranks to avoid HF cache race
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

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

        self.ref_model = None
        # TODO: support ref model
        if with_ref:
            raise NotImplementedError()

        self.weight_updator = UpdateWeightFromTensor(self.args, self.model)
        
        # Initialize data packing parameters
        self.max_seq_len = getattr(args, 'max_seq_len', 8192)
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

    def save_model(self, iteration, with_optimizer=True):
        if self.args.debug_rollout_only:
            return

        raise NotImplementedError()

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines

        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        self.weight_updator.connect_rollout_engines(rollout_engines, rollout_engine_lock)
        dist.barrier(group=get_gloo_group())

    def _process_packed_sequences_micro_batch(self, packed_tokens, cu_seqlens, micro_batch_size=4):
        """Process packed sequences in micro-batches to reduce memory usage."""
        num_sequences = len(cu_seqlens) - 1
        total_tokens = len(packed_tokens)
        
        # Pre-allocate output tensor to ensure exact size match
        # Get vocab size from the model (handling FSDP wrapper)
        if hasattr(self.model, '_fsdp_wrapped_module'):
            vocab_size = self.model._fsdp_wrapped_module.config.vocab_size
        elif hasattr(self.model, 'module'):
            vocab_size = self.model.module.config.vocab_size
        else:
            vocab_size = self.model.config.vocab_size
        all_logits = torch.zeros(total_tokens, vocab_size, dtype=torch.float16, device=packed_tokens.device)
        
        # Process in micro-batches
        for mb_start in range(0, num_sequences, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, num_sequences)
            
            # Extract sequences for this micro-batch
            sequences = []
            seq_positions = []  # Track where each sequence goes in the output
            max_len = 0
            
            for i in range(mb_start, mb_end):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                seq = packed_tokens[start:end]
                sequences.append(seq)
                seq_positions.append((start, end))
                max_len = max(max_len, len(seq))
            
            # Create padded batch
            batch_size = len(sequences)
            padded_input_ids = torch.zeros(batch_size, max_len, dtype=packed_tokens.dtype, device=packed_tokens.device)
            attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=packed_tokens.device)
            
            for i, seq in enumerate(sequences):
                seq_len = len(seq)
                padded_input_ids[i, :seq_len] = seq
                attention_mask[i, :seq_len] = 1
            
            # Process this micro-batch
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(
                    input_ids=padded_input_ids,
                    attention_mask=attention_mask
                ).logits
            
            # Place logits in the correct positions in the output tensor
            for i, (start, end) in enumerate(seq_positions):
                seq_len = end - start
                all_logits[start:end] = logits[i, :seq_len].float()
            
            # Clear intermediate tensors to save memory
            del padded_input_ids, attention_mask, logits
        
        return all_logits

    def compute_log_prob(
        self,
        model_tag,
        padded_batches,
        store_prefix="",
    ):
        rollout_data = {f"{store_prefix}log_probs": []}
        with timer(f"{store_prefix}log_probs") and torch.no_grad():
            for batch in padded_batches:
                # Process packed sequences in micro-batches to save memory
                logits = self._process_packed_sequences_micro_batch(
                    batch["tokens"],
                    batch["cu_seqlens"],
                    micro_batch_size=2  # Reduced for lower memory usage
                )
                batch[f"{store_prefix}log_probs"] = gather_log_probs_packed(
                    logits, batch["tokens"], batch["cu_seqlens"]
                )
        return rollout_data

    def pad_and_move_to_device(self, rollout_data):
        # Pack sequences efficiently
        packs = pack_sequences(
            tokens=rollout_data["tokens"],
            loss_masks=rollout_data["loss_masks"],
            rewards=rollout_data["rewards"],
            raw_rewards=rollout_data["raw_reward"],
            max_seq_len=self.max_seq_len,
            max_tokens_per_gpu=self.max_tokens_per_gpu,
        )
        
        # Move to device
        device = torch.cuda.current_device()
        for pack in packs:
            pack["tokens"] = pack["tokens"].to(device)
            pack["loss_masks"] = pack["loss_masks"].to(device)
            pack["cu_seqlens"] = pack["cu_seqlens"].to(device)
            pack["rewards"] = pack["rewards"].to(device)
            # raw_rewards stays on CPU
        
        return packs

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

        if self.ref_model is not None:
            self.compute_log_prob("ref", padded_batches, store_prefix="ref_")

        self.compute_log_prob("actor", padded_batches)

        # Compute advantages
        for batch in padded_batches:
            if self.args.advantage_estimator in ["grpo", "gspo"]:
                # For packed sequences, expand rewards per sequence
                cu_seqlens = batch["cu_seqlens"]
                num_sequences = len(cu_seqlens) - 1
                rewards_expanded = []
                for i in range(num_sequences):
                    start = cu_seqlens[i].item()
                    end = cu_seqlens[i + 1].item()
                    seq_len = end - start
                    if seq_len > 1:  # Only sequences with >1 token contribute to log_probs
                        rewards_expanded.extend([batch["rewards"][i]] * (seq_len - 1))
                batch["advantages"] = batch["returns"] = torch.tensor(
                    rewards_expanded, device=batch["rewards"].device, dtype=batch["rewards"].dtype
                )
            else:
                raise NotImplementedError(f"Unsupported advantage_estimator {self.args.advantage_estimator}")

        # TODO: finish log rollout_data
        log_dict = {}
        for key in ["log_probs", "ref_log_probs", "advantages", "returns", "raw_reward"]:
            if key not in padded_batches[0]:
                continue
            val = torch.tensor([0.0], device=torch.cuda.current_device())
            for batch in padded_batches:
                if isinstance(batch[key], torch.Tensor):
                    # Adjust loss_masks for values that are N-1 in length (next-token prediction)
                    # log_probs, ref_log_probs, advantages, and returns are all N-1 sized
                    if key in ["log_probs", "ref_log_probs", "advantages", "returns"]:
                        adjusted_masks = adjust_loss_masks_for_packed(batch["loss_masks"], batch["cu_seqlens"])
                        val += per_sample_mean(batch[key], adjusted_masks).item()
                    else:
                        # raw_reward and other per-sequence values use original masks
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
            # Process packed sequences in micro-batches to save memory
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self._process_packed_sequences_micro_batch(
                    batch["tokens"],
                    batch["cu_seqlens"],
                    micro_batch_size=2  # Reduced for lower memory usage
                )
            
            # Handle packed sequences
            log_probs = gather_log_probs_packed(logits, batch["tokens"], batch["cu_seqlens"])
            
            # Adjust loss_masks to match log_probs dimensions (N-1 due to next-token prediction)
            adjusted_loss_masks = adjust_loss_masks_for_packed(batch["loss_masks"], batch["cu_seqlens"])

            if self.args.advantage_estimator == "gspo":
                raise NotImplementedError("implement GSPO")

            ppo_kl = batch["log_probs"] - log_probs
            pg_loss, pg_clipfrac = compute_policy_loss(
                ppo_kl, batch["advantages"], self.args.eps_clip, self.args.eps_clip_high
            )

            pg_loss = per_sample_mean(pg_loss, adjusted_loss_masks)
            pg_clipfrac = per_sample_mean(pg_clipfrac, adjusted_loss_masks)
            ppo_kl = per_sample_mean(ppo_kl.abs(), adjusted_loss_masks)

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
                kl_loss = per_sample_mean(kl, adjusted_loss_masks)

                loss = loss + self.args.kl_loss_coef * kl_loss

            reported = {
                "loss": pg_loss.detach(),
                "pg_loss": pg_loss.detach(),
                "pg_clipfrac": pg_clipfrac.detach(),
                "ppo_kl": ppo_kl.detach(),
            }
            if self.args.use_kl_loss:
                reported["kl_loss"] = kl_loss.detach()

            # Scale loss for gradient accumulation
            loss = loss / grad_accum
            loss.backward()

            # Accumulate reported metrics (store tensors for later mean)
            for k, v in reported.items():
                reported_accum.setdefault(k, []).append(v)

            if (mbs_id + 1) % grad_accum == 0:
                # TODO: check if the grad norm is global grad norm.
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                # Aggregate logs
                aggregated = {k: torch.stack(v).mean().item() for k, v in reported_accum.items()}
                # TODO: change this, this is slow.
                reduced_aggregated = [None] * world_size
                dist.all_gather_object(reduced_aggregated, aggregated)
                # Mean across dp ranks
                aggregated = {}
                for k in reported_accum.keys():
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
                    print(f"step {self.global_step}: {log_dict}")
                    if self.args.use_wandb:
                        log_dict["train/step"] = self.global_step
                        wandb.log(log_dict)
                self.global_step += 1

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


def gather_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # log_probs: [B, T-1, V]; input_ids: [B, T]
    pred_logits = logits[:, :-1]
    log_probs_all = torch.log_softmax(pred_logits, dim=-1)
    tgt = input_ids[:, 1:].contiguous()
    log_probs = log_probs_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return log_probs


def adjust_loss_masks_for_packed(loss_masks: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Adjust loss masks to match log_probs dimensions (N-1 for next-token prediction)."""
    # Create mask to exclude first token of each sequence
    mask = torch.ones(len(loss_masks), dtype=torch.bool, device=loss_masks.device)
    mask[cu_seqlens[:-1]] = False  # Exclude first token of each sequence
    
    # Return loss_masks without first tokens (matching log_probs dimensions)
    return loss_masks[mask]


def gather_log_probs_packed(logits: torch.Tensor, input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Gather log probabilities for packed sequences."""
    # Create mask to exclude first token of each sequence
    mask = torch.ones(len(input_ids), dtype=torch.bool, device=input_ids.device)
    mask[cu_seqlens[:-1]] = False  # Exclude first token of each sequence
    
    # Shift for next-token prediction: logits[:-1] predicts input_ids[1:]
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    targets = input_ids[1:]
    
    # Gather log probs for targets
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    
    # Apply mask to exclude first tokens
    return gathered[mask[1:]]


def per_sample_mean(x, loss_mask):
    # TODO: impl per token loss
    # For packed sequences, x is already flattened
    return (x * loss_mask).sum() / loss_mask.sum().clamp_min(1)
