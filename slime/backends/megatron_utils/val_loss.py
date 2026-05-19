"""Validation loss computation for SFT training.

Periodically computes validation NLL loss during SFT training with full
Data Parallel coordination. Reuses the existing training infrastructure:

- slime/utils/data.read_file() — load jsonl/parquet
- slime/utils/mask_utils.MultiTurnLossMaskGenerator — tokenize + loss mask
- slime/backends/megatron_utils/data.get_data_iterator() — dynamic batching
- slime/backends/megatron_utils/model.forward_only() — pipeline-parallel forward
- slime/backends/megatron_utils/loss.get_log_probs_and_entropy() — log_probs
- slime/backends/megatron_utils/cp_utils.get_sum_of_sample_mean() — CP-correct reduction
"""

import logging
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP

from slime.utils import logging_utils
from slime.utils.data import read_file
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


class ValDataLoader:
    """Loads and tokenizes validation data, sharded across DP ranks.

    Reuses the same tokenization pipeline as sft_rollout.py
    (MultiTurnLossMaskGenerator). Each DP rank keeps an interleaved shard
    of the full dataset. Data is tokenized once at init and cached in memory.
    """

    def __init__(self, args: Namespace, dp_rank: int, dp_size: int):
        self.args = args
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self._index = 0
        self._samples = self._load_and_tokenize()
        logger.info(
            f"ValDataLoader: rank {dp_rank}/{dp_size}, "
            f"{len(self._samples)} samples from {args.val_data}"
        )

    def _load_and_tokenize(self) -> list[dict]:
        """Load val file, shard by DP rank, tokenize via MultiTurnLossMaskGenerator."""
        tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True)
        mask_generator = MultiTurnLossMaskGenerator(
            tokenizer, tokenizer_type=self.args.loss_mask_type
        )
        val_input_key = getattr(self.args, "val_input_key", "messages")
        val_tool_key = getattr(self.args, "val_tool_key", None)

        all_records = list(read_file(self.args.val_data))
        if len(all_records) < self.dp_size:
            # Replicate small datasets so every rank has data (avoids skip).
            logger.info(
                f"Val data ({len(all_records)} samples) < dp_size ({self.dp_size}), "
                f"replicating to all ranks."
            )
            my_records = all_records
        else:
            my_records = all_records[self.dp_rank :: self.dp_size]

        samples = []
        for record in my_records:
            messages = record[val_input_key]
            tools = record.get(val_tool_key) if val_tool_key else None
            try:
                token_ids, loss_mask = mask_generator.get_loss_mask(messages, tools=tools)
            except Exception as e:
                logger.debug(f"Skipping val sample: {e}")
                continue

            if len(token_ids) != len(loss_mask):
                continue

            response_length = mask_generator.get_response_lengths([loss_mask])[0]
            if response_length == 0:
                continue

            samples.append({
                "token_ids": token_ids,
                "loss_mask": loss_mask[-response_length:],
                "response_length": response_length,
                "total_length": len(token_ids),
            })

        return samples

    def get_batch(self, batch_size: int) -> dict:
        """Build a RolloutBatch-compatible dict for get_data_iterator().

        Sets `dynamic_global_batch_size` so that get_data_iterator treats
        the entire val batch as one "rollout step" and applies dynamic
        batching (max_tokens_per_gpu) to split into microbatches.

        Returns dict with keys: tokens, loss_masks, response_lengths,
        total_lengths, dynamic_global_batch_size.
        """
        if not self._samples:
            return {}

        device = torch.cuda.current_device()
        tokens_list = []
        loss_masks_list = []
        response_lengths = []
        total_lengths = []

        for _ in range(batch_size):
            if self._index >= len(self._samples):
                self._index = 0

            sample = self._samples[self._index]
            self._index += 1

            tokens_list.append(
                torch.tensor(sample["token_ids"], dtype=torch.long, device=device)
            )
            loss_masks_list.append(
                torch.tensor(sample["loss_mask"], dtype=torch.int, device=device)
            )
            response_lengths.append(sample["response_length"])
            total_lengths.append(sample["total_length"])

        # Tell get_data_iterator the effective global batch size so it computes
        # num_local_gbs = batch_size, num_steps_per_rollout = 1, and applies
        # dynamic batching (max_tokens_per_gpu) to split into microbatches.
        return {
            "tokens": tokens_list,
            "loss_masks": loss_masks_list,
            "response_lengths": response_lengths,
            "total_lengths": total_lengths,
            "dynamic_global_batch_size": batch_size * self.dp_size,
        }

    @property
    def has_data(self) -> bool:
        return len(self._samples) > 0


def compute_val_loss(
    args: Namespace,
    model: Sequence[DDP],
    val_data_loader: ValDataLoader,
    rollout_id: int,
) -> None:
    """Compute validation loss using the same forward pipeline as training.

    Reuses get_data_iterator (dynamic batch / CP / PP support) and
    forward_only (pipeline-parallel forward pass) from the training path.
    Uses get_sum_of_sample_mean for CP-correct reduction.
    Loss is aggregated across DP ranks with token-weighted mean.

    All ranks MUST call this function together (even if their shard is empty)
    to avoid collective deadlocks.

    Args:
        args: Runtime arguments (same as training).
        model: DDP-wrapped model chunks.
        val_data_loader: Initialized ValDataLoader for this DP rank.
        rollout_id: Current rollout step (for logging x-axis).
    """
    from .cp_utils import get_sum_of_sample_mean
    from .data import get_data_iterator
    from .loss import get_log_probs_and_entropy
    from .model import forward_only

    # --- Synchronize: ensure all ranks agree on whether to proceed ---
    # Prevents deadlock if some ranks have empty val data.
    has_data = torch.tensor(
        [1 if val_data_loader.has_data else 0],
        device=torch.cuda.current_device(),
        dtype=torch.int,
    )
    dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
    if has_data.item() == 0:
        logger.warning("Some DP ranks have no val data, skipping val loss computation")
        return

    val_batch = val_data_loader.get_batch(args.val_batch_size)

    # get_data_iterator handles dynamic batching (max_tokens_per_gpu),
    # context parallelism, VPP, etc. — same path as training data.
    data_iterator, num_microbatches = get_data_iterator(args, model, val_batch)

    # Temporarily enable entropy in forward_only if log_sft_entropy is set.
    # forward_only uses args.use_rollout_entropy to decide whether to compute
    # entropy. We use a separate val-specific flag to avoid mutating the RL flag.
    log_entropy = getattr(args, "log_sft_entropy", False)
    orig_flag = args.use_rollout_entropy
    if log_entropy:
        args.use_rollout_entropy = True

    # forward_only runs pipeline-parallel forward, collects log_probs
    # (and entropy if enabled) on the last PP stage.
    # Model is switched to eval mode internally.
    result = forward_only(
        get_log_probs_and_entropy,
        args,
        model,
        data_iterator,
        num_microbatches,
    )

    # Restore immediately — don't leak into RL path
    args.use_rollout_entropy = orig_flag

    # --- Compute NLL on last PP stage using CP-correct reduction ---
    local_nll = torch.zeros(1, device=torch.cuda.current_device())
    local_tokens = torch.zeros(1, device=torch.cuda.current_device())
    local_entropy = torch.zeros(1, device=torch.cuda.current_device())

    if mpu.is_pipeline_last_stage() and "log_probs" in result:
        log_probs_list = result["log_probs"]
        response_lengths = val_batch["response_lengths"]
        total_lengths = val_batch["total_lengths"]
        loss_masks = val_batch["loss_masks"]

        # Use get_sum_of_sample_mean for CP-correct per-token reduction
        # (handles zigzag CP slicing, allgather-CP, etc.)
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            loss_masks,
            calculate_per_token_loss=True,
            qkv_format=args.qkv_format,
            max_seq_lens=val_batch.get("max_seq_lens", None),
        )
        log_probs_cat = torch.cat(log_probs_list, dim=0)
        # sum_of_sample_mean with per_token_loss=True sums all masked log_probs
        nll = -sum_of_sample_mean(log_probs_cat)

        num_tokens = sum(
            torch.clamp_min(m.sum(), 0) for m in loss_masks
        )

        local_nll.fill_(nll.item())
        # Divide by cp_size: each CP rank holds the full loss_mask but only
        # computes a partial NLL. When all_reduced across the DP+CP group,
        # NLL partials sum correctly, but tokens would be overcounted by cp_size.
        cp_size = mpu.get_context_parallel_world_size()
        local_tokens.fill_(num_tokens.item() / cp_size)

        # Compute entropy if available (same CP-correct reduction)
        if log_entropy and "entropy" in result:
            entropy_list = result["entropy"]
            entropy_cat = torch.cat(entropy_list, dim=0)
            entropy_sum = sum_of_sample_mean(entropy_cat)
            local_entropy.fill_(entropy_sum.item())

    # --- Token-weighted aggregation across DP ranks ---
    # All-reduce sum: total_nll and total_tokens across all DP ranks.
    # This gives correct token-weighted mean regardless of per-rank token count.
    dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
    dist.all_reduce(local_nll, op=dist.ReduceOp.SUM, group=dp_group)
    dist.all_reduce(local_tokens, op=dist.ReduceOp.SUM, group=dp_group)
    if log_entropy:
        dist.all_reduce(local_entropy, op=dist.ReduceOp.SUM, group=dp_group)

    # --- Log on primary rank ---
    if (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.is_pipeline_last_stage()
    ):
        total_nll = local_nll.item()
        total_tokens = local_tokens.item()

        if total_tokens > 0:
            val_loss = total_nll / total_tokens
        else:
            val_loss = 0.0

        step = compute_rollout_step(args, rollout_id)
        log_dict = {
            "val/loss": val_loss,
            "val/num_tokens": total_tokens,
            "val/step": step,
        }

        if log_entropy and total_tokens > 0:
            log_dict["val/entropy"] = local_entropy.item() / total_tokens

        logging_utils.log(args, log_dict, step_key="val/step")
        logger.info(f"val {rollout_id}: {log_dict}")
