"""CISPO: Clipped Importance Sampling Policy Optimization.

A REINFORCE-style RL loss that clips importance sampling weights (upper bound
only) as stop-gradient multipliers. Unlike PPO/GRPO/DAPO which zero out
gradients outside the trust region, CISPO always allows gradients but scales
them by clamped IS ratios.

Reference:
    MiniMax-M1 (arxiv:2506.13585)
    Swift implementation: https://swift.readthedocs.io/zh-cn/latest/Instruction/GRPO/AdvancedResearch/CISPO.html

Usage:
    --loss-type custom_loss
    --custom-loss-function-path examples.cispo.cispo_loss.cispo_loss_function
    --eps-clip-high 5.0
"""

from argparse import Namespace
from collections.abc import Callable

import torch
from megatron.core import mpu

from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
from slime.utils.ppo_utils import compute_approx_kl
from slime.utils.types import RolloutBatch


def cispo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute CISPO loss and metrics.

    Loss formula:
        L = -E[ detach(min(r_t, ε_high)) * Â_t * log π_θ(a_t|s_t) ]

    where r_t = π_θ / π_old is the IS ratio, detach() is stop-gradient,
    and Â_t is the group-relative advantage (same as GRPO).

    Args:
        args: Configuration. Key fields: eps_clip_high (raw upper bound,
            e.g. 5.0), entropy_coef, use_kl_loss, kl_loss_coef.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths",
            "loss_masks", and optionally "ref_log_probs".
        logits: Policy logits with shape [1, T, V].
        sum_of_sample_mean: Reduction function (handles loss_masks internally).

    Returns:
        (loss, metrics) where loss is a scalar tensor and metrics is a dict
        of detached scalars.
    """
    # Early return for non-last pipeline stages — must come before torch.cat
    # to avoid operations on empty tensors from intermediate PP stages.
    if not mpu.is_pipeline_last_stage():
        dummy = 0 * logits.sum()
        return dummy, {"loss": dummy.clone().detach()}

    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    # 1. Compute current policy log probs and entropy
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0)
    old_log_probs = torch.cat(old_log_probs, dim=0)

    # 2. IS ratio: r = π_θ / π_old
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # 3. Clamp IS ratio — UPPER BOUND ONLY (key CISPO difference from PPO/GRPO)
    #    eps_clip_high is the raw upper bound (e.g. 5.0), NOT the 1+ε offset
    clamped_ratio = torch.clamp(ratio, max=args.eps_clip_high).detach()

    # 4. Clip fraction metric (how often ratio exceeds ε_high)
    clipfrac = (ratio > args.eps_clip_high).float()

    # 5. CISPO loss: -detach(min(r, ε_high)) * A * log π_θ
    #    log_probs are log π_θ ∈ (-∞, 0], so:
    #    - positive advantage → loss > 0 → minimization increases log π_θ ✓
    #    - negative advantage → loss < 0 → minimization decreases log π_θ ✓
    pg_loss = -(clamped_ratio) * advantages * log_probs

    # 6. KL divergence for monitoring (not part of loss unless use_kl_loss)
    ppo_kl = (old_log_probs - log_probs).detach()

    # 7. Reduce via sum_of_sample_mean (handles loss_masks automatically)
    pg_loss = sum_of_sample_mean(pg_loss)
    pg_clipfrac = sum_of_sample_mean(clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # 8. Entropy bonus
    entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
    entropy_loss = sum_of_sample_mean(entropy)
    loss = pg_loss - args.entropy_coef * entropy_loss

    # 9. Optional KL loss term (against reference model)
    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    # 10. Ensure gradient flows through logits even with empty log_probs
    #     (safety net; early return above should handle PP non-last stages)
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
        "is_ratio_mean": sum_of_sample_mean(ratio.detach()),
    }
    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    return loss, reported_loss
