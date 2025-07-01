# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py
from typing import Optional, List

import torch
import torch.distributed as dist

from slime.utils.distributed_utils import distributed_masked_whiten


# def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis=None, epsilon: float = 1e-8) -> torch.Tensor:
#     """
#     Calculates the mean of tensor elements specified by a mask.

#     Args:
#         values (Tensor): The source tensor containing values.
#         mask (Tensor): A mask tensor, typically of booleans or floats,
#                        that specifies which elements to include in the mean.
#         axis (int or tuple of int, optional): The dimension(s) to reduce. 
#                                               Defaults to reducing all dimensions.
#         epsilon (float): A small value to avoid division by zero.

#     Returns:
#         Tensor: A tensor containing the masked mean.
#     """
#     return (values * mask).sum(axis=axis) / (mask.sum(axis=axis) + epsilon)


# def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
#     """
#     Compute variance of tensor with masked values.

#     Args:
#         values (Tensor): The tensor of values for which to compute variance.
#         mask (Tensor): A mask that selects the elements to consider.
#         unbiased (bool): If True, applies Bessel's correction to compute the
#                          unbiased sample variance. Defaults to True.

#     Returns:
#         Tensor: A tensor containing the masked variance.
    
#     Raises:
#         ValueError: If the mask is empty (`mask.sum() == 0`), or if `unbiased`
#                     is True and the mask contains fewer than two elements.
#     """
#     mean = masked_mean(values, mask)
#     centered_values = values - mean
    
#     mask_sum = mask.sum()
#     if mask_sum == 0:
#         raise ValueError("Cannot compute variance over an empty mask (mask sum is zero).")

#     variance = masked_mean(centered_values**2, mask)
#     if unbiased:
#         # Apply Bessel's correction for unbiased variance
#         if mask_sum < 2:
#             raise ValueError(
#                 f"Cannot compute unbiased variance with mask sum less than 2. Got mask_sum={mask_sum.item()}"
#             )
#         bessel_correction = mask_sum / (mask_sum - 1)
#         variance = variance * bessel_correction

#     return variance


# def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True, epsilon: float = 1e-8) -> torch.Tensor:
#     """
#     Normalizes a tensor using its masked mean and standard deviation (whitening).
    
#     Args:
#         values (Tensor): The input tensor to be whitened.
#         mask (Tensor): Boolean tensor of same shape, selects elements for stats.
#         shift_mean (bool): If True (default), output is zero-mean;
#                            if False, the original mean is re-added after scaling.

#     Returns:
#         Tensor: The whitened tensor, having the same shape as `values`.
#     """
#     mean, var = masked_mean(values, mask), masked_var(values, mask)
#     whitened = (values - mean) * torch.rsqrt(var + epsilon)
#     if not shift_mean:
#         whitened += mean
#     return whitened


@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "kl":
        return log_ratio
    elif kl_loss_type == "low_var_kl":
        # The non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio
        return torch.clamp(log_ratio, min=-10, max=10)
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")


@torch.compile(dynamic=True)
def compute_policy_loss(
    log_probs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: Optional[float] = None,
):
    approx_kl = old_logprobs - log_probs
    ratio = (-approx_kl).exp()
    pg_losses = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    pg_loss = torch.max(pg_losses, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses).float()

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses3 = -eps_clip_c * advantages
        pg_loss = torch.min(pg_loss, pg_losses3)
    return pg_loss, clipfrac, approx_kl


def compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, process_group: Optional[dist.ProcessGroup]):
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    # convert to [seq_len, batch_size, vocab_size] as expected by fused_vocab_parallel_cross_entropy
    logits = logits.unsqueeze(1)
    tokens = tokens.unsqueeze(1)
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)


# from https://github.com/volcengine/verl/blob/0bdf7f469854815177e73dcfe9e420836c952e6e/verl/utils/megatron/tensor_parallel.py#L99
class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:

        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits, None


def compute_entropy_from_logits(logits: torch.Tensor, process_group) -> torch.Tensor:
    return _VocabParallelEntropy.apply(logits, process_group)


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns


# def get_reinforce_plus_plus_baseline_advantages(
#     rewards: torch.Tensor,
#     kl: list[torch.Tensor],
#     loss_masks: list[torch.Tensor],
#     response_lengths: list[int],
# ) -> list[torch.Tensor]:
#     """
#     Calculates the final advantages for the REINFORCE++-baseline algorithm.

#     This process involves two main steps:
#     1. Broadcasting the scalar (reward - group_baseline) to each token.
#     2. Applying a global whitening (normalization) across all advantages in the batch.

#     Args:
#         rewards (torch.Tensor): A tensor of scalar rewards, where the group-wise
#                                 baseline has already been subtracted.
#         kl (list[torch.Tensor]): A list of per-token KL divergence tensors. Used to
#                                  get the shape for broadcasting.
#         loss_masks (list[torch.Tensor]): A list of per-token loss masks, required for
#                                          whitening.
#         response_lengths (list[int]): A list of sequence lengths, required for
#                                       splitting the whitened tensor back.

#     Returns:
#         list[torch.Tensor]: A list of tensors containing the final, whitened advantages.
#     """
#     # Broadcast to get unwhitened advantages
#     unwhitened_advantages = []
#     for i in range(len(rewards)):
#         # reward here is already R - baseline
#         unwhitened_advantages.append(torch.ones_like(kl[i]) * rewards[i])
#     print(f"{rewards.size()=} | {rewards.device=}")
#     # Concatenate tensors for a global operation
#     if loss_masks is None:
#         loss_masks = [
#             torch.ones_like(adv) for adv in unwhitened_advantages
#         ]

#     all_advs = torch.cat(unwhitened_advantages)
#     all_masks = torch.cat(loss_masks)
    
#     whitened_advs_flat = masked_whiten(all_advs, all_masks, shift_mean=True)
#     advantages = list(torch.split(whitened_advs_flat, response_lengths))
    
#     return advantages


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: List[torch.Tensor],
    loss_masks: Optional[List[torch.Tensor]],
    response_lengths: List[int],
) -> List[torch.Tensor]:
    """
    Calculates the final advantages for the REINFORCE++-baseline algorithm.

    This process involves two main steps:
    1. Broadcasting the scalar (reward - group_baseline) to each token.
    2. Applying a *distributed* global whitening (normalization) across all 
       advantages in the data-parallel group.

    Args:
        rewards (Tensor): A tensor of scalar rewards, where the group-wise
                                baseline has already been subtracted.
        kl (list[Tensor]): A list of per-token KL divergence tensors. Used to
                                 get the shape for broadcasting.
        loss_masks (list[Tensor] | None): A list of per-token loss masks. If None,
                                          it's assumed all tokens are active.
        response_lengths (list[int]): A list of sequence lengths, required for
                                      splitting the whitened tensor back.

    Returns:
        list[Tensor]: A list of tensors containing the final, whitened advantages.
    """
    # Broadcast to get unwhitened advantages
    unwhitened_advantages = [
        torch.ones_like(kl_tensor) * reward_val
        for kl_tensor, reward_val in zip(kl, rewards)
    ]
    # Concatenate tensors for a global operation
    if loss_masks is None:
        loss_masks = [
            torch.ones_like(adv) for adv in unwhitened_advantages
        ]

    all_advs = torch.cat(unwhitened_advantages)
    all_masks = torch.cat(loss_masks)
    
    whitened_advs_flat = distributed_masked_whiten(all_advs, all_masks, shift_mean=True)
    
    advantages = list(torch.split(whitened_advs_flat, response_lengths))
    
    return advantages
