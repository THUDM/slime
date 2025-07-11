# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py
from typing import Optional, List

import torch
import torch.distributed as dist

from slime.utils.distributed_utils import distributed_masked_whiten


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
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac, approx_kl


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


def get_reinforce_plus_plus_advantages(
    rewards: torch.Tensor,
    kl: List[torch.Tensor],
    loss_masks: Optional[List[torch.Tensor]],
    response_lengths: List[int],
    kl_coef: float,
    gamma: float,
) -> (List[torch.Tensor], List[torch.Tensor]):
    """
    Calculates discounted returns and whitened advantages for REINFORCE++ (https://arxiv.org/pdf/2501.03262).

    Args:
        rewards (Tensor): A tensor of scalar rewards for each sequence. Shape: (batch_size,).
        kl (List[Tensor]): A list of per-token KL divergence tensors.
        loss_masks (Optional[List[Tensor]]): Loss masks for whitening and identifying the last token.
        response_lengths (List[int]): Sequence lengths for splitting.
        kl_coef (float): Coefficient for the KL penalty.
        gamma (float): The discount factor.

    Returns:
        A tuple of (advantages, returns):
        - advantages (List[Tensor]): The final, whitened advantages.
        - returns (List[Tensor]): The original, unwhitened discounted returns.
    """
    if loss_masks is None:
        # Assume the entire response is used for loss calculation.
        loss_masks = [torch.ones(length, device=kl[0].device, dtype=torch.long) for length in response_lengths]

    # token-level rewards (final_task_reward + per_token_kl_penalty).
    token_level_rewards = []
    for i in range(len(rewards)):
        r = -kl_coef * kl[i]
        assert loss_masks[i].sum() > 0, f"Sequence at index {i} is fully masked..."

        # Find the index of the last valid token to add the final task reward.
        last_response_idx = loss_masks[i].nonzero(as_tuple=True)[0][-1]
        r[last_response_idx] += rewards[i]
        token_level_rewards.append(r)

    # Calculate discounted returns per sequence
    returns = []
    for r_i in token_level_rewards:
        seq_len = len(r_i)
        returns_i = torch.zeros_like(r_i)
        running_return = 0.0
        for t in reversed(range(seq_len)):
            # G_t = r_t + gamma * G_{t+1}
            running_return = r_i[t] + gamma * running_return
            returns_i[t] = running_return
        returns.append(returns_i)

    all_returns = torch.cat(returns)
    all_masks = torch.cat(loss_masks)

    whitened_advs_flat = distributed_masked_whiten(all_returns, all_masks, shift_mean=True)
    advantages = list(torch.split(whitened_advs_flat, response_lengths))

    return advantages, returns

def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: List[torch.Tensor],
    loss_masks: Optional[List[torch.Tensor]],
    response_lengths: List[int],
    kl_coef: float,
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
        torch.ones_like(kl_tensor) * reward_val - kl_coef * kl_tensor
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
