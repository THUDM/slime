# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py

from argparse import Namespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        kl_loss_type: Type of KL estimator (k1, k2, k3, low_var_kl).
        importance_ratio: Optional IS ratio (π_θ/π_old) for unbiased KL estimation.
    """
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        # The non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    # Apply IS ratio for unbiased KL estimation (DeepSeek-V3.2)
    if importance_ratio is not None:
        kl = importance_ratio * kl

    # Clamp only for low_var_kl for numerical stability
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl


def compute_opsm_mask(
    args: Namespace,
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Off-Policy Sequence Masking (OPSM) mask.

    Args:
        args: Configuration containing `opsm_delta` threshold.
        full_log_probs: Current policy log-probs per sample.
        full_old_log_probs: Old policy log-probs per sample.
        advantages: Advantage values per sample.
        loss_masks: Loss masks per sample.

    Returns:
        Tuple of `(opsm_mask, opsm_clipfrac)` where `opsm_mask` is a
        concatenated tensor of per-token masks and
        `opsm_clipfrac` is the count of masked sequences.
    """
    opsm_mask_list = []
    device = advantages[0].device
    opsm_clipfrac = torch.tensor(0.0, device=device)

    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(
        full_log_probs, full_old_log_probs, advantages, loss_masks, strict=False
    ):
        # Calculate sequence-level KL
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)

        # Create mask: 0 if (advantage < 0 and seq_kl > delta), else 1
        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

        opsm_mask_list.append(1 - mask)

    opsm_mask = torch.cat(opsm_mask_list, dim=0)
    return opsm_mask, opsm_clipfrac


def compute_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Compute GSPO-style per-sequence KL divergence.

    Args:
        full_log_probs: Current policy log-probs per sample (full or CP-local).
        full_old_log_probs: Old policy log-probs per sample (full or CP-local).
        local_log_probs: Local (CP-local) log-probs for expansion shape reference.
        loss_masks: Loss masks per sample.

    Returns:
        Concatenated tensor of per-token KL values where each token in a
        sequence has the same KL value (the sequence-level KL).
    """
    # Compute sequence-level KL and expand to per-token
    ppo_kl = [
        ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
    ]
    ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
    ppo_kl = torch.cat(ppo_kl, dim=0)

    return ppo_kl


@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = (-ppo_kl).exp()
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

    return pg_losses, clipfrac


def compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, process_group: dist.ProcessGroup | None):
    # TODO: when megatron is not installed, fall back to naive implementation
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


class _VocabParallelReverseKL(torch.autograd.Function):
    """Compute D_KL(π_student ∥ π_teacher) over a vocabulary-parallel partition.

    Both *student_logits* and *teacher_logits* are partial tensors along the
    vocab dimension (each TP rank holds V/tp_size entries).  The function
    performs the necessary all-reduces to compute the exact reverse KL
    divergence in a numerically stable manner:

        D_KL(π_s ∥ π_t) = Σ_y π_s(y) [log π_s(y) - log π_t(y)]

    Forward returns a tensor of shape [R] (one KL value per response token).
    Backward propagates gradients w.r.t. *student_logits* only; teacher logits
    are treated as constants (detached).

    Gradient derivation:
        KL = Σ_y π_s(y) [log π_s(y) - log π_t(y)]
        ∂KL/∂z_j = π_s(j) [log π_s(j) - log π_t(j) + 1 - Σ_k π_s(k)(log π_s(k) - log π_t(k) + 1)]
                  = π_s(j) [log π_s(j) - log π_t(j) - KL]     (since Σ_k π_s(k) = 1)
    where z_j are the student logits and log π_s is log_softmax(z).
    """

    @staticmethod
    def forward(
        ctx,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        process_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        # --- student softmax (numerically stable, TP-aware) ---
        s_max = student_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(s_max, op=dist.ReduceOp.MAX, group=process_group)
        s_shifted = student_logits - s_max
        s_exp = s_shifted.exp()
        s_sum_exp = s_exp.sum(dim=-1, keepdim=True)
        dist.all_reduce(s_sum_exp, op=dist.ReduceOp.SUM, group=process_group)
        s_softmax = s_exp / s_sum_exp  # π_s(y)  [R, V_local]
        s_log_sum_exp = s_sum_exp.log()  # [R, 1]

        # --- teacher log-softmax (numerically stable, TP-aware) ---
        t_max = teacher_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX, group=process_group)
        t_shifted = teacher_logits - t_max
        t_exp = t_shifted.exp()
        t_sum_exp = t_exp.sum(dim=-1, keepdim=True)
        dist.all_reduce(t_sum_exp, op=dist.ReduceOp.SUM, group=process_group)
        t_log_sum_exp = t_sum_exp.log()  # [R, 1]

        # --- KL = Σ_y π_s(y) [log π_s(y) - log π_t(y)] ---
        # log π_s(y) = s_shifted - s_log_sum_exp  (local slice)
        # log π_t(y) = t_shifted - t_log_sum_exp  (local slice)
        local_s_log_prob = s_shifted - s_log_sum_exp
        local_t_log_prob = t_shifted - t_log_sum_exp

        local_kl_sum = (s_softmax * (local_s_log_prob - local_t_log_prob)).sum(dim=-1, keepdim=True)
        dist.all_reduce(local_kl_sum, op=dist.ReduceOp.SUM, group=process_group)
        kl = local_kl_sum.squeeze(dim=-1)  # [R]

        # Save for backward
        # We need: s_softmax, local_s_log_prob, local_t_log_prob, and kl (per-token)
        ctx.save_for_backward(s_softmax, local_s_log_prob.detach(), local_t_log_prob.detach(), kl.detach())
        ctx.process_group = process_group
        return kl

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        s_softmax, local_s_log_prob, local_t_log_prob, kl = ctx.saved_tensors
        process_group = ctx.process_group

        # Gradient: ∂KL/∂z_j = π_s(j) * [log π_s(j) - log π_t(j) - KL]
        # This is completely local per token — no all_reduce needed in backward.
        grad_local = s_softmax * (local_s_log_prob - local_t_log_prob - kl.unsqueeze(-1))

        grad_input = grad_output.unsqueeze(-1) * grad_local  # [R, V_local]
        return grad_input, None, None


def vocab_parallel_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    process_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Compute D_KL(π_student ∥ π_teacher) with TP-aware vocab parallelism.

    Both inputs are partial logits along the vocab dimension (each TP rank
    holds V/tp_size logits).  Returns per-token KL of shape [R].

    Teacher logits are detached (no gradient flows to the teacher).

    Args:
        student_logits: [R, V_local] student logits (with grad).
        teacher_logits: [R, V_local] teacher logits (detached).
        process_group: TP process group for all-reduce.

    Returns:
        Per-token KL divergence tensor of shape [R].
    """
    # Detach teacher logits — we never backprop through the teacher
    teacher_logits = teacher_logits.detach()

    tp_size = dist.get_world_size(group=process_group) if process_group is not None else 1
    if tp_size <= 1:
        # No TP — simple local computation
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_probs = student_log_probs.exp()
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        return kl

    # TP mode — use custom autograd function with correct backward
    return _VocabParallelReverseKL.apply(student_logits, teacher_logits, process_group)


def vocab_parallel_topk_reverse_kl(
    student_logits: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    vocab_size: int,
    process_group: dist.ProcessGroup,
    valid_topk_mask: torch.Tensor | None = None,
    is_log_probs: bool = False,
    teacher_log_sum_exp: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute approximate D_KL(π_student ∥ π_teacher) using top-k teacher logits plus tail correction.

    This is a memory-efficient alternative to full-vocab KL. The teacher provides only
    its top-k logits and indices (pre-computed during the teacher forward pass), while
    the student has full vocab logits.

    The KL is decomposed into:
        KL = KL_topk + KL_tail
    where:
        KL_topk = Σ_{y ∈ topk} π_s(y) [log π_s(y) - log π_t(y)]
        KL_tail ≈ π_s_tail * log(π_s_tail / π_t_tail)

    For TP: teacher_topk_indices are LOCAL indices within each TP shard. We gather
    the student logits at those local positions directly (no cross-shard indexing needed).

    Args:
        student_logits: [R, V_local] student logits (with grad), vocab-sharded across TP.
        teacher_topk_logits: [R, k] teacher top-k logits (detached), fp32.
            When is_log_probs=False (Megatron mode): raw logits from teacher forward pass.
            When is_log_probs=True (SGLang mode): log probabilities (log softmax) from
            SGLang's input_top_logprobs. The function will skip the log_softmax step
            and use them directly as teacher log-probs.
        teacher_topk_indices: [R, k] teacher top-k LOCAL indices within each TP shard, int.
        vocab_size: Full (unsharded) vocabulary size V.
        process_group: TP process group for all-reduce.
        valid_topk_mask: Optional [R, k] boolean mask. True = valid entry, False = padding.
            When provided (e.g. for SGLang-sourced top-k with TP>1 where different shards
            have different numbers of real entries), padding entries are zeroed out so they
            don't contribute to the KL. When None, all k entries are assumed valid (Megatron
            mode where each shard independently computes its top-k).
        is_log_probs: If True, teacher_topk_logits contains log probabilities (already
            softmax-normalized) rather than raw logits. This is the case for SGLang mode
            where the server returns log-probs directly. When True, the function skips
            the log_softmax computation and uses the values as-is for teacher log-probs.
        teacher_log_sum_exp: Optional [R] tensor with the teacher's full-vocabulary
            log_sum_exp per token position (computed from complete logits during the
            Megatron teacher forward pass). When provided along with is_log_probs=False
            (Megatron mode), enables exact teacher tail mass computation:
            teacher_tail_mass = 1 - sum(exp(topk_logits - log_sum_exp)).
            This replaces the inaccurate uniform assumption (V - V_eff) / V that
            can over-estimate tail mass by orders of magnitude when k << V.

    Returns:
        Per-token KL divergence tensor of shape [R].
    """
    # Detach teacher inputs
    teacher_topk_logits = teacher_topk_logits.detach()
    teacher_topk_indices = teacher_topk_indices.detach()

    # torch.gather requires LongTensor (int64) indices.
    # Accept int32 from the data pipeline and cast defensively.
    if teacher_topk_indices.dtype != torch.int64:
        teacher_topk_indices = teacher_topk_indices.long()

    tp_size = dist.get_world_size(group=process_group) if process_group is not None else 1
    k = teacher_topk_logits.size(-1)

    # Compute validity mask from teacher_topk_logits if not provided.
    # Entries with -inf logits are padding (e.g., from SGLang TP sharding).
    if valid_topk_mask is None:
        # Auto-detect: any entry that is not -inf is valid.
        # This is backward-compatible with Megatron mode where all entries are valid.
        valid_topk_mask = ~torch.isinf(teacher_topk_logits)

    # Zero out padding entries in teacher_topk_logits to prevent NaN in exp()
    # This replaces -inf with a large negative value that won't affect the max but
    # will become 0 after exp. The valid_topk_mask handles the rest.
    teacher_topk_logits_safe = teacher_topk_logits.clone()
    teacher_topk_logits_safe[~valid_topk_mask] = -1e9  # large negative, not -inf

    # --- student softmax (numerically stable, TP-aware) ---
    s_max = student_logits.max(dim=-1, keepdim=True).values
    if tp_size > 1:
        dist.all_reduce(s_max, op=dist.ReduceOp.MAX, group=process_group)
    s_shifted = student_logits - s_max
    s_exp = s_shifted.exp()
    s_sum_exp = s_exp.sum(dim=-1, keepdim=True)
    if tp_size > 1:
        dist.all_reduce(s_sum_exp, op=dist.ReduceOp.SUM, group=process_group)
    s_softmax = s_exp / s_sum_exp  # π_s(y)  [R, V_local]
    s_log_sum_exp = s_sum_exp.log()  # [R, 1]

    # Gather student probs and log-probs at teacher's top-k positions
    # teacher_topk_indices are LOCAL to this TP shard
    student_topk_probs = s_softmax.gather(-1, teacher_topk_indices)  # [R, k]
    student_topk_shifted = s_shifted.gather(-1, teacher_topk_indices)  # [R, k]
    student_topk_log_probs = student_topk_shifted - s_log_sum_exp  # [R, k]

    # Zero out student contributions at padding positions
    student_topk_probs = student_topk_probs * valid_topk_mask.float()
    # student_topk_log_probs: we only use this in KL_topk where it's multiplied by
    # student_topk_probs (which is zero at padding). So no separate masking needed.

    # --- teacher distribution from top-k entries ---
    if is_log_probs:
        # SGLang mode: teacher_topk_logits already contains log probabilities.
        # No need to compute log_softmax — use them directly.
        # teacher_topk_log_probs_approx = teacher_topk_logits (with padding zeroed out)
        # teacher_topk_probs = exp(teacher_topk_logits) (only for tail mass computation)
        teacher_topk_log_probs_approx = teacher_topk_logits_safe * valid_topk_mask.float()
        teacher_topk_probs = teacher_topk_log_probs_approx.exp() * valid_topk_mask.float()
    else:
        # Megatron mode: teacher_topk_logits contains raw logits. Apply log_softmax
        # over the top-k entries to get teacher log-probs (TP-aware).
        t_max = teacher_topk_logits_safe.max(dim=-1, keepdim=True).values
        if tp_size > 1:
            # teacher_topk_logits are per-shard, so we need global max across shards
            # BUT: each shard's top-k is independent (local indices).
            # To compute the correct global log_sum_exp, we need:
            #   (a) the global max of ALL top-k logits across shards, and
            #   (b) the sum of exp(logits - global_max) across all shards.
            dist.all_reduce(t_max, op=dist.ReduceOp.MAX, group=process_group)
        t_shifted = teacher_topk_logits_safe - t_max
        t_exp = t_shifted.exp()
        # Zero out padding contributions in exp sum
        t_exp = t_exp * valid_topk_mask.float()
        t_sum_exp = t_exp.sum(dim=-1, keepdim=True)
        if tp_size > 1:
            # Sum of exp across all shards (each shard contributes its valid top-k values)
            dist.all_reduce(t_sum_exp, op=dist.ReduceOp.SUM, group=process_group)
        t_log_sum_exp = t_sum_exp.log()  # [R, 1]

        # Compute teacher log-probs from the safe (non-inf) logits
        teacher_topk_log_probs_approx = t_shifted - t_log_sum_exp  # [R, k]
        # Zero out padding entries
        teacher_topk_log_probs_approx = teacher_topk_log_probs_approx * valid_topk_mask.float()
        teacher_topk_probs = (t_shifted.exp() * valid_topk_mask.float()) / t_sum_exp  # [R, k]

    # --- tail mass (TP-aware) ---
    # Student tail mass: 1 - sum(π_s(y) for y in valid top-k of this shard)
    student_topk_mass = student_topk_probs.sum(dim=-1)  # [R]
    if tp_size > 1:
        # Sum the top-k mass across all TP shards to get the total mass in all shards' top-k
        dist.all_reduce(student_topk_mass, op=dist.ReduceOp.SUM, group=process_group)
    student_tail_mass = (1.0 - student_topk_mass).clamp(min=0.0)  # [R]

    # Teacher tail mass: 1 - sum(π_t(y) for y in top-k)
    # Computed from the actual teacher probability mass in the top-k partition,
    # NOT from the uniform assumption (V - V_eff) / V which is wildly inaccurate
    # when k << V (e.g., k=128, V=152064 → uniform tail ≈ 0.999 → catastrophic
    # rescaling of ~-7 nats).
    if is_log_probs:
        # SGLang mode: teacher_topk_probs already reflects the true probability mass
        # because log_probs came from a full softmax over the entire vocabulary.
        # Sum exp(log_prob) across all valid top-k entries to get the actual mass.
        teacher_topk_mass = teacher_topk_probs.sum(dim=-1)  # [R]
        if tp_size > 1:
            # Sum across TP shards to get total mass from all shards' top-k entries
            dist.all_reduce(teacher_topk_mass, op=dist.ReduceOp.SUM, group=process_group)
        teacher_tail_mass = (1.0 - teacher_topk_mass).clamp(min=0.0)  # [R]
    else:
        # Megatron mode: teacher_topk_logits are raw logits and teacher_topk_probs
        # are from softmax over top-k entries only (not the full vocabulary).
        # The sum is ~1 within the top-k partition, so we need an external
        # reference to compute the true tail mass.
        if teacher_log_sum_exp is not None:
            # Exact tail mass from the full-vocabulary log_sum_exp computed
            # during the teacher forward pass. This is the preferred method:
            # teacher_topk_mass = sum(exp(logits - log_sum_exp)) for valid entries
            # teacher_tail_mass = 1 - teacher_topk_mass
            # No TP all-reduce needed for teacher_log_sum_exp — it was already
            # reduced when computed in actor.py.
            # teacher_topk_logits_safe contains the safe (non-inf) logits
            # with padding replaced by -1e9. Use valid_topk_mask to zero out
            # padding contributions.
            topk_shifted = teacher_topk_logits_safe - teacher_log_sum_exp.unsqueeze(-1)  # [R, k]
            topk_probs_from_full = topk_shifted.exp() * valid_topk_mask.float()  # [R, k]
            teacher_topk_mass = topk_probs_from_full.sum(dim=-1)  # [R]
            if tp_size > 1:
                dist.all_reduce(teacher_topk_mass, op=dist.ReduceOp.SUM, group=process_group)
            teacher_tail_mass = (1.0 - teacher_topk_mass).clamp(min=0.0)  # [R]
        else:
            # Fallback: uniform tail assumption (V - V_eff) / V.
            # This is inaccurate when k << V (e.g., k=128, V=152K → tail ≈ 0.999)
            # and will over-estimate the KL by ~5-7 nats. Should only be used
            # when teacher_log_sum_exp is not available (legacy fallback).
            valid_count = valid_topk_mask.float().sum(dim=-1)  # [R]
            if tp_size > 1:
                dist.all_reduce(valid_count, op=dist.ReduceOp.SUM, group=process_group)
            V_eff = valid_count  # [R]
            teacher_tail_mass = torch.clamp((vocab_size - V_eff) / vocab_size, min=0.0)  # [R]

    # Scale teacher log-probs to account for tail mass.
    #
    # Megatron mode (is_log_probs=False):
    #   teacher_topk_log_probs_approx = log_softmax(topk_logits) — these are
    #   normalized only within the top-k partition (sum to ~1 within top-k).
    #   We need to rescale: log π_t(y) = log_softmax(topk) + log(1 - tail_mass)
    #   so that the top-k probabilities sum to (1 - tail_mass) over the full vocab.
    #
    # SGLang mode (is_log_probs=True):
    #   teacher_topk_log_probs_approx = log(π_t(y)) from SGLang's full-vocab softmax.
    #   These are already normalized over the full vocabulary, so NO rescaling needed.
    #   The top-k probabilities naturally sum to (1 - teacher_tail_mass) which we
    #   computed above as 1 - sum(exp(log_prob)).
    if is_log_probs:
        teacher_topk_log_probs = teacher_topk_log_probs_approx
    else:
        safe_tail = (teacher_tail_mass > 0) & (teacher_tail_mass < 1.0)
        teacher_topk_log_probs = teacher_topk_log_probs_approx.clone()
        if safe_tail.any():
            # log(1 - teacher_tail_mass) is per-token [R], need to broadcast to [R, 1]
            scale = torch.log((1.0 - teacher_tail_mass).clamp(min=1e-10)).unsqueeze(-1)  # [R, 1]
            teacher_topk_log_probs = torch.where(
                safe_tail.unsqueeze(-1),
                teacher_topk_log_probs_approx + scale,
                teacher_topk_log_probs_approx,
            )

    # --- KL computation ---
    # KL_topk = Σ_{y ∈ top-k (all shards)} π_s(y) [log π_s(y) - log π_t(y)]
    local_kl_topk = (student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)).sum(dim=-1)  # [R]
    if tp_size > 1:
        dist.all_reduce(local_kl_topk, op=dist.ReduceOp.SUM, group=process_group)

    # KL_tail ≈ π_s_tail * log(π_s_tail / π_t_tail)
    # π_s_tail = student_tail_mass per token
    # π_t_tail = teacher_tail_mass (estimated above)
    kl_tail = torch.zeros_like(student_tail_mass)
    tail_mask = (student_tail_mass > 1e-10) & (teacher_tail_mass > 1e-10)
    kl_tail[tail_mask] = student_tail_mass[tail_mask] * (
        torch.log(student_tail_mass[tail_mask]) - torch.log(
            teacher_tail_mass[tail_mask]
        )
    )
    # If teacher_tail_mass ≈ 0 but student_tail_mass > 0, we have an unbounded KL.
    # This shouldn't happen if k is large enough. We treat it as 0 for numerical safety.

    kl = local_kl_topk + kl_tail  # [R]

    return kl


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns


def get_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    kl_coef: float,
    gamma: float,
) -> list[torch.Tensor]:
    """
    Calculates discounted returns for REINFORCE++ (https://arxiv.org/pdf/2501.03262)

    Args:
        rewards (Tensor): A tensor of scalar rewards for each sequence.
        kl (List[Tensor]): List of per-token KL divergence tensors for sequence chunks.
        loss_masks (List[Tensor]): List of response-only loss masks for each full sequence.
        response_lengths (List[int]): The full length of each response sequence.
        total_lengths (List[int]): The full length of each sequence (prompt + response).
        kl_coef (float): Coefficient for the KL penalty.
        gamma (float): The discount factor.

    Returns:
        List[torch.Tensor]: A list of return (G_t) tensors for the
                            local sequence chunks owned by the current GPU rank.
    """
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()

    final_returns_chunks = []
    for i in range(len(rewards)):
        local_kl_chunk = kl[i]
        total_len, response_len = total_lengths[i], response_lengths[i]

        if cp_size > 1:
            # Step 1,2:Gather all chunks and token_offsets from all ranks and reconstruct the full response tensor by splitting and placing each part
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
        else:
            full_kl_response = local_kl_chunk

        # Step 3: Compute returns on full response kl tensor.
        full_mask = loss_masks[i]
        assert full_mask.sum().item() > 0, f"Sequence at index {i} is fully masked."
        masked_kl = full_kl_response * full_mask
        token_level_rewards = -kl_coef * masked_kl
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        token_level_rewards[last_idx] += rewards[i]

        returns_for_seq = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            # G_t = r_t + gamma * G_{t+1}
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return

        # Step 4: Pick up the results corresponding to our local chunk's parts.
        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
        else:
            local_returns_chunk = returns_for_seq

        final_returns_chunks.append(local_returns_chunk)

    return final_returns_chunks


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl_coef: float,
) -> list[torch.Tensor]:
    """
    Calculates the unwhitened advantages for the REINFORCE++-baseline algorithm.
    Broadcasting the scalar (reward - group_baseline) to each token.

    Args:
        rewards (Tensor): A tensor of scalar rewards, where the group-wise
                                baseline has already been subtracted.
        kl (list[Tensor]): A list of per-token KL divergence tensors. Used to
                                 get the shape for broadcasting.
        loss_masks (list[Tensor]): A list of per-token loss masks.
        kl_coef (float): Coefficient for the KL penalty.

    Returns:
        list[Tensor]: A list of tensors containing the unwhitened advantages.
    """
    # Broadcast to get unwhitened advantages
    unwhitened_advantages = [
        torch.ones_like(kl_tensor) * reward_val - kl_coef * kl_tensor
        for kl_tensor, reward_val in zip(kl, rewards, strict=False)
    ]

    return unwhitened_advantages


def get_advantages_and_returns(
    total_len: int,
    response_len: int,
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Input:
    - values: Tensor of shape (response_size,)
    - rewards: Tensor of shape (response_size,)

    Output:
    - advantages: Tensor of shape (response_size,)
    - returns: Tensor of shape (response_size,)
    """
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

        full_rewards = all_gather_with_cp(rewards, total_len, response_len)
        full_values = all_gather_with_cp(values, total_len, response_len)
    else:
        full_rewards = rewards
        full_values = values

    lastgaelam = 0
    advantages_reversed = []

    for t in reversed(range(response_len)):
        nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
        delta = full_rewards[t] + gamma * nextvalues - full_values[t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    full_advantages = torch.tensor(advantages_reversed[::-1], dtype=full_values.dtype, device=full_values.device)
    full_returns = full_advantages + full_values

    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        advantages = slice_log_prob_with_cp(full_advantages, total_len, response_len)
        returns = slice_log_prob_with_cp(full_returns, total_len, response_len)
    else:
        advantages = full_advantages
        returns = full_returns

    return advantages.detach(), returns


def get_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    rewards_list,
    gamma,
    lambd,
    chunked: bool = True,
):
    """
    Batched GAE with CP support.
    Input:
        total_lengths:     list[int], each sample's total_len
        response_lengths:  list[int], each sample's response_len
        values_list:       list[Tensor], each shape = [resp_len_i]
        rewards_list:      list[Tensor], same shape
    Output:
        advantages_list:   list[Tensor], each shape = [resp_len_i]
        returns_list:      list[Tensor], same shape
    """

    from megatron.core import mpu

    with torch.no_grad():
        B = len(response_lengths)
        assert B == len(values_list)
        assert B == len(rewards_list)

        cp_size = mpu.get_context_parallel_world_size()
        device = values_list[0].device
        dtype = values_list[0].dtype

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_values_list = []
            full_rewards_list = []

            for total_len, resp_len, v, r in zip(
                total_lengths, response_lengths, values_list, rewards_list, strict=False
            ):
                full_v = all_gather_with_cp(v, total_len, resp_len)
                full_r = all_gather_with_cp(r, total_len, resp_len)
                full_values_list.append(full_v)
                full_rewards_list.append(full_r)

            # full_values_list[i].shape = [total_len_i]
        else:
            full_values_list = values_list
            full_rewards_list = rewards_list

        # pad to max_len for batched GAE
        max_len = max(response_lengths)

        full_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        full_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)

        for i in range(B):
            L = response_lengths[i]
            full_values[i, :L] = full_values_list[i][:L]
            full_rewards[i, :L] = full_rewards_list[i][:L]

        if not chunked:
            full_advantages, full_returns = vanilla_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )
        else:
            full_advantages, full_returns = chunked_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )

        advantages_list = []
        returns_list = []

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            for total_len, resp_len, adv_row, ret_row in zip(
                total_lengths,
                response_lengths,
                full_advantages,
                full_returns,
                strict=False,
            ):
                adv_full = adv_row  # shape = [resp_len_i padded to max_len]
                ret_full = ret_row

                adv_sliced = slice_log_prob_with_cp(adv_full[:resp_len], total_len, resp_len)
                ret_sliced = slice_log_prob_with_cp(ret_full[:resp_len], total_len, resp_len)

                advantages_list.append(adv_sliced)
                returns_list.append(ret_sliced)

        else:
            for i in range(B):
                L = response_lengths[i]
                advantages_list.append(full_advantages[i, :L])
                returns_list.append(full_returns[i, :L])

    return advantages_list, returns_list


def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
):
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)

    full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
    full_returns = full_advantages + values  # [B, max_len]
    return full_advantages, full_returns


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):
    """
    Compute Generalized Advantage Estimation (GAE) using a FlashLinearAttention-
    inspired algorithm: parallel prefix scan within chunks and recurrent state
    propagation across chunks.

    This reduces the sequential dependency length from O(T) to O(T / chunk_size),
    while keeping chunk computations fully parallelizable (O(C^2) per chunk).

    Args:
        rewards (Tensor): [B, T] reward sequence.
        values (Tensor):  [B, T] value predictions. The next-value of the final
                          step is assumed to be zero (standard PPO convention).
        gamma (float): discount factor.
        lam (float): GAE lambda.
        chunk_size (int): sequence chunk length for parallel scan.

    Returns:
        advantages (Tensor): [B, T] computed advantages.
        returns (Tensor):    [B, T] advantages + values.
    """

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    device = rewards.device
    dtype = rewards.dtype

    # -------------------------------------------------------------------------
    # Build δ_t = r_t + γ * V_{t+1} - V_t   with V_{T} = 0
    # -------------------------------------------------------------------------
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    # Reformulate backward GAE as a forward scan on the reversed sequence:
    #   S[i] = Δ[i] + w * S[i - 1],   w = γλ
    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])  # [B, T]

    # -------------------------------------------------------------------------
    # Pad to a multiple of chunk_size
    # -------------------------------------------------------------------------
    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size

    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    # -------------------------------------------------------------------------
    # Construct the intra-chunk parallel scan kernel M
    #
    # For a chunk Δ[0..C-1], we want:
    #   S_local[t] = sum_{k=0..t} w^(t-k) * Δ[k]
    #
    # This is implemented as:
    #   S_local = Δ @ M
    #
    # where:
    #   M[i, j] = w^(j - i)    if j >= i
    #             0            otherwise
    # -------------------------------------------------------------------------
    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row

    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0

    if w == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = w ** diff[mask].to(dtype)

    # pow_vec[t] = w^(t+1), used to inject the recurrent state s_prev
    if w == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Parallel compute local chunk results (assuming initial state = 0)
    # -------------------------------------------------------------------------
    deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    # Effective length of each chunk (the last chunk may be padded)
    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    # -------------------------------------------------------------------------
    # Recurrent propagation between chunks
    #
    # Each chunk contributes:
    #   S_global[t] = S_local[t] + w^(t+1) * s_prev
    #
    # And updates:
    #   s_prev = S_global[last_t]
    # -------------------------------------------------------------------------
    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]  # state for next chunk

    # Remove padding and flip back to original time order
    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values

    return advantages, returns


def calculate_log_probs_and_entropy(logits, tokens, tp_group, with_entropy: bool = False, chunk_size: int = -1):
    logits = logits.contiguous()
    entropy = None
    if logits.size(0) != 0:
        if chunk_size > 0:
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            logits_chunks = logits.chunk(num_chunks, dim=0)
            tokens_chunks = tokens.chunk(num_chunks, dim=0)

            if with_entropy:
                entropys = []
                for logits_chunk in logits_chunks:
                    entropy_input = logits_chunk.clone()
                    entropys.append(compute_entropy_from_logits(entropy_input, tp_group))
                entropy = torch.cat(entropys, dim=0)

            log_probs = []
            for tokens_chunk, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                log_prob = compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group)
                log_probs.append(log_prob)
            log_prob = torch.cat(log_probs, dim=0)
        else:
            if with_entropy:
                entropy_input = logits.clone()
                entropy = compute_entropy_from_logits(entropy_input, tp_group)

            log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
    else:
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            entropy = logits.new_zeros((0,))

    return log_prob, entropy
