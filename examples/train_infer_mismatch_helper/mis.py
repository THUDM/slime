from typing import Any, Dict, Optional, Tuple

import torch

from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp


def masked_sum(x: torch.Tensor, loss_mask: torch.Tensor, expand: bool = False) -> torch.Tensor:
    result = (x * loss_mask).sum()
    return result.expand_as(x) if expand else result


def masked_mean(x: torch.Tensor, loss_mask: torch.Tensor, expand: bool = False) -> torch.Tensor:
    result = masked_sum(x, loss_mask) / torch.clamp_min(loss_mask.sum(), 1)
    return result.expand_as(x) if expand else result


def metrics_append(metrics: Dict[str, list[torch.Tensor]], key: str, value: torch.Tensor) -> None:
    """

    Every metrics-dict value is a list of 1D tensor, i.e., [torch.Tensor] with shapes exactly the same as log_probs.

    All metrics will be aggregated and averaged by `sum_of_sample_mean` and divided by DP size automatically
    - If calculate_per_token_loss=False (default), the final results will first be averaged in each sequence,
      then across all the sequences in the global batch.
    - If calculate_per_token_loss=True, the final results will be the mean of all the tokens in the global batch.

    No need to specifically handle loss_mask, sum_of_sample_mean automatically ignores statistics where loss_mask = 0.

    e.g.
    For token-level metric:
        value = [
            [0.1, 0.2],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6]
        ]
        When calculate_per_token_loss = False (default):
            result = (0.1 + 0.2) / 2 + (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5 + (0.6) / 1 = 0.15 + 0.3 + 0.6 = 1.05 / 3 = 0.35
        When calculate_per_token_loss = True:
            result = (0.1 + 0.2 + 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6) / 8 = 2.4 / 8 = 0.3
    For sequence-level metric:
        original sequence lengths = [2, 5, 2]
        We should expand the metrics to the length of each sequence:
        value = [
            [2, 2],
            [5, 5, 5, 5, 5],
            [1, 1]
        ]
        When calculate_per_token_loss = False (default):
            result = (2 + 2) / 2 + (5 + 5 + 5 + 5 + 5) / 5 + (1 + 1) / 2 = 2 + 5 + 1 = 8 / 3 = 2.6665
        Note that for sequence-level, calculating token-level loss is invalid; thus, calculate_per_token_loss should always be False.
    """
    if key not in metrics:
        metrics[key] = []
    metrics[key].append(value.clone().detach())


def calculate_veto_mask(
    log_ratio: torch.Tensor,
    loss_mask: torch.Tensor,
    veto_threshold: Optional[float],
    metrics: Dict[str, list[torch.Tensor]],
) -> torch.Tensor:
    if veto_threshold is None:
        return torch.ones_like(log_ratio)
    log_veto_threshold = torch.log(torch.tensor(veto_threshold, device=log_ratio.device))
    # For each sequence, if it has any catastrophic tokens, return 0 for the sequence
    catastrophic_tokens = ((log_ratio < log_veto_threshold)) & loss_mask.bool()
    has_catastrophic = catastrophic_tokens.any()
    veto_mask = (~has_catastrophic).float().expand_as(log_ratio)

    metrics_append(metrics, "catastrophic_token_fraction", catastrophic_tokens.int())
    metrics_append(metrics, "catastrophic_seq_fraction", has_catastrophic.int().expand_as(loss_mask))
    return veto_mask


def truncate(
    weights: torch.Tensor, loss_mask: torch.Tensor, metrics: Dict[str, list[torch.Tensor]], upper_bound: float
) -> torch.Tensor:
    assert upper_bound is not None
    metrics_append(metrics, "truncate_fraction", (weights > upper_bound).int())
    return weights.clamp(0, upper_bound) * loss_mask


def clip(
    weights: torch.Tensor,
    loss_mask: torch.Tensor,
    metrics: Dict[str, list[torch.Tensor]],
    lower_bound: float,
    upper_bound: float,
) -> torch.Tensor:
    assert lower_bound is not None and upper_bound is not None and lower_bound < upper_bound
    metrics_append(metrics, "clip_fraction_low", (weights < lower_bound).int())
    metrics_append(metrics, "clip_fraction_high", (weights > upper_bound).int())
    return weights.clamp(lower_bound, upper_bound) * loss_mask


def mask(
    weights: torch.Tensor,
    loss_mask: torch.Tensor,
    metrics: Dict[str, list[torch.Tensor]],
    lower_bound: float,
    upper_bound: float,
) -> torch.Tensor:
    assert lower_bound is not None and upper_bound is not None and lower_bound < upper_bound
    metrics_append(metrics, "mask_fraction_low", (weights < lower_bound).int())
    metrics_append(metrics, "mask_fraction_high", (weights > upper_bound).int())
    mask = (weights >= lower_bound) & (weights <= upper_bound)
    return weights * mask * loss_mask


def compute_mis_weights(
    args,
    *,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> Tuple[list[torch.Tensor], Dict[str, list[torch.Tensor]]]:
    """
    Compute the importance sampling (IS) weights and metrics between the inference and training engine.
    Args:
        train_log_probs: List of log probs from training backend. 1D tensor each. Lengths can be different.
        rollout_log_probs: List of log probs from inference backend. 1D tensor each.
        loss_masks: List of loss masks. 1D tensor each.
            Note that for single turn RL, the loss_mask is [1] * response_length tensor for each sequence
            For multi-turn RL, the tool response will be marked as 0 in the loss_mask.

    Returns:
        weights: List of importance sampling weights. 1D tensor each.
        metrics: The metrics for the importance sampling weights, a dict of list[torch.Tensor]. 1D tensor each.
    """

    level: str = args.mis_level
    metrics: Dict[str, list[torch.Tensor]] = {}

    if args.mis_lower_bound is None:
        return 1.0 / args.mis_upper_bound

    # Validate input lists have same length and each sequence has matching shapes
    assert (
        len(train_log_probs) == len(rollout_log_probs) == len(loss_masks)
    ), f"Input lists must have the same number of sequences: {len(train_log_probs)} vs {len(rollout_log_probs)} vs {len(loss_masks)}"

    for i, (train, rollout, loss_mask) in enumerate(zip(train_log_probs, rollout_log_probs, loss_masks)):
        assert (
            train.shape == rollout.shape == loss_mask.shape
        ), f"Sequence {i}: shapes must match - train: {train.shape}, rollout: {rollout.shape}, loss_mask: {loss_mask.shape}"

    SAFETY_BOUND = 20.0  # Add a safety bound to avoid exp overflow
    all_weights = []

    # handle each sequence independently
    for train_log_prob, rollout_log_prob, loss_mask in zip(train_log_probs, rollout_log_probs, loss_masks):
        loss_mask = loss_mask.float()
        add_ppl_metrics(train_log_prob, rollout_log_prob, loss_mask, metrics)
        raw_log_ratio_diff = train_log_prob - rollout_log_prob

        # level: The aggregation level for the importance sampling weights.
        if level == "token":
            # Per-token ratio (biased)
            log_ratio_for_metrics = raw_log_ratio_diff
        elif level == "sequence":
            # Product of ratios (unbiased but high variance)
            log_ratio_for_metrics = masked_sum(raw_log_ratio_diff, loss_mask, expand=True)
        elif level == "geometric":
            # Geometric mean of ratios (biased but low variance)
            log_ratio_for_metrics = masked_mean(raw_log_ratio_diff, loss_mask, expand=True)
        else:
            raise ValueError(f"Invalid importance sampling level: {level}")

        log_ratio_safe = torch.clamp(log_ratio_for_metrics, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        weights = torch.exp(log_ratio_safe)
        metrics_append(metrics, "mean_is_weight_before_clip", weights)

        # mask out catastrophic tokens
        if args.mis_veto_threshold is not None:
            veto_mask = calculate_veto_mask(raw_log_ratio_diff, loss_mask, args.mis_veto_threshold, metrics)

        # mode: how to handle the importance sampling weights exceeding the thresholds.
        if args.mis_mode == "truncate":
            # Cap the importance sampling weights at the upper threshold
            # https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33
            weights = truncate(weights, loss_mask, metrics, args.mis_upper_bound)
        elif args.mis_mode == "mask":
            # Zero the importance sampling weights outside the [lower, upper] range.
            # https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda
            weights = mask(
                weights,
                loss_mask,
                metrics,
                args.mis_lower_bound,
                args.mis_upper_bound,
            )
        elif args.mis_mode == "clip":
            # Clip the importance sampling weights to the [lower, upper] range.
            # Original behavior in slime.
            weights = clip(
                weights,
                loss_mask,
                metrics,
                args.mis_lower_bound,
                args.mis_upper_bound,
            )
        else:
            raise ValueError(f"Unsupported mis_mode: {args.mis_mode}")

        metrics_append(metrics, "ratio_mean_after_mis", weights)
        if args.mis_veto_threshold is not None:
            weights = weights * veto_mask
            metrics_append(metrics, "ratio_mean_after_veto_mask", weights)

        weights = weights.detach()
        all_weights.append(weights)

    return all_weights, metrics


def compute_mis_weights_with_cp(
    args,
    *,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the importance sampling (IS) weights and metrics with context parallel.
    Args:
        train_log_probs: List of log probs from training backend on this cp rank. 1D tensor each. Lengths can be different.
        rollout_log_probs: List of log probs from inference backend on this cp rank. 1D tensor each.
        loss_masks: List of loss masks. 1D tensor each.
        total_lengths: List of total lengths.
        response_lengths: List of response lengths.
    Returns:
        is_weights: Importance sampling weights on this CP rank and flattened along dim=0.
        is_metrics: The metrics for the importance sampling weights, a dict of list[torch.Tensor]. 1D tensor each.
                    Also flattened along dim=0.
    """
    # Gather cp slice from other cp ranks
    full_rollout_log_probs = [
        all_gather_with_cp(log_prob, total_length, response_length)
        for log_prob, total_length, response_length in zip(rollout_log_probs, total_lengths, response_lengths)
    ]
    full_old_log_probs = [
        all_gather_with_cp(old_log_prob, total_length, response_length)
        for old_log_prob, total_length, response_length in zip(train_log_probs, total_lengths, response_lengths)
    ]

    # Main logic for is
    is_weights, is_metrics = compute_mis_weights(
        args=args,
        train_log_probs=full_old_log_probs,
        rollout_log_probs=full_rollout_log_probs,
        loss_masks=loss_masks,
    )

    # Slice out the value shards for this CP rank and concat them into a 1D tensor along dim=0 for loss.py computation.
    def slice_cp_and_concat(
        values: list[torch.Tensor], total_lengths: list[int], response_lengths: list[int]
    ) -> torch.Tensor:
        values = [
            # TODO: A rename of this function?
            slice_log_prob_with_cp(values[i], total_lengths[i], response_lengths[i])
            for i in range(len(values))
        ]
        return torch.cat(values, dim=0)

    result_metrics = {}
    is_weights = slice_cp_and_concat(is_weights, total_lengths, response_lengths)
    for key, values in is_metrics.items():
        key_name = f"mis_{key}"
        values = slice_cp_and_concat(values, total_lengths, response_lengths)
        result_metrics[key_name] = values

    return is_weights, result_metrics


def add_ppl_metrics(
    train_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    loss_mask: torch.Tensor,
    metrics: Dict[str, list[torch.Tensor]],
):
    loss_mask = loss_mask.float()

    # 1. Training policy perplexity metrics
    mean_log_prob_training = masked_mean(train_log_prob, loss_mask, expand=True)
    training_log_ppl = -mean_log_prob_training
    training_ppl = torch.exp(training_log_ppl)
    metrics_append(metrics, "training_log_ppl", training_log_ppl)
    metrics_append(metrics, "training_ppl", training_ppl)

    # 2. Rollout policy perplexity metrics
    mean_log_prob_rollout = masked_mean(rollout_log_prob, loss_mask, expand=True)
    rollout_log_ppl = -mean_log_prob_rollout
    rollout_ppl = torch.exp(rollout_log_ppl)
    metrics_append(metrics, "rollout_log_ppl", rollout_log_ppl)
    metrics_append(metrics, "rollout_ppl", rollout_ppl)

    # 3a. kl: Direct estimator for KL(π_rollout || π_training)
    # This is the standard KL divergence: E[log(π_rollout) - log(π_training)]
    # Positive value means rollout policy is more confident than training policy
    kl_per_token = rollout_log_prob - train_log_prob
    metrics_append(metrics, "kl", kl_per_token)

    # 3b. K3 KL estimator for improved stability
    # More stable for small KL values using: E[exp(log_ratio) - log_ratio - 1]
    # Formula: KL ≈ E[r - log(r) - 1] where r = π_training/π_rollout
    log_ratio = train_log_prob - rollout_log_prob
    k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
    metrics_append(metrics, "k3_kl", k3_kl_matrix)

    # 3c. Log PPL difference (sequence-level perplexity difference)
    # log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
    # Since ppl = exp(-log_prob), we have:
    #   log(ppl_ratio) = log(training_ppl/rollout_ppl) = log_ppl_diff
    # Positive value means training assigns lower probability (higher PPL) than rollout
    log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
    metrics_append(metrics, "log_ppl_diff", log_ppl_diff)
    metrics_append(metrics, "log_ppl_abs_diff", log_ppl_diff.abs())

    # 3d. PPL ratio (how much higher is training PPL vs rollout PPL)
    # For numerical stability, compute in log space using log_ppl_diff
    # Note: log_ppl_diff = log(ppl_ratio), so ppl_ratio = exp(log_ppl_diff)
    ppl_ratio = torch.exp(log_ppl_diff)
    metrics_append(metrics, "ppl_ratio", ppl_ratio)
