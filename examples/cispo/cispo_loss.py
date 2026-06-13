from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping

import torch

from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, get_sum_of_sample_mean
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
from slime.utils.misc import load_function
from slime.utils.ppo_utils import compute_approx_kl, compute_opsm_mask
from slime.utils.types import RolloutBatch


@torch.compile(dynamic=True)
def compute_cispo_loss(
    ppo_kl: torch.Tensor,
    target_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_low_threshold: float,
    clip_high_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token CISPO losses.

    CISPO clips the importance ratio and detaches it before multiplying by
    target log-probs, so the ratio is a bounded coefficient rather than a path
    for gradients.
    """
    ratio = (-ppo_kl).exp()
    clipped_ratio = torch.clamp(ratio, min=clip_low_threshold, max=clip_high_threshold)
    pg_losses = -clipped_ratio.detach() * advantages * target_log_probs
    clipfrac = (clipped_ratio != ratio).float()
    return pg_losses, clipfrac


def _get_cispo_config(args: Namespace) -> tuple[float, float]:
    loss_config = getattr(args, "loss_config", None)
    if isinstance(loss_config, Mapping):
        config = loss_config.get("cispo", {})
    else:
        config = getattr(args, "cispo", {})
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise TypeError("CISPO config must be a mapping. Use loss_config.cispo in --custom-config-path YAML.")

    try:
        clip_low = float(config["clip_low_threshold"])
        clip_high = float(config["clip_high_threshold"])
    except KeyError as exc:
        required_keys = "loss_config.cispo.clip_low_threshold and loss_config.cispo.clip_high_threshold"
        raise KeyError(f"CISPO config requires {required_keys}.") from exc
    if clip_low < 0:
        raise ValueError(f"clip_low_threshold must be non-negative, got {clip_low}.")
    if clip_high <= clip_low:
        raise ValueError(
            f"clip_high_threshold must be greater than clip_low_threshold, got {clip_high} <= {clip_low}."
        )
    return clip_low, clip_high


def _get_sampling_log_probs(
    args: Namespace,
    batch: RolloutBatch,
    target_log_probs: list[torch.Tensor],
) -> list[torch.Tensor]:
    if getattr(args, "use_rollout_logprobs", False):
        rollout_log_probs = batch.get("rollout_log_probs")
        if rollout_log_probs:
            return rollout_log_probs

    old_log_probs = batch.get("log_probs")
    if old_log_probs:
        return old_log_probs

    return [log_prob.detach() for log_prob in target_log_probs]


def _validate_supported_options(args: Namespace) -> None:
    if getattr(args, "use_tis", False):
        raise ValueError(
            "examples.cispo.cispo_loss does not support --use-tis. "
            "CISPO applies a detached clipped ratio inside the policy objective."
        )


def cispo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Megatron custom loss entrypoint for `--loss-type custom_loss`.

    Expected CLI:

        --loss-type custom_loss
        --custom-loss-function-path examples.cispo.cispo_loss.cispo_loss_function
        --custom-config-path examples/cispo/cispo.yaml
    """
    _validate_supported_options(args)
    clip_low, clip_high = _get_cispo_config(args)

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    target_log_probs_list = log_probs_and_entropy["log_probs"]
    sampling_log_probs_list = _get_sampling_log_probs(args, batch, target_log_probs_list)
    train_log_probs_for_mismatch = batch.get("log_probs")
    if not train_log_probs_for_mismatch:
        train_log_probs_for_mismatch = [log_prob.detach() for log_prob in target_log_probs_list]

    if getattr(args, "use_opsm", False):
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                target_log_probs_list, total_lengths, response_lengths, strict=False
            )
        ]
        full_sampling_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                sampling_log_probs_list, total_lengths, response_lengths, strict=False
            )
        ]
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_sampling_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    target_log_probs = torch.cat(target_log_probs_list, dim=0)
    sampling_log_probs = torch.cat(sampling_log_probs_list, dim=0)
    advantages = torch.cat(batch["advantages"], dim=0)

    if target_log_probs.shape != sampling_log_probs.shape or target_log_probs.shape != advantages.shape:
        raise ValueError(
            "CISPO input shapes must match: "
            f"{target_log_probs.shape=}, {sampling_log_probs.shape=}, {advantages.shape=}."
        )

    ppo_kl = sampling_log_probs - target_log_probs
    pg_loss, pg_clipfrac = compute_cispo_loss(ppo_kl, target_log_probs, advantages, clip_low, clip_high)

    if getattr(args, "use_opsm", False):
        pg_loss = pg_loss * opsm_mask

    if getattr(args, "get_mismatch_metrics", False):
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean
        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for mismatch metrics"

        ois = (-ppo_kl).exp()
        mismatch_func = load_function(args.custom_tis_function_path)
        mismatch_pg_loss, modified_response_masks, mismatch_metrics = mismatch_func(
            args=args,
            pg_loss=pg_loss,
            train_log_probs=train_log_probs_for_mismatch,
            rollout_log_probs=batch["rollout_log_probs"],
            loss_masks=batch["loss_masks"],
            total_lengths=total_lengths,
            response_lengths=response_lengths,
        )
        if mismatch_pg_loss is not None:
            pg_loss = mismatch_pg_loss
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            batch["rollout_mask_sums"],
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        pg_loss_masks = (
            modified_response_masks if getattr(args, "get_mismatch_metrics", False) else batch["loss_masks"]
        )
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(target_log_probs - sampling_log_probs)
        kl = compute_approx_kl(
            target_log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    if target_log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    rollout_log_probs = batch.get("rollout_log_probs")
    if rollout_log_probs:
        rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((sampling_log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }
    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()
    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()
    if getattr(args, "get_mismatch_metrics", False):
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        for metric_key, metric_value in mismatch_metrics.items():
            reported_loss[metric_key] = sum_of_sample_mean_for_mismatch_metrics(metric_value)
    if getattr(args, "use_opsm", False):
        reported_loss["opsm_clipfrac"] = opsm_clipfrac
    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss
