from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any, Dict, Tuple, Union

import torch
from megatron.core import mpu

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_policy_loss,
    get_advantages_and_returns,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from slime.utils.types import RolloutBatch

from .cp_utils import all_gather_with_cp, get_logits_and_tokens_offset_with_cp, get_sum_of_sample_mean


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    assert logits.size(0) == 1, f"{logits.shape}"
    assert logits.dtype == torch.float32, f"{logits.dtype}"

    logits = logits.squeeze(0)
    logits = logits.div(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    for tokens, total_length, response_length in zip(unconcat_tokens, total_lengths, response_lengths):
        if cp_size == 1:
            end += total_length
            start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        yield logits_chunk, tokens_chunk


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    log_probs_list = []
    entropy_list = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    ):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk, tokens_chunk, mpu.get_tensor_model_parallel_group(), with_entropy=with_entropy
        )

        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    return res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    non_loss_data: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    return {
        "values": value_list,
    }


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". When `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    log_probs: list[torch.Tensor] = rollout_data.get("log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: Union[None, list[torch.Tensor]] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")

    # return when not the last pp stage.
    if log_probs is None and values is None:
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    if args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        # TODO: optimize this
        old_rewards = rewards
        rewards = []
        for reward, k in zip(old_rewards, kl):
            k *= -args.kl_coef
            cp_rank = mpu.get_context_parallel_rank()
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = list(
            zip(
                *[
                    get_advantages_and_returns(total_length, response_length, value, reward, args.gamma, args.lambd)
                    for total_length, response_length, value, reward in zip(
                        total_lengths, response_lengths, values, rewards
                    )
                ]
            )
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(total_len, response_len)

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Temporal Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
    )

    log_probs = log_probs_and_entropy["log_probs"]

    if args.advantage_estimator == "gspo":
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(log_probs, total_lengths, response_lengths)
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(old_log_probs, total_lengths, response_lengths)
        ]

        loss_masks = batch["loss_masks"]
        ppo_kl = [
            ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
            for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks)
        ]
        ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, log_probs)]
        ppo_kl = torch.cat(ppo_kl, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    # Apply off-policy correction using importance sampling if enabled
    if args.use_tis:

        def vanilla_tis_function(
            args,
            *,
            train_log_probs: list[torch.Tensor],
            rollout_log_probs: list[torch.Tensor],
            **kwargs: Any,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
            old_log_probs = torch.cat(train_log_probs, dim=0)
            tis = torch.exp(old_log_probs - rollout_log_probs)
            tis_abs = torch.exp((old_log_probs - rollout_log_probs).abs())
            tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
            tis_clipfrac = (tis_weights != tis).float()
            metrics = {
                "tis": tis.clone().detach(),
                "tis_clipfrac": tis_clipfrac.clone().detach(),
                "tis_abs": tis_abs.clone().detach(),
            }
            return tis_weights, metrics

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        tis_weights, tis_metrics = tis_func(**tis_kwargs)

        pg_loss = pg_loss * tis_weights

    pg_loss = sum_of_sample_mean(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.use_tis:
        reported_loss["ois"] = sum_of_sample_mean(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean(metric_value)

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    num_samples = len(batch["response_lengths"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
    )

    loss_function_kwargs = {
        "args": args,
        "batch": batch,
        "logits": logits,
        "sum_of_sample_mean": sum_of_sample_mean,
    }

    match args.loss_type:
        case "policy_loss":
            loss, log = policy_loss_function(**loss_function_kwargs)
        case "value_loss":
            loss, log = value_loss_function(**loss_function_kwargs)
        case "sft_loss":
            loss, log = sft_loss_function(**loss_function_kwargs)
        case "custom_loss":
            custom_loss_function = load_function(args.custom_loss_function_path)
            loss, log = custom_loss_function(**loss_function_kwargs)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    loss = (
        loss * num_microbatches / args.global_batch_size * mpu.get_data_parallel_world_size(with_context_parallel=True)
    )

    return (
        loss,
        num_tokens if args.calculate_per_token_loss else 1,
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not args.calculate_per_token_loss else num_tokens,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
