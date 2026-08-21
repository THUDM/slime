"""Internal coordination and math helpers for bounded, group-atomic RS batch refill."""

from __future__ import annotations

import hashlib
import logging
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefillSelection:
    """Result of applying per-sample RS reports to prompt groups."""

    accepted_groups: list[list[Any]]
    rejected_groups: list[list[Any]]
    surplus_groups: list[list[Any]]
    target_size: int

    @property
    def deficit(self) -> int:
        return max(0, self.target_size - len(self.accepted_groups))


def _require_integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _resolve_rs_admission_bounds(args) -> tuple[Any, Any]:
    tis_lower_bound = args.tis_lower_bound if args.tis_lower_bound is not None else 1.0 / args.tis_upper_bound
    lower_bound = args.rs_lower_bound if args.rs_lower_bound is not None else tis_lower_bound
    upper_bound = args.rs_upper_bound if args.rs_upper_bound is not None else args.tis_upper_bound
    return lower_bound, upper_bound


def _resolve_rs_admission_log_bounds(args) -> tuple[float, float]:
    lower_bound, upper_bound = _resolve_rs_admission_bounds(args)
    lower_log_bound = -math.inf if lower_bound == 0 else math.log(lower_bound)
    return lower_log_bound, math.log(upper_bound)


def get_rs_refill_candidate_group_multiple(args, train_parallel_config: Mapping[str, Any]) -> int:
    """Return the prompt-group quantum that makes every preflight schedulable."""

    try:
        dp_size = _require_integer(train_parallel_config["dp_size"], "dp_size", positive=True)
        vpp_size = _require_integer(train_parallel_config["vpp_size"], "vpp_size", positive=True)
        microbatch_group_size = _require_integer(
            train_parallel_config["microbatch_group_size_per_vp_stage"],
            "microbatch_group_size_per_vp_stage",
            positive=True,
        )
        samples_per_group = _require_integer(args.n_samples_per_prompt, "n_samples_per_prompt", positive=True)
    except KeyError as error:
        raise ValueError("Invalid training topology for RS refill planning") from error

    aligned_samples = dp_size * (microbatch_group_size if vpp_size > 1 else 1)
    return aligned_samples // math.gcd(aligned_samples, samples_per_group)


def validate_rs_refill_target_batch_alignment(args, train_parallel_config: Mapping[str, Any]) -> None:
    """Fail before rollout when the final effective batch cannot always be scheduled."""

    candidate_group_multiple = get_rs_refill_candidate_group_multiple(args, train_parallel_config)
    if args.rollout_batch_size % candidate_group_multiple == 0:
        return

    dp_size = int(train_parallel_config["dp_size"])
    vpp_size = int(train_parallel_config["vpp_size"])
    microbatch_group_size = int(train_parallel_config["microbatch_group_size_per_vp_stage"])
    aligned_samples = dp_size * (microbatch_group_size if vpp_size > 1 else 1)
    target_samples = args.rollout_batch_size * args.n_samples_per_prompt
    raise ValueError(
        "--rs-batch-refill requires the final effective sample count to be divisible by the DP/VPP "
        "microbatch alignment so every possible dynamic packing remains schedulable: "
        f"rollout_batch_size * n_samples_per_prompt = {target_samples}, alignment = {aligned_samples}. "
        f"Choose a rollout_batch_size divisible by {candidate_group_multiple}."
    )


def plan_topology_aligned_rs_refill(
    args,
    train_parallel_config: Mapping[str, Any],
    required_successes: int,
) -> int:
    """Round an exact deficit up to the smallest schedulable group count."""

    if not isinstance(required_successes, int) or isinstance(required_successes, bool) or required_successes <= 0:
        raise ValueError("required_successes must be a positive integer")
    candidate_multiple = get_rs_refill_candidate_group_multiple(args, train_parallel_config)
    return ((required_successes + candidate_multiple - 1) // candidate_multiple) * candidate_multiple


def compute_sequence_rs_masks(
    args,
    *,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Compute the sequence-local RS admission mask used before and during training."""

    if not (len(train_log_probs) == len(rollout_log_probs) == len(loss_masks)):
        raise ValueError(
            "RS admission inputs must have the same sample count: "
            f"train={len(train_log_probs)}, rollout={len(rollout_log_probs)}, masks={len(loss_masks)}"
        )
    if args.rs_level not in {"sequence", "geometric"}:
        raise ValueError(f"RS batch refill does not support rs_level={args.rs_level!r}")

    lower_log_bound, upper_log_bound = _resolve_rs_admission_log_bounds(args)
    modified_masks = []
    for sample_index, (train, rollout, loss_mask) in enumerate(
        zip(train_log_probs, rollout_log_probs, loss_masks, strict=True)
    ):
        rollout = torch.as_tensor(rollout, device=train.device)
        mask = torch.as_tensor(loss_mask, device=train.device).float()
        if train.shape != rollout.shape or train.shape != mask.shape:
            raise ValueError(
                "RS admission sample shapes must match: "
                f"sample={sample_index}, train={tuple(train.shape)}, rollout={tuple(rollout.shape)}, "
                f"mask={tuple(mask.shape)}"
            )

        raw_log_ratio = train - rollout
        finite_log_ratio = torch.isfinite(raw_log_ratio).all()
        safe_log_ratio = torch.where(torch.isfinite(raw_log_ratio), raw_log_ratio, 0.0)
        sequence_log_ratio = (safe_log_ratio * mask).sum()
        if args.rs_level == "geometric":
            sequence_log_ratio = sequence_log_ratio / torch.clamp_min(mask.sum(), 1)
        accepted = finite_log_ratio & (sequence_log_ratio >= lower_log_bound) & (sequence_log_ratio <= upper_log_bound)
        if args.rs_veto_threshold is not None:
            veto_log_threshold = torch.log(
                torch.tensor(args.rs_veto_threshold, device=raw_log_ratio.device, dtype=torch.float32)
            )
            accepted = accepted & ~((raw_log_ratio < veto_log_threshold) & mask.bool()).any()
        modified_masks.append((mask * accepted).detach())

    return modified_masks


def apply_rs_refill_tis(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **_: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Apply bounded TIS weights and exactly the same RS rule as preflight."""

    if not (len(train_log_probs) == len(rollout_log_probs) == len(loss_masks)):
        raise ValueError(
            "RS refill TIS inputs must have the same sample count: "
            f"train={len(train_log_probs)}, rollout={len(rollout_log_probs)}, masks={len(loss_masks)}"
        )
    if not train_log_probs:
        raise ValueError("RS refill TIS requires at least one sample")
    if args.tis_batch_normalize:
        raise ValueError("RS refill does not support DP-local TIS batch normalization")

    metrics: dict[str, list[torch.Tensor]] = {}

    def append_metric(key: str, value: torch.Tensor) -> None:
        metrics.setdefault(key, []).append(value.clone().detach())

    def masked_sum(value: torch.Tensor, mask: torch.Tensor, *, expand: bool = False) -> torch.Tensor:
        result = (value * mask).sum()
        return result.expand_as(value) if expand else result

    def masked_mean(value: torch.Tensor, mask: torch.Tensor, *, expand: bool = False) -> torch.Tensor:
        result = masked_sum(value, mask) / torch.clamp_min(mask.sum(), 1)
        return result.expand_as(value) if expand else result

    def aggregate_log_ratio(raw_log_ratio: torch.Tensor, mask: torch.Tensor, level: str) -> torch.Tensor:
        if level == "token":
            return raw_log_ratio
        if level == "sequence":
            return masked_sum(raw_log_ratio, mask, expand=True)
        if level == "geometric":
            return masked_mean(raw_log_ratio, mask, expand=True)
        raise ValueError(f"RS refill TIS does not support tis_level={level!r}")

    tis_lower_bound = args.tis_lower_bound if args.tis_lower_bound is not None else 1.0 / args.tis_upper_bound
    all_weights = []
    normalized_train_log_probs = []
    normalized_rollout_log_probs = []
    normalized_loss_masks = []
    for sample_index, (train, rollout, loss_mask) in enumerate(
        zip(train_log_probs, rollout_log_probs, loss_masks, strict=True)
    ):
        rollout = torch.as_tensor(rollout, device=train.device)
        mask = torch.as_tensor(loss_mask, device=train.device).float()
        if train.shape != rollout.shape or train.shape != mask.shape:
            raise ValueError(
                "RS refill TIS sample shapes must match: "
                f"sample={sample_index}, train={tuple(train.shape)}, rollout={tuple(rollout.shape)}, "
                f"mask={tuple(mask.shape)}"
            )

        raw_log_ratio = train - rollout
        mean_train_log_prob = masked_mean(train, mask, expand=True)
        mean_rollout_log_prob = masked_mean(rollout, mask, expand=True)
        training_log_ppl = -mean_train_log_prob
        rollout_log_ppl = -mean_rollout_log_prob
        log_ppl_diff = mean_rollout_log_prob - mean_train_log_prob
        append_metric("mis_training_log_ppl", training_log_ppl)
        append_metric("mis_training_ppl", torch.exp(training_log_ppl))
        append_metric("mis_rollout_log_ppl", rollout_log_ppl)
        append_metric("mis_rollout_ppl", torch.exp(rollout_log_ppl))
        append_metric("mis_kl", rollout - train)
        append_metric("mis_k3_kl", torch.exp(raw_log_ratio) - raw_log_ratio - 1)
        append_metric("mis_log_ppl_diff", log_ppl_diff)
        append_metric("mis_log_ppl_abs_diff", log_ppl_diff.abs())
        append_metric("mis_ppl_ratio", torch.exp(log_ppl_diff))
        safe_raw_log_ratio = torch.clamp(raw_log_ratio, min=-20.0, max=20.0)
        chi2_token = masked_mean(torch.exp(safe_raw_log_ratio).square(), mask) - 1.0
        append_metric("mis_chi2_token", chi2_token.expand_as(train))
        sequence_log_ratio = torch.clamp(masked_sum(raw_log_ratio, mask, expand=True), min=-20.0, max=20.0)
        append_metric("mis_chi2_seq", torch.exp(2.0 * sequence_log_ratio) - 1.0)

        log_ratio = aggregate_log_ratio(raw_log_ratio, mask, args.tis_level)
        weights = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
        append_metric("mis_tis_weight_before_bound", weights)
        if args.tis_mode == "truncate":
            append_metric("mis_tis_truncate_fraction", (weights > args.tis_upper_bound).int())
            weights = weights.clamp(0, args.tis_upper_bound) * mask
        elif args.tis_mode == "clip":
            append_metric("mis_tis_clip_fraction_low", (weights < tis_lower_bound).int())
            append_metric("mis_tis_clip_fraction_high", (weights > args.tis_upper_bound).int())
            weights = weights.clamp(tis_lower_bound, args.tis_upper_bound) * mask
        else:
            raise ValueError(f"RS refill TIS does not support tis_mode={args.tis_mode!r}")
        append_metric("mis_tis_weight_after_bound", weights)
        append_metric("mis_is_ratio_mean_after_tis_rs", weights)

        all_weights.append(weights.detach())
        normalized_train_log_probs.append(train)
        normalized_rollout_log_probs.append(rollout)
        normalized_loss_masks.append(mask)

    modified_masks = compute_sequence_rs_masks(
        args,
        train_log_probs=normalized_train_log_probs,
        rollout_log_probs=normalized_rollout_log_probs,
        loss_masks=normalized_loss_masks,
    )

    lower_log_bound, upper_log_bound = _resolve_rs_admission_log_bounds(args)
    for train, rollout, mask in zip(
        normalized_train_log_probs, normalized_rollout_log_probs, normalized_loss_masks, strict=True
    ):
        raw_log_ratio = train - rollout
        rs_log_ratio = aggregate_log_ratio(raw_log_ratio, mask, args.rs_level)
        append_metric("mis_rs_mask_fraction_low", (rs_log_ratio < lower_log_bound).int())
        append_metric("mis_rs_mask_fraction_high", (rs_log_ratio > upper_log_bound).int())
        if args.rs_veto_threshold is not None:
            veto_log_threshold = torch.log(
                torch.tensor(args.rs_veto_threshold, device=raw_log_ratio.device, dtype=torch.float32)
            )
            catastrophic_tokens = (raw_log_ratio < veto_log_threshold) & mask.bool()
            append_metric("mis_rs_catastrophic_token_fraction", catastrophic_tokens.int())
            append_metric("mis_rs_catastrophic_seq_fraction", catastrophic_tokens.any().int().expand_as(mask))

    for weight, mask in zip(all_weights, normalized_loss_masks, strict=True):
        valid = mask.bool()
        mean = masked_mean(weight, mask, expand=True)
        minimum = weight[valid].min() if valid.any() else weight.new_tensor(0.0)
        maximum = weight[valid].max() if valid.any() else weight.new_tensor(0.0)
        append_metric("mis_is_ratio_mean_final", mean)
        append_metric("mis_is_ratio_min_final", minimum.expand_as(weight))
        append_metric("mis_is_ratio_max_final", maximum.expand_as(weight))

    flat_weights = torch.cat(all_weights, dim=0)
    if flat_weights.shape != pg_loss.shape:
        raise ValueError(
            "RS refill TIS weights must match the policy-gradient loss shape: "
            f"weights={tuple(flat_weights.shape)}, pg_loss={tuple(pg_loss.shape)}"
        )
    flat_metrics = {key: torch.cat(values, dim=0) for key, values in metrics.items()}
    return pg_loss * flat_weights, modified_masks, flat_metrics


def merge_replacement_metrics(
    destination: dict[str, Any],
    source: dict[str, Any],
    *,
    round_index: int,
) -> None:
    """Merge counters with known semantics and namespace every other metric."""

    for key, value in source.items():
        if key.startswith("rollout/dynamic_filter/drop_"):
            destination[key] = destination.get(key, 0) + value
        else:
            destination[f"rollout/rs_refill/replacement_round_{round_index}/{key}"] = value


def validate_refill_rollout_ids(
    groups: list[list[Any]],
    *,
    known_rollout_ids: set[Any] | None = None,
) -> set[Any]:
    """Require explicit effective rollout IDs to be unique across refill rounds."""

    seen = set(known_rollout_ids or ())
    candidate_ids = set()
    for group in groups:
        for sample in group:
            rollout_id = sample.rollout_id
            if rollout_id is None:
                continue
            try:
                duplicate = rollout_id in seen
            except TypeError as error:
                raise ValueError(f"RS refill rollout_id must be hashable, got {rollout_id!r}") from error
            if duplicate:
                raise ValueError(
                    "RS batch refill requires one unique effective rollout_id per training sample; "
                    f"duplicate={rollout_id!r}. Compact/fan-out rollouts are not supported yet."
                )
            seen.add(rollout_id)
            candidate_ids.add(rollout_id)
    return candidate_ids


def merge_selected_log_prob_caches(
    worker_caches: list[dict[int, Any] | None],
    expected_sample_indices: list[int],
) -> dict[int, Any]:
    """Merge actor-local caches and require exactly the selected samples."""

    expected_indices = [_require_integer(index, "selected sample index") for index in expected_sample_indices]
    expected = set(expected_indices)
    if len(expected) != len(expected_indices):
        raise ValueError("selected RS sample indices must be unique")

    merged: dict[int, Any] = {}
    for worker_cache in worker_caches:
        if not worker_cache:
            continue
        for sample_index, log_probs in worker_cache.items():
            sample_index = _require_integer(sample_index, "proximal logprob cache sample index")
            if sample_index in merged:
                raise ValueError(f"duplicate proximal logprob cache for sample_index={sample_index}")
            merged[sample_index] = log_probs

    actual = set(merged)
    if actual != expected:
        raise ValueError(
            "Selected RS proximal logprob cache is incomplete: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return merged


def attach_proximal_log_probs(
    train_data: dict[str, Any],
    samples: list[Any],
    log_probs_by_sample_index: dict[int, Any],
) -> None:
    """Attach an in-memory preflight cache to the final batch exactly once."""

    if "rs_preflight_log_probs" in train_data:
        raise ValueError("RS preflight log probabilities are already attached to the final batch")
    sample_indices = [_require_integer(sample.index, "RS-refilled sample index") for sample in samples]
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("RS-refilled samples must have unique sample indices")

    expected = set(sample_indices)
    cached = set(log_probs_by_sample_index)
    if cached != expected:
        raise ValueError(
            "RS proximal logprob cache does not match the final batch: "
            f"missing={sorted(expected - cached)}, extra={sorted(cached - expected)}"
        )
    train_data["rs_preflight_log_probs"] = [log_probs_by_sample_index[index] for index in sample_indices]


def snapshot_sample_masks(samples: list[Any]) -> dict[int, tuple[int, bytes]]:
    """Snapshot response lengths and fixed-size mask digests used by the preflight gate."""

    fingerprints: dict[int, tuple[int, bytes]] = {}
    for sample in samples:
        sample_index = _require_integer(sample.index, "RS-refilled sample index")
        if sample_index in fingerprints:
            raise ValueError(f"duplicate sample index in RS mask snapshot: {sample_index}")
        if sample.loss_mask is None:
            raise ValueError(f"sample_index={sample_index} has no loss mask after RS preflight")
        response_length = _require_integer(sample.response_length, "RS-refilled response length")
        mask = torch.as_tensor(sample.loss_mask)
        if mask.ndim != 1 or mask.numel() != response_length:
            raise ValueError(
                "RS-refilled loss mask must be one-dimensional and match response_length: "
                f"sample_index={sample_index}, mask_shape={tuple(mask.shape)}, response_length={response_length}"
            )
        if not torch.logical_or(mask == 0, mask == 1).all().item():
            raise ValueError(f"sample_index={sample_index} has a non-binary loss mask after RS preflight")
        mask_bytes = mask.to(dtype=torch.uint8, device="cpu").contiguous().numpy().tobytes()
        fingerprints[sample_index] = (response_length, hashlib.sha256(mask_bytes).digest())
    return fingerprints


def validate_sample_masks(
    samples: list[Any],
    expected: dict[int, tuple[int, bytes]],
) -> None:
    """Reject post-gate mutations that would change the effective batch."""

    actual = snapshot_sample_masks(samples)
    if actual != expected:
        expected_ids = set(expected)
        actual_ids = set(actual)
        changed = sorted(index for index in expected_ids & actual_ids if expected[index] != actual[index])
        raise RuntimeError(
            "RS-refilled sample masks changed after proximal preflight: "
            f"missing={sorted(expected_ids - actual_ids)}, extra={sorted(actual_ids - expected_ids)}, "
            f"changed={changed}"
        )


def clone_rs_masks(masks: list[Any]) -> list[torch.Tensor]:
    """Clone masks before invoking an extension that may mutate its inputs."""

    return [torch.as_tensor(mask).clone() for mask in masks]


def validate_final_rs_masks(original_masks: list[Any], *candidate_mask_sets: list[Any]) -> None:
    """Require the training-time RS gate to preserve every preflight-accepted token."""

    if not candidate_mask_sets:
        raise ValueError("at least one candidate mask set is required")

    shape_changed: set[int] = set()
    value_checks = []
    check_positions = []
    for candidate_masks in candidate_mask_sets:
        if len(original_masks) != len(candidate_masks):
            raise RuntimeError(
                "RS training gate returned a different sample count from preflight: "
                f"original={len(original_masks)}, modified={len(candidate_masks)}"
            )
        for index, (original, modified) in enumerate(zip(original_masks, candidate_masks, strict=True)):
            original = torch.as_tensor(original, dtype=torch.float32)
            modified = torch.as_tensor(modified, dtype=torch.float32, device=original.device)
            if original.shape != modified.shape:
                shape_changed.add(index)
                continue
            value_checks.append(torch.all(original == modified))
            check_positions.append(index)

    changed = set(shape_changed)
    if value_checks:
        failed_checks = (~torch.stack(value_checks)).nonzero().flatten().tolist()
        changed.update(check_positions[check_index] for check_index in failed_checks)
    if changed:
        raise RuntimeError(
            "RS training gate rejected or changed samples accepted by preflight: "
            f"microbatch_positions={sorted(changed)}"
        )


def validate_replacement_policy_version(groups: list[list[Any]], reports: list[dict[str, Any]]) -> str:
    """Require every reactive replacement to come from the scored policy version."""

    policy_versions = {str(report["policy_version"]) for report in reports}
    if len(policy_versions) != 1:
        raise ValueError(f"RS refill trainer ranks disagree on policy version: {policy_versions}")
    expected_version = next(iter(policy_versions))

    for group in groups:
        for sample in group:
            generated_versions = {str(version) for version in sample.weight_versions}
            if generated_versions != {expected_version}:
                raise ValueError(
                    "Reactive RS replacement was not generated entirely by the current rollout policy: "
                    f"sample_index={sample.index}, generated={sorted(generated_versions)}, "
                    f"expected={expected_version}."
                )
    return expected_version


def validate_initial_policy_staleness(
    groups: list[list[Any]],
    reports: list[dict[str, Any]],
    *,
    max_staleness: int = 1,
) -> int:
    """Require one initial rollout version no more than ``max_staleness`` behind the actor."""

    if max_staleness < 0:
        raise ValueError("max_staleness must be non-negative")
    report_versions = {str(report["policy_version"]) for report in reports}
    if len(report_versions) != 1:
        raise ValueError(f"RS refill trainer ranks disagree on policy version: {report_versions}")

    generated_versions: set[str] = set()
    for group in groups:
        for sample in group:
            sample_versions = {str(version) for version in sample.weight_versions}
            if len(sample_versions) != 1:
                raise ValueError(
                    "Each initial RS candidate must come from exactly one rollout policy version: "
                    f"sample_index={sample.index}, versions={sorted(sample_versions)}"
                )
            generated_versions.update(sample_versions)
    if len(generated_versions) != 1:
        raise ValueError(f"Initial RS candidates span multiple rollout policy versions: {sorted(generated_versions)}")

    actor_version_text = next(iter(report_versions))
    rollout_version_text = next(iter(generated_versions))
    try:
        actor_version = int(actor_version_text)
        rollout_version = int(rollout_version_text)
    except ValueError as error:
        raise ValueError(
            "RS batch refill requires integer actor and rollout policy versions, "
            f"got actor={actor_version_text!r}, rollout={rollout_version_text!r}"
        ) from error

    staleness = actor_version - rollout_version
    if not 0 <= staleness <= max_staleness:
        raise ValueError(
            "Initial RS candidates exceed the hard policy-staleness bound: "
            f"actor={actor_version}, rollout={rollout_version}, staleness={staleness}, "
            f"allowed=[0, {max_staleness}]"
        )
    return staleness


def select_accepted_groups(
    groups: list[list[Any]],
    reports: list[dict[str, Any]],
    *,
    target_size: int,
    known_sample_indices: set[int] | None = None,
    known_group_indices: set[int] | None = None,
) -> RefillSelection:
    """Select complete prompt groups in input order without mismatch ranking."""

    if target_size <= 0:
        raise ValueError("target_size must be positive")

    report_by_index: dict[int, dict[str, Any]] = {}
    for report in reports:
        sample_index = report["sample_index"]
        if sample_index in report_by_index:
            raise ValueError(f"duplicate RS report for sample_index={sample_index}")
        report_by_index[sample_index] = report

    accepted: list[list[Any]] = []
    rejected: list[list[Any]] = []
    surplus: list[list[Any]] = []
    seen_sample_indices = set(known_sample_indices or ())
    seen_group_indices = set(known_group_indices or ())
    candidate_sample_indices: set[int] = set()

    for group in groups:
        if not group:
            raise ValueError("RS refill received an empty prompt group")
        group_indices = {sample.group_index for sample in group}
        if len(group_indices) != 1:
            raise ValueError(f"samples in one prompt group have different group_index values: {group_indices}")
        group_index = next(iter(group_indices))
        if group_index in seen_group_indices:
            raise ValueError(f"duplicate prompt group_index in RS candidates: {group_index}")
        seen_group_indices.add(group_index)

        group_accepted = True
        for sample in group:
            if sample.index in seen_sample_indices:
                raise ValueError(f"duplicate sample index in candidate groups: {sample.index}")
            seen_sample_indices.add(sample.index)
            candidate_sample_indices.add(sample.index)
            if sample.index not in report_by_index:
                raise ValueError(f"missing RS report for sample_index={sample.index}")

            report = report_by_index[sample.index]
            if report.get("group_index") != sample.group_index:
                raise ValueError(
                    f"RS report group mismatch for sample_index={sample.index}: "
                    f"sample={sample.group_index}, report={report.get('group_index')}"
                )
            if int(report["valid_tokens"]) <= 0 or not bool(report["gate_passed"]):
                group_accepted = False

        if group_accepted and len(accepted) < target_size:
            accepted.append(group)
        elif group_accepted:
            surplus.append(group)
        else:
            rejected.append(group)

    unknown_reports = set(report_by_index) - candidate_sample_indices
    if unknown_reports:
        raise ValueError(f"RS reports contain unknown sample indices: {sorted(unknown_reports)}")

    return RefillSelection(accepted, rejected, surplus, target_size)


def run_rs_batch_refill(
    actor_model,
    rollout_manager,
    rollout_id: int,
    *,
    resolve: Callable[[Any], Any],
    clock: Callable[[], float],
    rpc_timeout_seconds: float = 1800.0,
):
    """Drive bounded refill rounds through Ray-like actor interfaces."""

    if not math.isfinite(rpc_timeout_seconds) or rpc_timeout_seconds <= 0:
        raise ValueError("rpc_timeout_seconds must be finite and positive")

    def resolve_rpc(value):
        return resolve(value, timeout=rpc_timeout_seconds)

    coordinator_start = clock()
    try:
        while True:
            preflight_start = clock()
            candidate_refs = resolve_rpc(rollout_manager.prepare_rs_candidate_data.remote(rollout_id))
            report_refs = actor_model.async_score_rs_candidates(rollout_id, candidate_refs)
            preflight_seconds = clock() - preflight_start
            status = resolve_rpc(
                rollout_manager.apply_rs_candidate_reports.remote(rollout_id, report_refs, preflight_seconds)
            )
            selected_cache_refs = actor_model.async_take_rs_candidate_log_probs(
                rollout_id,
                status["accepted_sample_indices"],
            )
            resolve_rpc(rollout_manager.store_rs_accepted_log_probs.remote(rollout_id, selected_cache_refs))

            if status["complete"]:
                coordinator_seconds = clock() - coordinator_start
                return resolve_rpc(rollout_manager.finalize_rs_batch.remote(rollout_id, coordinator_seconds))
            if status["exhausted"]:
                raise RuntimeError(
                    "RS batch refill exhausted its retry budget before optimizer.step: "
                    f"accepted={status['accepted_groups']}, target={status['target_groups']}, "
                    f"rounds={status['round']}, remaining={status['deficit']}."
                )
            # Replacement generation retains the rollout backend's own timeout
            # and health-monitor semantics.  The coordination timeout applies
            # only after generation has returned.
            resolve(rollout_manager.generate_rs_replacement_candidates.remote(rollout_id))
    except Exception:
        actor_cleanup = None
        manager_cleanup = None
        try:
            actor_cleanup = actor_model.async_discard_rs_candidate_log_probs(rollout_id)
        except Exception:
            logger.exception("Failed to submit actor-local RS candidate cache cleanup after a coordination error")
        try:
            manager_cleanup = rollout_manager.abort_rs_batch.remote(rollout_id)
        except Exception:
            logger.exception("Failed to submit manager-local RS pending-state cleanup after a coordination error")
        if actor_cleanup is not None:
            try:
                resolve_rpc(actor_cleanup)
            except Exception:
                logger.exception("Failed to discard actor-local RS candidate caches after a coordination error")
        if manager_cleanup is not None:
            try:
                resolve_rpc(manager_cleanup)
            except Exception:
                logger.exception("Failed to discard manager-local RS pending state after a coordination error")
        raise
