"""Per-rollout DP/microbatch scheduling.

Pure-Python logic that decides, for one rollout's worth of sample lengths,
how to group samples into micro-batches and which DP rank owns each mbs.
Lives outside the ray/sglang-importing modules so it can be unit-tested
under CPU-only CI.

The scheduling philosophy is **pack first, distribute second**:

  1. Group samples by rollout id (``rollout_indices[i]`` =
     ``samples[i].index``) and split rollouts into steps of
     ``global_batch_size`` rollouts each. In the common case one rollout
     emits one training sample so this is the same as a contiguous chunk;
     under compact / subagent one rollout may emit multiple training
     samples, in which case all of those samples stay in the same step.
  2. For each step, pack its samples into ``K`` micro-batches with a
     single first-fit pass (dynamic batch) or fixed-size chunking
     (static batch).
  3. Adjust ``K`` to a multiple of ``dp_size * (mb_group if vpp>1 else 1)``
     by splitting the largest multi-sample bins (dynamic only).
  4. Distribute the ``K`` mbs across ``dp_size`` ranks, ``K / dp_size``
     each, with either a strided round-robin or a Karmarkar-Karp pass on
     mbs token sums.

Invariants guaranteed by :func:`build_dp_schedule` (asserted by the tests):
  - every DP rank runs the **same** ``num_microbatches`` per training step
    (required for PP sync);
  - every mbs (dynamic path) holds ``<= max_tokens_per_gpu * cp_size``
    tokens, with one exception — an individual sample larger than that cap
    lands alone in its own mbs (and that mbs is the only one allowed to
    exceed the cap);
  - the union of per-rank sample indices equals the set of samples kept
    after trimming trailing rollouts (every kept sample placed exactly
    once);
  - flattening ``micro_batch_indices`` for a rank yields
    ``range(num_samples_rank)`` (each rank's samples are tiled exactly
    once by its mbs schedule).
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils.seqlen_balancing import expand_bins_by_splitting, first_fit_pack, get_seqlen_balanced_partitions

logger = logging.getLogger(__name__)


def _pack_step_into_mbs(
    step_lengths: list[int],
    *,
    use_dynamic_batch_size: bool,
    max_per_bin: int | None,
    micro_batch_size: int | None,
    multimodal_aware_packing: str = "off",
    step_is_multimodal: list[bool] | None = None,
) -> list[list[int]]:
    """Group a step's samples into mbs. Returns ``mbs[k]`` = local indices into ``step_lengths``."""
    if use_dynamic_batch_size:
        assert max_per_bin is not None
        if (
            multimodal_aware_packing in {"separate", "separate_raw"}
            and step_is_multimodal is not None
            and any(step_is_multimodal)
        ):
            return _first_fit_pack_separate_multimodal(
                step_lengths,
                step_is_multimodal,
                max_per_bin,
                padded_multimodal_cost=multimodal_aware_packing == "separate",
            )
        return first_fit_pack(step_lengths, max_per_bin)
    assert micro_batch_size is not None
    n = len(step_lengths)
    return [list(range(i, min(i + micro_batch_size, n))) for i in range(0, n, micro_batch_size)]


def _first_fit_pack_separate_multimodal(
    step_lengths: list[int],
    step_is_multimodal: list[bool],
    max_tokens_per_bin: int,
    *,
    padded_multimodal_cost: bool,
) -> list[list[int]]:
    """First-fit packing that does not mix true multimodal and text-only samples.

    ``separate_raw`` uses the usual sum of token lengths for both text-only and
    multimodal bins while still avoiding text/mm mixing. ``separate`` uses
    ``len(bin) * max(seq_len in bin)`` for multimodal bins because the QwenVL
    unsplit path pads samples in the same microbatch to the longest sequence
    before entering the wrapper.
    """
    assert len(step_lengths) == len(step_is_multimodal)

    bins: list[list[int]] = []
    bin_is_multimodal: list[bool] = []
    bin_sums: list[int] = []
    bin_maxes: list[int] = []

    for idx, length in enumerate(step_lengths):
        is_mm = step_is_multimodal[idx]
        for j, bin_ in enumerate(bins):
            if bin_is_multimodal[j] != is_mm:
                continue
            if is_mm and padded_multimodal_cost:
                new_max = max(bin_maxes[j], length)
                new_cost = new_max * (len(bin_) + 1)
            else:
                new_cost = bin_sums[j] + length
            if new_cost <= max_tokens_per_bin:
                bin_.append(idx)
                bin_sums[j] += length
                bin_maxes[j] = max(bin_maxes[j], length)
                break
        else:
            bins.append([idx])
            bin_is_multimodal.append(is_mm)
            bin_sums.append(length)
            bin_maxes.append(length)

    return bins


def _group_samples_by_rollout(rollout_indices: list[int]) -> tuple[list[int], dict[int, list[int]]]:
    rollout_id_to_samples: dict[int, list[int]] = {}
    for sample_pos, rid in enumerate(rollout_indices):
        rollout_id_to_samples.setdefault(rid, []).append(sample_pos)
    return list(rollout_id_to_samples.keys()), rollout_id_to_samples


def _fixed_rollout_steps(rollout_ids: list[int], global_batch_size: int) -> list[list[int]]:
    num_steps = len(rollout_ids) // global_batch_size
    assert num_steps >= 1, (
        f"num_rollouts ({len(rollout_ids)}) < global_batch_size ({global_batch_size}); "
        f"need at least one rollout per step."
    )
    return [rollout_ids[i * global_batch_size : (i + 1) * global_batch_size] for i in range(num_steps)]


def _token_budget_rollout_steps(
    rollout_ids: list[int],
    rollout_id_to_samples: dict[int, list[int]],
    total_lengths: list[int],
    target_tokens: int,
    dp_size: int,
) -> list[list[int]]:
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")

    steps: list[list[int]] = []
    start = 0
    while start < len(rollout_ids):
        chosen = len(rollout_ids)
        prev = None
        c = start + 1
        while c <= len(rollout_ids):
            step_rollouts = rollout_ids[start:c]
            sample_count = sum(len(rollout_id_to_samples[rid]) for rid in step_rollouts)
            token_count = sum(
                total_lengths[i]
                for rid in step_rollouts
                for i in rollout_id_to_samples[rid]
            )
            dp_aligned = sample_count >= dp_size and sample_count % dp_size == 0
            if dp_aligned and token_count >= target_tokens:
                if prev is not None:
                    prev_rollouts = rollout_ids[start:prev]
                    prev_tokens = sum(
                        total_lengths[i]
                        for rid in prev_rollouts
                        for i in rollout_id_to_samples[rid]
                    )
                    under_gap = target_tokens - prev_tokens
                    over_gap = token_count - target_tokens
                    chosen = prev if under_gap < over_gap else c
                else:
                    chosen = c
                break
            if dp_aligned:
                prev = c
            c += 1

        step_rollouts = rollout_ids[start:chosen]
        sample_count = sum(len(rollout_id_to_samples[rid]) for rid in step_rollouts)
        if sample_count < dp_size or sample_count % dp_size != 0:
            break
        steps.append(step_rollouts)
        start = chosen

    return steps


def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    rollout_indices: list[int],
    sample_is_multimodal: list[bool] | None = None,
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    """Compute the per-rank DP partition and micro-batch schedule.

    See module docstring for the pack-first-distribute-second strategy.

    Args:
        args: Namespace with ``micro_batch_size``, ``use_dynamic_batch_size``,
            ``max_tokens_per_gpu``, ``balance_data``.
        train_parallel_config: ``{"dp_size", "cp_size", "vpp_size",
            "microbatch_group_size_per_vp_stage"}``.
        total_lengths: token count per sample, indexed globally.
        global_batch_size: number of rollouts (NOT training samples) per
            training step. Number of training steps =
            ``num_rollouts // global_batch_size``; trailing rollouts whose
            samples don't fit are dropped.
        rollout_indices: rollout id for each sample (``samples[i].index``).
            Samples sharing the same id are kept together in one step.
        sample_is_multimodal: optional boolean per sample. When
            ``args.multimodal_aware_packing == "separate"``, dynamic packing
            avoids mixing multimodal and text-only samples and estimates
            multimodal mbs cost by padded unsplit size.

    Returns:
        ``(partitions, micro_batch_indices, num_microbatches, global_batch_sizes)``.
        ``global_batch_sizes[s]`` = rollout count for step s (constant
        ``global_batch_size`` for every step).
    """
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]
    if sample_is_multimodal is not None:
        assert len(sample_is_multimodal) == len(total_lengths), (
            f"sample_is_multimodal length {len(sample_is_multimodal)} does not match "
            f"total_lengths length {len(total_lengths)}"
        )

    max_per_bin = None
    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_tokens = args.max_tokens_per_gpu * cp_size
        packing_safety_margin = getattr(args, "packing_safety_margin", 1.0)
        max_per_bin = max(1, int(max_tokens * packing_safety_margin))

    # mbs count per step must be divisible by (dp_size * mb_group_for_vpp) so
    # every rank ends up with the same num_mbs and (for VPP) the per-rank mbs
    # count is a multiple of mb_group.
    align_to = dp_size * (mb_group if vpp_size > 1 else 1)

    # Group samples by rollout id (preserve first-occurrence order). All
    # samples from one rollout stay in a single step so the per-rollout loss
    # reducer is well-defined.
    rollout_ids, rollout_id_to_samples = _group_samples_by_rollout(rollout_indices)
    if getattr(args, "global_batch_tokens", None) is not None:
        step_rollout_groups = _token_budget_rollout_steps(
            rollout_ids,
            rollout_id_to_samples,
            total_lengths,
            args.global_batch_tokens,
            dp_size,
        )
        assert step_rollout_groups, (
            f"num_rollouts ({len(rollout_ids)}) cannot form a token-based global batch "
            f"with dp_size {dp_size} and global_batch_tokens {args.global_batch_tokens}"
        )
    else:
        step_rollout_groups = _fixed_rollout_steps(rollout_ids, global_batch_size)

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []
    global_batch_sizes: list[int] = []

    for step_i, step_rollouts in enumerate(step_rollout_groups):
        sample_indices = [pos for rid in step_rollouts for pos in rollout_id_to_samples[rid]]
        step_lengths = [total_lengths[i] for i in sample_indices]
        step_is_multimodal = [sample_is_multimodal[i] for i in sample_indices] if sample_is_multimodal is not None else None
        global_batch_sizes.append(len(step_rollouts))
        assert len(sample_indices) >= dp_size, (
            f"step {step_i}: {len(sample_indices)} samples < dp_size {dp_size}; "
            f"each step needs at least one sample per rank."
        )

        # 1. Pack samples in this step into mbs with one global pass.
        # ``step_mbs`` indices are LOCAL into ``sample_indices``.
        step_mbs = _pack_step_into_mbs(
            step_lengths,
            use_dynamic_batch_size=args.use_dynamic_batch_size,
            max_per_bin=max_per_bin,
            micro_batch_size=getattr(args, "micro_batch_size", None),
            multimodal_aware_packing=getattr(args, "multimodal_aware_packing", "off"),
            step_is_multimodal=step_is_multimodal,
        )

        # 2. Align mbs count to a multiple of ``align_to``.
        target_K = max(((len(step_mbs) + align_to - 1) // align_to) * align_to, align_to)
        if target_K != len(step_mbs):
            if args.use_dynamic_batch_size:
                expand_bins_by_splitting(step_mbs, target_K, step_lengths)
                assert len(step_mbs) == target_K, (
                    f"dynamic path: could only produce {len(step_mbs)} mbs after maximal splitting; "
                    f"need {target_K}. step {step_i} has {len(sample_indices)} samples, below the "
                    f"alignment threshold ({align_to})."
                )
            else:
                raise AssertionError(
                    f"static path: num_mbs ({len(step_mbs)}) is not a multiple of "
                    f"dp_size * mb_group ({align_to}); got "
                    f"step_size={len(sample_indices)}, micro_batch_size={args.micro_batch_size}, "
                    f"dp_size={dp_size}, mb_group={mb_group if vpp_size > 1 else 1}. "
                    f"Splitting static mbs would break the fixed-size invariant; adjust the config "
                    f"so step_size % (dp_size * micro_batch_size * mb_group) == 0."
                )

        K = len(step_mbs)
        num_mbs_per_rank = K // dp_size
        num_microbatches.append(num_mbs_per_rank)

        # 3. Distribute mbs across ranks: KK on mbs token sums when balance_data is on,
        # otherwise a strided round-robin. Both produce ``num_mbs_per_rank`` mbs per
        # rank (equal_size=True is what KK needs for PP to stay synced).
        if args.balance_data:
            mbs_token_sums = [sum(step_lengths[i] for i in bin_) for bin_ in step_mbs]
            rank_mbs_idx = get_seqlen_balanced_partitions(mbs_token_sums, dp_size, equal_size=True)
        else:
            rank_mbs_idx = [list(range(r, K, dp_size)) for r in range(dp_size)]

        # 4. Build per-rank partitions (global sample indices) and micro_batch_indices
        # (local indices into partitions[r]).
        for r in range(dp_size):
            for mbs_idx in rank_mbs_idx[r]:
                mbs_locals = step_mbs[mbs_idx]  # local indices into sample_indices
                local_start = len(partitions[r])
                partitions[r].extend(sample_indices[i] for i in mbs_locals)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_locals))))

    return partitions, micro_batch_indices, num_microbatches, global_batch_sizes
