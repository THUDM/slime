"""Per-rollout DP/microbatch scheduling.

Pure-Python logic that decides, for one rollout's worth of sample lengths,
how to group samples into micro-batches and which DP rank owns each mbs.
Lives outside the ray/sglang-importing modules so it can be unit-tested
under CPU-only CI.

The scheduling philosophy is **pack first, distribute second**:

  1. Pack the step's samples into ``K`` micro-batches with a single global
     first-fit pass (dynamic batch) or fixed-size chunking (static batch).
  2. Adjust ``K`` to a multiple of ``dp_size * (mb_group if vpp>1 else 1)``
     by splitting the largest multi-sample bins (dynamic only).
  3. Distribute the ``K`` mbs across ``dp_size`` ranks, ``K / dp_size``
     each, with either a strided round-robin or a Karmarkar-Karp pass on
     mbs token sums.

This is cleaner than the previous "split samples → pack per rank → pad each
rank to the global max mbs count" approach because:

  - PP only needs ``num_mbs`` to be equal across DP ranks; it does **not**
    need each rank to hold the same number of samples. The pack-first
    scheme satisfies that trivially.
  - At most one mbs can be partially filled (the last one). The old scheme
    leaked padding into N mbs (one per rank).
  - The number of samples per rank is allowed to vary, so unbalanced
    rollouts (``num_samples`` not divisible by ``dp_size``) no longer
    require trimming. The train side already normalises by per-step
    ``global_batch_sizes`` and per-rank ``(sum, count)`` tuples; rank-level
    sample asymmetry is therefore mathematically harmless.

Invariants guaranteed by :func:`build_dp_schedule` (asserted by the tests):
  - every DP rank runs the **same** ``num_microbatches`` per training step
    (required for PP sync);
  - every mbs (dynamic path) holds ``<= max_tokens_per_gpu * cp_size``
    tokens, with one exception — an individual sample larger than that cap
    lands alone in its own mbs (and that mbs is the only one allowed to
    exceed the cap);
  - the union of per-rank sample indices equals ``range(num_samples)`` and
    the per-rank index sets are disjoint;
  - flattening ``micro_batch_indices`` for a rank yields
    ``range(num_samples_rank)`` (each rank's samples are tiled exactly once
    by its mbs schedule).
"""

from __future__ import annotations

from typing import Any

from slime.utils.seqlen_balancing import expand_bins_by_splitting, first_fit_pack, get_seqlen_balanced_partitions


def compute_dynamic_global_batch_size(num_samples: int, dp_size: int) -> int:
    """Return ``num_samples`` as the per-step gbs when dynamic-gbs is enabled.

    With the pack-first-distribute-later schedule, ``num_samples`` no longer
    needs to be a multiple of ``dp_size`` — rank-level sample asymmetry is
    handled by per-rank ``(sum, count)`` log aggregation and per-step
    ``global_batch_size`` loss scaling. We still clamp to ``>= dp_size`` so
    the schedule has at least one mbs per rank to place.
    """
    if num_samples < dp_size:
        # Edge case: not enough samples to place one per rank. The caller
        # is expected to dummy-pad to dp_size in this regime; the schedule
        # itself can't fabricate samples. We round up so the math agrees
        # with the padded count.
        return dp_size
    return num_samples


def _pack_step_into_mbs(
    step_lengths: list[int],
    *,
    use_dynamic_batch_size: bool,
    max_per_bin: int | None,
    micro_batch_size: int | None,
) -> list[list[int]]:
    """Group a step's samples into mbs. Returns ``mbs[k]`` = local indices into ``step_lengths``."""
    if use_dynamic_batch_size:
        assert max_per_bin is not None
        return first_fit_pack(step_lengths, max_per_bin)
    assert micro_batch_size is not None
    n = len(step_lengths)
    return [list(range(i, min(i + micro_batch_size, n))) for i in range(0, n, micro_batch_size)]


def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    """Compute the per-rank DP partition and micro-batch schedule.

    See module docstring for the pack-first-distribute-second strategy.
    Returns ``(partitions, micro_batch_indices, num_microbatches, global_batch_sizes)``.
    """
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    num_steps = len(total_lengths) // global_batch_size

    max_per_bin = None
    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_per_bin = args.max_tokens_per_gpu * cp_size

    # mbs count per step must be divisible by (dp_size * mb_group_for_vpp) so
    # every rank ends up with the same num_mbs and (for VPP) the per-rank mbs
    # count is a multiple of mb_group.
    align_to = dp_size * (mb_group if vpp_size > 1 else 1)

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []
    global_batch_sizes: list[int] = []

    for step_i in range(num_steps):
        step_start = step_i * global_batch_size
        step_lengths = total_lengths[step_start : step_start + global_batch_size]
        global_batch_sizes.append(len(step_lengths))

        # 1. Pack samples in this step into mbs with one global pass.
        step_mbs = _pack_step_into_mbs(
            step_lengths,
            use_dynamic_batch_size=args.use_dynamic_batch_size,
            max_per_bin=max_per_bin,
            micro_batch_size=getattr(args, "micro_batch_size", None),
        )

        # 2. Align mbs count to a multiple of ``align_to``.
        target_K = max(((len(step_mbs) + align_to - 1) // align_to) * align_to, align_to)
        if target_K != len(step_mbs):
            if args.use_dynamic_batch_size:
                expand_bins_by_splitting(step_mbs, target_K, step_lengths)
                assert len(step_mbs) == target_K, (
                    f"dynamic path: could only produce {len(step_mbs)} mbs after maximal splitting; "
                    f"need {target_K}. This happens when num_samples ({len(step_lengths)}) is below the "
                    f"alignment threshold ({align_to}); pad the rollout with dummy samples before calling."
                )
            else:
                raise AssertionError(
                    f"static path: num_mbs ({len(step_mbs)}) is not a multiple of "
                    f"dp_size * mb_group ({align_to}); got "
                    f"global_batch_size={global_batch_size}, micro_batch_size={args.micro_batch_size}, "
                    f"dp_size={dp_size}, mb_group={mb_group if vpp_size > 1 else 1}. "
                    f"Splitting static mbs would break the fixed-size invariant; adjust the config "
                    f"so global_batch_size % (dp_size * micro_batch_size * mb_group) == 0."
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
                mbs_locals = step_mbs[mbs_idx]
                local_start = len(partitions[r])
                partitions[r].extend(step_start + i for i in mbs_locals)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_locals))))

    return partitions, micro_batch_indices, num_microbatches, global_batch_sizes
