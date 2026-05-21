"""Per-rollout DP/microbatch scheduling.

Pure-Python logic that decides, for one rollout's worth of sample lengths,
how to group samples into micro-batches and which DP rank owns each mbs.
Lives outside the ray/sglang-importing modules so it can be unit-tested
under CPU-only CI.

The scheduling philosophy is **pack first, distribute second**:

  1. The caller provides ``step_sample_indices`` — one list of global sample
     indices per training step. The default split is even chunks of
     ``global_batch_size``; users can plug a custom splitter (e.g. uneven
     7/8/9 batches from 24 samples) via ``--custom-rollout-step-split-path``.
  2. For each step, pack its samples into ``K`` micro-batches with a single
     first-fit pass (dynamic batch) or fixed-size chunking (static batch).
  3. Adjust ``K`` to a multiple of ``dp_size * (mb_group if vpp>1 else 1)``
     by splitting the largest multi-sample bins (dynamic only).
  4. Distribute the ``K`` mbs across ``dp_size`` ranks, ``K / dp_size``
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
  - Per-step sample counts can also vary across steps (e.g. 7/8/9 instead
    of 8/8/8) — the per-step ``global_batch_sizes`` plumbing handles that
    on the train side.

Invariants guaranteed by :func:`build_dp_schedule` (asserted by the tests):
  - every DP rank runs the **same** ``num_microbatches`` per training step
    (required for PP sync);
  - every mbs (dynamic path) holds ``<= max_tokens_per_gpu * cp_size``
    tokens, with one exception — an individual sample larger than that cap
    lands alone in its own mbs (and that mbs is the only one allowed to
    exceed the cap);
  - the union of per-rank sample indices equals the union of the input
    ``step_sample_indices`` (every assigned sample placed exactly once);
  - flattening ``micro_batch_indices`` for a rank yields
    ``range(num_samples_rank)`` (each rank's samples are tiled exactly once
    by its mbs schedule).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from slime.utils.seqlen_balancing import expand_bins_by_splitting, first_fit_pack, get_seqlen_balanced_partitions

logger = logging.getLogger(__name__)


def near_equal_step_split(num_samples: int, *, num_steps: int) -> list[list[int]]:
    """Split ``num_samples`` contiguous samples into ``num_steps`` near-equal chunks.

    Chunks differ in size by at most 1: the first ``num_samples % num_steps``
    chunks get ``ceil(num_samples / num_steps)`` samples each, the rest get
    ``floor(num_samples / num_steps)``. So 24 samples / 3 steps → ``[8, 8, 8]``;
    23 samples / 3 steps → ``[8, 8, 7]``; 25 samples / 3 steps → ``[9, 8, 8]``.
    """
    assert num_steps >= 1, f"num_steps must be >= 1, got {num_steps}"
    assert (
        num_samples >= num_steps
    ), f"num_samples ({num_samples}) < num_steps ({num_steps}); cannot place at least one sample per step."
    base = num_samples // num_steps
    extra = num_samples % num_steps
    out: list[list[int]] = []
    start = 0
    for i in range(num_steps):
        size = base + (1 if i < extra else 0)
        out.append(list(range(start, start + size)))
        start += size
    return out


def even_step_split(num_samples: int, global_batch_size: int) -> list[list[int]]:
    """Default step split: contiguous chunks of ``global_batch_size`` samples.

    Returns ``num_steps = num_samples // global_batch_size`` lists, each holding
    ``global_batch_size`` sample indices. Trailing samples that don't fill a
    complete step are dropped by the caller before this function is invoked.
    """
    num_steps = num_samples // global_batch_size
    return [list(range(s * global_batch_size, (s + 1) * global_batch_size)) for s in range(num_steps)]


def validate_step_sample_indices(
    step_sample_indices: list[list[int]],
    num_samples: int,
) -> list[list[int]]:
    """Validate (and normalise to ``list[list[int]]``) the output of a custom
    step splitter.

    Checks invariants the rest of the schedule relies on:
      * non-empty list of steps;
      * every sample index is in ``range(num_samples)``;
      * no index appears in more than one step.

    Returns the (possibly newly-listified) splits; logs a warning if the
    splitter dropped any samples (this is allowed but unusual).
    """
    assert step_sample_indices, "custom_rollout_step_split returned an empty list of steps"
    normalised: list[list[int]] = []
    seen: set[int] = set()
    for s, indices in enumerate(step_sample_indices):
        indices = list(indices)
        normalised.append(indices)
        dup = set(indices) & seen
        assert not dup, f"custom_rollout_step_split: step {s} reuses sample indices already placed: {sorted(dup)}"
        seen.update(indices)
    extra = seen - set(range(num_samples))
    assert not extra, f"custom_rollout_step_split returned out-of-range sample indices: {sorted(extra)}"
    missing = set(range(num_samples)) - seen
    if missing:
        logger.warning(
            "custom_rollout_step_split dropped %d sample(s); first few: %s",
            len(missing),
            sorted(missing)[:5],
        )
    return normalised


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
    step_sample_indices: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    """Compute the per-rank DP partition and micro-batch schedule.

    See module docstring for the pack-first-distribute-second strategy.

    Args:
        args: Namespace with ``micro_batch_size``, ``use_dynamic_batch_size``,
            ``max_tokens_per_gpu``, ``balance_data``.
        train_parallel_config: ``{"dp_size", "cp_size", "vpp_size",
            "microbatch_group_size_per_vp_stage"}``.
        total_lengths: token count per sample, indexed globally.
        step_sample_indices: per-step lists of indices into ``total_lengths``.
            Each step's sample count becomes that step's ``global_batch_size``.
            Step sample counts may vary across steps.

    Returns:
        ``(partitions, micro_batch_indices, num_microbatches, global_batch_sizes)``.
    """
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

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

    for step_i, sample_indices in enumerate(step_sample_indices):
        step_lengths = [total_lengths[i] for i in sample_indices]
        global_batch_sizes.append(len(sample_indices))
        assert len(sample_indices) >= dp_size, (
            f"step {step_i}: {len(sample_indices)} samples < dp_size {dp_size}; "
            f"each step needs at least one sample per rank."
        )

        # 1. Pack samples in this step into mbs with one global pass.
        # ``step_mbs`` indices are LOCAL into ``step_lengths`` / ``sample_indices``.
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
