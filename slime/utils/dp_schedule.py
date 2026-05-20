"""Per-rollout DP/microbatch scheduling.

Pure-Python logic that decides, for one (already-trimmed) rollout's worth of
sample lengths, which global sample goes to which DP rank and how each rank
groups its samples into micro-batches. Lives outside the ray/sglang-importing
modules so it can be unit-tested under CPU-only CI.

Trim, dynamic-global_batch_size resolution, and per-rank rollout_data
packaging all stay on the caller's side; this module only computes the
schedule itself.

Invariants guaranteed by :func:`build_dp_schedule` (asserted by the tests):
  - every DP rank holds the same number of samples (``num_samples // dp_size``);
  - every DP rank runs the same ``num_microbatches`` per training step
    (required for PP sync);
  - every mbs holds ``<= max_tokens_per_gpu * cp_size`` tokens, with one
    exception — an individual sample larger than that cap lands alone in its
    own mbs (and that mbs is the only one allowed to exceed the cap);
  - the union of per-rank sample indices equals ``range(num_samples)`` and
    the per-rank index sets are disjoint;
  - flattening ``micro_batch_indices`` for a rank yields
    ``range(num_samples_rank)``.
"""

from __future__ import annotations

from typing import Any

from slime.utils.seqlen_balancing import expand_bins_by_splitting, first_fit_pack, get_seqlen_balanced_partitions


def compute_dynamic_global_batch_size(num_samples: int, dp_size: int) -> int:
    """Round ``num_samples`` down to the nearest multiple of ``dp_size`` (min ``dp_size``).

    Used when ``args.use_dynamic_global_batch_size`` is set, so each rollout produces
    exactly one training step regardless of how many samples were collected.
    """
    dynamic_gbs = (num_samples // dp_size) * dp_size
    if dynamic_gbs == 0:
        return dp_size
    return dynamic_gbs


def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
) -> tuple[list[list[int]], list[list[list[int]]], list[int]]:
    """Compute the per-rank DP partition and micro-batch schedule.

    For each training step (chunk of ``global_batch_size`` samples):
      a. Split samples to DP ranks with equal counts (``balance_data`` => token-
         balanced via Karmarkar-Karp, otherwise strided).
      b. Static path: chunk each rank's samples into mbs of ``args.micro_batch_size``.
         Dynamic path: per-rank first-fit (``<= max_tokens_per_gpu * cp_size``), take
         ``MAX`` across ranks for PP/VPP alignment, then expand each rank to that
         count by splitting its largest multi-sample bins. Split halves are ``<=``
         their parent, so the cap is preserved.

    Samples are appended to ``partitions[r]`` in mbs order so each mbs occupies a
    contiguous range of positions there; ``micro_batch_indices[r][k]`` is that range.

    Args:
        args: Namespace with ``micro_batch_size``, ``use_dynamic_batch_size``,
            ``max_tokens_per_gpu``, ``balance_data``.
        train_parallel_config: ``{"dp_size", "cp_size", "vpp_size",
            "microbatch_group_size_per_vp_stage"}``.
        total_lengths: token count per sample, length must be a multiple of
            ``global_batch_size``.
        global_batch_size: samples per training step.

    Returns:
        ``(partitions, micro_batch_indices, num_microbatches)``:
          - ``partitions[r]`` — global sample indices going to rank r, concatenated
            across all steps in mbs order.
          - ``micro_batch_indices[r][k]`` — local indices into ``partitions[r]`` for
            the k-th mbs of rank r (flat across all steps).
          - ``num_microbatches[s]`` — mbs count for step s; same value on every rank.
    """
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    num_steps = len(total_lengths) // global_batch_size

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_per_bin = args.max_tokens_per_gpu * cp_size

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []

    for step_i in range(num_steps):
        step_start = step_i * global_batch_size
        step_lengths = total_lengths[step_start : step_start + global_batch_size]

        if args.balance_data:
            rank_parts = get_seqlen_balanced_partitions(step_lengths, dp_size, equal_size=True)
        else:
            rank_parts = [list(range(r, global_batch_size, dp_size)) for r in range(dp_size)]

        # rank_mbs[r][k] is one mbs of LOCAL indices into rank_parts[r] (positions
        # within this rank's sample list, not step- or global-indices).
        if not args.use_dynamic_batch_size:
            mbs = args.micro_batch_size
            n = len(rank_parts[0])  # gbs / dp, same for every rank
            rank_mbs = [[list(range(i, i + mbs)) for i in range(0, n, mbs)] for _ in range(dp_size)]
            num_mbs_per_rank = n // mbs
        else:
            rank_lens = [[step_lengths[i] for i in rank_parts[r]] for r in range(dp_size)]
            rank_mbs = [first_fit_pack(rank_lens[r], max_per_bin) for r in range(dp_size)]
            num_mbs_per_rank = max(len(b) for b in rank_mbs)
            if vpp_size > 1:
                # Match the original floor-to-mb_group rounding (with min=1).
                num_mbs_per_rank = max(num_mbs_per_rank // mb_group * mb_group, 1)
            for r in range(dp_size):
                expand_bins_by_splitting(rank_mbs[r], num_mbs_per_rank, rank_lens[r])

        num_microbatches.append(num_mbs_per_rank)

        for r in range(dp_size):
            for mbs_local in rank_mbs[r]:
                local_start = len(partitions[r])
                partitions[r].extend(step_start + rank_parts[r][i] for i in mbs_local)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_local))))

    return partitions, micro_batch_indices, num_microbatches
