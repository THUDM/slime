"""Per-rollout DP/microbatch scheduling.

Pure-Python logic that turns collected rollout samples into per-DP-rank training
shards plus the micro-batch schedule each rank will follow. Lives outside the
ray/sglang-importing modules so it can be unit-tested under CPU-only CI.

Invariants guaranteed by :func:`build_dp_rollout_data` (asserted by the tests):
  - every DP rank holds the same number of samples (``num_samples // dp_size``);
  - every DP rank runs the same ``num_microbatches`` per training step
    (required for PP sync);
  - every mbs holds ``<= max_tokens_per_gpu * cp_size`` tokens, with one exception
    — an individual sample larger than that cap lands alone in its own mbs (and
    that mbs is the only one allowed to exceed the cap);
  - the union of per-rank sample indices equals ``range(num_samples)`` and the
    per-rank index sets are disjoint;
  - flattening ``micro_batch_indices`` for a rank yields ``range(num_samples_rank)``.
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils.data import first_fit_pack
from slime.utils.seqlen_balancing import expand_bins_by_splitting, get_seqlen_balanced_partitions

logger = logging.getLogger(__name__)


# Sample-aligned fields: split per-rank using ``partition`` (rank-r's sample indices).
_SAMPLE_KEYS = (
    "tokens",
    "multimodal_train_inputs",
    "response_lengths",
    "rewards",
    "truncated",
    "loss_masks",
    "round_number",
    "sample_indices",
    "rollout_log_probs",
    "rollout_routed_experts",
    "prompt",
    "teacher_log_probs",
)
# Whole-batch fields: passed through unsliced; the training side picks its own slice.
_PASSTHROUGH_KEYS = ("raw_reward", "total_lengths")


def compute_dynamic_global_batch_size(num_samples: int, dp_size: int) -> int:
    """Round ``num_samples`` down to the nearest multiple of ``dp_size`` (min ``dp_size``).

    Used when ``args.use_dynamic_global_batch_size`` is set, so each rollout produces
    exactly one training step regardless of how many samples were collected.
    """
    dynamic_gbs = (num_samples // dp_size) * dp_size
    if dynamic_gbs == 0:
        return dp_size
    return dynamic_gbs


def build_dp_rollout_data(
    args: Any,
    train_parallel_config: dict,
    data: dict,
    *,
    dynamic_global_batch_size: int | None = None,
) -> list[dict]:
    """Return one ``rollout_data`` dict per DP rank.

    Pipeline (also see module docstring for invariants):
      1. Resolve the effective ``global_batch_size`` (the ``dynamic_global_batch_size``
         arg overrides ``args.global_batch_size`` when provided).
      2. Trim ``data`` to a multiple of that, unless ``args.disable_rollout_trim_samples``.
      3. For each training step (chunk of ``global_batch_size`` samples):
         a. Split samples to DP ranks with equal counts (``balance_data`` => token-
            balanced via Karmarkar-Karp, otherwise strided).
         b. Static path: chunk each rank's samples into mbs of ``args.micro_batch_size``.
            Dynamic path: per-rank first-fit (``<= max_tokens_per_gpu * cp_size``), take
            ``MAX`` across ranks for PP/VPP alignment, then expand each rank to that
            count by splitting its largest multi-sample bins. Split halves are ``<=``
            their parent, so the cap is preserved.
      4. Materialize each rank's sample list in mbs order so each mbs occupies a
         contiguous range of local positions, then build ``micro_batch_indices``
         pointing at those ranges.

    Mutates ``data`` in place: trims sample-aligned lists when needed and writes
    ``data["total_lengths"]``.

    Args:
        args: Namespace with ``global_batch_size``, ``micro_batch_size``,
            ``use_dynamic_batch_size``, ``max_tokens_per_gpu``, ``balance_data``,
            ``disable_rollout_trim_samples``.
        train_parallel_config: ``{"dp_size", "cp_size", "vpp_size",
            "microbatch_group_size_per_vp_stage"}``.
        data: Rollout dict with at least ``"tokens"`` (list of token id sequences);
            other ``_SAMPLE_KEYS`` are sliced per rank when present.
        dynamic_global_batch_size: precomputed dynamic gbs to use; if ``None``,
            ``args.global_batch_size`` is used. Caller should compute this via
            :func:`compute_dynamic_global_batch_size` to keep that decision local.

    Returns:
        List of ``dp_size`` dicts, each ready to be ``ray.put`` and consumed by the
        training side. Each dict carries: ``partition`` (global sample indices for
        that rank), the sliced ``_SAMPLE_KEYS``, the passthrough ``_PASSTHROUGH_KEYS``,
        ``num_microbatches`` (same list on every rank), ``micro_batch_indices``
        (rank-local), and ``dynamic_global_batch_size`` when applicable.
    """
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    # 1. Resolve effective global_batch_size
    if dynamic_global_batch_size is not None:
        global_batch_size = dynamic_global_batch_size
    else:
        global_batch_size = args.global_batch_size

    # 2. Trim data to a multiple of global_batch_size
    if not args.disable_rollout_trim_samples:
        num_samples = len(data["tokens"])
        trim_len = num_samples // global_batch_size * global_batch_size
        if trim_len == 0:
            raise ValueError(f"Not enough samples {num_samples} for global_batch_size {global_batch_size}")
        if trim_len < num_samples:
            logger.info(f"Trimmed samples from {num_samples} to {trim_len}")
            for key, val in data.items():
                if isinstance(val, list):
                    data[key] = val[:trim_len]

    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths
    num_steps = len(total_lengths) // global_batch_size

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_per_bin = args.max_tokens_per_gpu * cp_size

    # Accumulators indexed by DP rank.
    #   per_rank_sample_indices[r] — global sample indices going to rank r,
    #     concatenated across all steps in mbs-order.
    #   per_rank_mbs_indices[r] — flat list of mbs (across all steps in order),
    #     each mbs being LOCAL indices into per_rank_sample_indices[r].
    #   num_microbatches_per_step[s] — same value on every DP rank (PP sync).
    per_rank_sample_indices: list[list[int]] = [[] for _ in range(dp_size)]
    per_rank_mbs_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches_per_step: list[int] = []

    # 3. Per step: DP split, then per-rank mbs schedule.
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

        num_microbatches_per_step.append(num_mbs_per_rank)

        # 4. Materialize per-rank schedule. Samples are appended in mbs order so each
        # mbs occupies a contiguous range of positions in per_rank_sample_indices[r].
        for r in range(dp_size):
            for mbs_local in rank_mbs[r]:
                local_start = len(per_rank_sample_indices[r])
                per_rank_sample_indices[r].extend(step_start + rank_parts[r][i] for i in mbs_local)
                per_rank_mbs_indices[r].append(list(range(local_start, local_start + len(mbs_local))))

    # 5. Build per-rank rollout_data dicts (caller wraps in Ray Box).
    rollout_data_list: list[dict] = []
    for r in range(dp_size):
        partition = per_rank_sample_indices[r]
        rollout_data: dict = {"partition": partition}
        for key in _SAMPLE_KEYS:
            if key in data:
                rollout_data[key] = [data[key][j] for j in partition]
        for key in _PASSTHROUGH_KEYS:
            if key in data:
                rollout_data[key] = data[key]
        if dynamic_global_batch_size is not None:
            rollout_data["dynamic_global_batch_size"] = dynamic_global_batch_size
        rollout_data["num_microbatches"] = num_microbatches_per_step
        rollout_data["micro_batch_indices"] = per_rank_mbs_indices[r]
        rollout_data_list.append(rollout_data)
    return rollout_data_list
