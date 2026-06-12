import logging
import math
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
    group_ids: list | None = None,
):
    """Estimate pass@k per prompt-group and average across groups.

    Two regimes:

    * **Rigid** (``group_ids is None``): assume every group has exactly
      ``group_size`` samples laid out contiguously, i.e.
      ``len(flat_rewards) == num_groups * group_size``. This is the legacy
      fixed-size shape and is numerically identical to the prior reshape.

    * **Ragged** (``group_ids`` given): over-sampled batches do *not* keep a
      rigid ``num_groups * group_size`` layout — dynamic sampling, group
      replacement, and per-group sample drops yield a variable number of
      samples per prompt-group (and a total that need not be a multiple of
      ``group_size``). We then bucket ``flat_rewards`` by their actual group
      id and estimate pass@k over the samples that actually exist for each
      group, so the metric never asserts on a ragged batch.

    ``num_groups`` is only used by the rigid path (to validate the reshape);
    it is ignored when ``group_ids`` is provided, since the ragged path derives
    the group count from the distinct ids in ``group_ids``.

    ``group_size`` only sets which ``pass@{1,2,4,...}`` rungs are reported
    (the ladder ``[2**i for i in range(log2(group_size)+1)]``). In the ragged
    regime, pass@k for a rung is averaged only over groups that have at least
    ``k`` samples — a group with fewer than ``k`` samples cannot define an
    unbiased pass@k draw, so it is excluded from that rung's mean rather than
    counted as a trivial 1.0. Rungs whose every group is too small are dropped.
    """
    if group_size == 1:
        return {}

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    if group_ids is None:
        if num_groups is None:
            num_groups = len(flat_rewards) // group_size
        assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
        rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)
        num_samples_per_group = np.full(num_groups, group_size)
        num_correct_per_group = np.sum(rewards_of_group == 1, axis=1)
    else:
        # Ragged layout: bucket rewards by their actual group id. Group order
        # does not matter — the final metric is an order-independent mean.
        assert len(flat_rewards) == len(group_ids), f"{len(flat_rewards)=} {len(group_ids)=}"
        grouped: dict = {}
        for reward, gid in zip(flat_rewards, group_ids, strict=True):
            grouped.setdefault(gid, []).append(reward)
        group_rewards = list(grouped.values())
        num_samples_per_group = np.array([len(g) for g in group_rewards])
        num_correct_per_group = np.array([sum(1 for r in g if r == 1) for g in group_rewards])

    log_dict = {}
    for k in pass_rate_name_list:
        # A group must have >= k samples to define an unbiased pass@k draw.
        eligible = num_samples_per_group >= k
        if not np.any(eligible):
            continue
        pass_k_estimates = _estimate_pass_at_k(num_samples_per_group[eligible], num_correct_per_group[eligible], k)
        log_dict[f"pass@{k}"] = np.mean(pass_k_estimates)

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
