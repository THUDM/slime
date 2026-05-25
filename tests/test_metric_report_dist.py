"""Multi-process distributed tests for the cp_utils report helpers.

Spawn ``dp_size * cp_size`` workers with real ``torch.distributed`` (gloo
backend) and exercise the actual production helpers end-to-end. The
single-process numerical contracts live in ``test_metric_report.py``; this
file pins the cross-rank plumbing.

Mapping to the production train_one_step / log_rollout_data flows:

  - ``_train_step_distributed_worker`` mirrors ``train_one_step``:
        per-rank reducer → ``reduce_train_step_metrics``
        (which calls ``dist.all_reduce`` over the dp-with-cp group and
        applies the cp_size cancellation for the per-token-loss path).
  - ``_rollout_log_distributed_worker`` mirrors ``log_rollout_data``:
        per-rank reducer → ``rollout_log_metric_contribution`` →
        ``gather_and_reduce_log_dict`` (which calls ``dist.gather_object``
        and applies per-key reductions).

ALL (dp, cp) configurations must give the same reported number — that's
the contract a user touches when they flip any parallelism dial.
"""

from __future__ import annotations

# IMPORTANT: import the helpers (and the megatron stub it installs) BEFORE
# any slime import. Spawned workers re-import this module from scratch, so
# the same ordering must hold there — see ``stub_megatron_in_worker``
# for the worker-side details. pytest's prepend importmode puts
# ``tests/`` on sys.path so the bare-name import works without an
# ``__init__.py``; mp.spawn children inherit the parent's sys.path.
import _cp_dist_helpers
import pytest
import torch
from _cp_dist_helpers import (
    FOUR_ROLLOUT_EXPECTED_REPORT,
    FOUR_ROLLOUT_RESPONSE_LENGTHS,
    FOUR_ROLLOUT_TOTAL_LENGTHS,
    FOUR_ROLLOUT_X_VALUES,
    cp_chunk_response_tensor,
    free_port,
    init_worker_process_group,
    stub_megatron_in_worker,
)


def _train_step_distributed_worker(
    rank: int,
    world_size: int,
    cp_size: int,
    dp_size: int,
    per_token_loss: bool,
    master_port: int,
    result_path: str,
) -> None:
    """Per-rank entrypoint for ``mp.spawn``: init gloo pg, run one rank's
    share of the train-step report, write rank-0's result to a file."""
    import torch.distributed as _dist

    cp_rank = rank % cp_size
    dp_rank = rank // cp_size
    stub_megatron_in_worker(cp_size, cp_rank)

    dp_with_cp_group = init_worker_process_group(rank, world_size, master_port)
    try:
        # Import AFTER the megatron stub override so cp_utils still binds
        # against the pre-installed stub (which we've now pinned for this
        # worker's CP rank).
        from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean, reduce_train_step_metrics

        all_total_lengths = FOUR_ROLLOUT_TOTAL_LENGTHS
        all_response_lengths = FOUR_ROLLOUT_RESPONSE_LENGTHS
        all_loss_masks = [torch.ones(r, dtype=torch.float32) for r in all_response_lengths]
        all_x = [torch.tensor(v) for v in FOUR_ROLLOUT_X_VALUES]
        step_global_batch_size = 4  # 4 rollouts in the step

        # Round-robin DP partition: with 4 samples / dp=N, rank i gets
        # samples i, i+N, ... (matches what _split_train_data_by_dp does
        # for evenly divisible cases).
        my_indices = [i for i in range(4) if i % dp_size == dp_rank]
        my_tl = [all_total_lengths[i] for i in my_indices]
        my_rl = [all_response_lengths[i] for i in my_indices]
        my_masks = [all_loss_masks[i] for i in my_indices]
        my_x = [all_x[i] for i in my_indices]
        my_denoms = torch.tensor([float(m.sum().item()) for m in my_masks], dtype=torch.float32)

        if cp_size == 1:
            x_for_rank = torch.cat(my_x)
        else:
            x_for_rank = torch.cat(
                [cp_chunk_response_tensor(x, tl, rl) for tl, rl, x in zip(my_tl, my_rl, my_x, strict=True)]
            )

        if per_token_loss:
            reducer = get_sum_of_sample_mean(my_tl, my_rl, my_masks, calculate_per_token_loss=True)
            # num_tokens is computed off the FULL mask (not the chunked
            # one) in loss.py — every CP rank reports the same number,
            # which is why ``reduce_train_step_metrics`` cancels by
            # ``cp_factor = cp_size`` afterwards.
            num_tokens = sum(int(m.sum().item()) for m in my_masks)
            values_tensor = torch.tensor([float(num_tokens), reducer(x_for_rank).item()], dtype=torch.float32)
        else:
            reducer = get_sum_of_sample_mean(my_tl, my_rl, my_masks, my_denoms)
            values_tensor = torch.tensor([0.0, reducer(x_for_rank).item()], dtype=torch.float32)

        reduced = reduce_train_step_metrics(
            [{"keys": ["metric"], "values": values_tensor}],
            calculate_per_token_loss=per_token_loss,
            step_global_batch_size=step_global_batch_size,
            cp_size=cp_size,
            dp_with_cp_group=dp_with_cp_group,
        )

        if rank == 0:
            with open(result_path, "w") as f:
                f.write(repr(reduced["metric"]))
    finally:
        _dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.parametrize(
    "dp_size,cp_size",
    [(dp, cp) for dp in [1, 2, 4] for cp in [1, 2, 4]],
)
def test_train_step_per_rollout_mean_real_distributed(dp_size, cp_size, tmp_path):
    """End-to-end multi-process: spawn ``dp_size * cp_size`` workers, each
    runs its share with real ``torch.distributed`` (gloo); ALL parallelism
    combinations must give the same reported per-rollout-mean number.

    Expected = sum of per-rollout token-means / step_gbs
             = (4.5 + 45 + 450 + 4500) / 4 = 1249.875
    """
    import torch.multiprocessing as mp

    world_size = dp_size * cp_size
    result_path = str(tmp_path / "result.txt")
    mp.spawn(
        _train_step_distributed_worker,
        args=(world_size, cp_size, dp_size, False, free_port(), result_path),
        nprocs=world_size,
        join=True,
    )
    with open(result_path) as f:
        result = float(f.read())
    assert result == pytest.approx(FOUR_ROLLOUT_EXPECTED_REPORT)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dp_size,cp_size",
    [(dp, cp) for dp in [1, 2, 4] for cp in [1, 2, 4]],
)
def test_train_step_per_token_loss_real_distributed(dp_size, cp_size, tmp_path):
    """Same end-to-end multi-process check for the per-token-loss path.

    Expected = sum of all x / total_tokens
             = (36 + 360 + 3600 + 36000) / 32 = 1249.875
    """
    import torch.multiprocessing as mp

    world_size = dp_size * cp_size
    result_path = str(tmp_path / "result.txt")
    mp.spawn(
        _train_step_distributed_worker,
        args=(world_size, cp_size, dp_size, True, free_port(), result_path),
        nprocs=world_size,
        join=True,
    )
    with open(result_path) as f:
        result = float(f.read())
    assert result == pytest.approx(FOUR_ROLLOUT_EXPECTED_REPORT)


def _rollout_log_distributed_worker(
    rank: int,
    world_size: int,
    cp_size: int,
    dp_size: int,
    master_port: int,
    result_path: str,
) -> None:
    """Per-rank entrypoint for ``mp.spawn``: build a multi-key log_dict
    covering all three reduction modes ``gather_and_reduce_log_dict``
    supports, run real ``dist.gather_object``, have rank 0 dump the
    reduced dict via pickle for the parent to assert on.
    """
    import pickle

    import torch.distributed as _dist

    cp_rank = rank % cp_size
    dp_rank = rank // cp_size
    stub_megatron_in_worker(cp_size, cp_rank)

    dp_group = init_worker_process_group(rank, world_size, master_port)
    try:
        from slime.backends.megatron_utils.cp_utils import (
            gather_and_reduce_log_dict,
            get_sum_of_sample_mean,
            rollout_log_metric_contribution,
        )

        all_total_lengths = FOUR_ROLLOUT_TOTAL_LENGTHS
        all_response_lengths = FOUR_ROLLOUT_RESPONSE_LENGTHS
        all_loss_masks = [torch.ones(r, dtype=torch.float32) for r in all_response_lengths]
        all_x = [torch.tensor(v) for v in FOUR_ROLLOUT_X_VALUES]
        num_rollouts_in_rollout = 4

        my_indices = [i for i in range(4) if i % dp_size == dp_rank]
        my_tl = [all_total_lengths[i] for i in my_indices]
        my_rl = [all_response_lengths[i] for i in my_indices]
        my_masks = [all_loss_masks[i] for i in my_indices]
        my_x = [all_x[i] for i in my_indices]
        my_denoms = torch.tensor([float(m.sum().item()) for m in my_masks], dtype=torch.float32)

        if cp_size == 1:
            x_for_rank = torch.cat(my_x)
        else:
            x_for_rank = torch.cat(
                [cp_chunk_response_tensor(x, tl, rl) for tl, rl, x in zip(my_tl, my_rl, my_x, strict=True)]
            )

        reducer = get_sum_of_sample_mean(my_tl, my_rl, my_masks, my_denoms)
        per_rank_reducer_sum = reducer(x_for_rank).item()

        # Exercise every reduction mode the production log_rollout_data emits.
        log_dict = {
            # per-rollout-mean: (sum, count) via rollout_log_metric_contribution.
            # gather: Σsum / Σcount = sum_DP_full / num_rollouts.
            "logp_per_rollout": rollout_log_metric_contribution(
                per_rank_reducer_sum,
                cp_size=cp_size,
                num_rollouts_in_rollout=num_rollouts_in_rollout,
                dp_size=dp_size,
            ),
            # per-sample-mean: (Σval, num_samples) — matches the
            # ``total_lengths`` style in log_rollout_data. gather: Σsum/Σcount
            # = total / total_samples = per-sample mean of total_lengths.
            "total_lengths_per_sample": (float(sum(my_tl)), float(len(my_tl))),
            # mean-across-ranks: plain scalar — matches log_multi_turn_data
            # style. gather: Σvalue / dp_world.
            "rank_local_mean": float(sum(my_tl)) / len(my_tl),
        }

        reduced = gather_and_reduce_log_dict(log_dict, dp_size=world_size, dp_src_rank=0, dp_group=dp_group)

        if rank == 0:
            with open(result_path, "wb") as f:
                pickle.dump(reduced, f)
    finally:
        _dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.parametrize(
    "dp_size,cp_size",
    [(dp, cp) for dp in [1, 2, 4] for cp in [1, 2, 4]],
)
def test_rollout_log_real_distributed_multi_key(dp_size, cp_size, tmp_path):
    """End-to-end multi-process for ``gather_and_reduce_log_dict``.

    Covers the three key shapes ``log_rollout_data`` produces:
      - per-rollout-mean ((sum, count) via ``rollout_log_metric_contribution``)
      - per-sample-mean ((Σval, num_samples) tuple — e.g. ``total_lengths``)
      - mean-across-ranks (plain float — e.g. multi_turn stats)

    All (dp, cp) configs must yield the same reduced numbers; matches the
    expected values written in pure Python from the fixture. In particular
    the per-rollout-mean number must equal what the train-step report tests
    above land on (FOUR_ROLLOUT_EXPECTED_REPORT), pinning the cross-phase
    contract.
    """
    import pickle

    import torch.multiprocessing as mp

    world_size = dp_size * cp_size
    result_path = str(tmp_path / "result.pkl")
    mp.spawn(
        _rollout_log_distributed_worker,
        args=(world_size, cp_size, dp_size, free_port(), result_path),
        nprocs=world_size,
        join=True,
    )
    with open(result_path, "rb") as f:
        reduced = pickle.load(f)

    # per-rollout-mean: matches the train-side report — 1249.875.
    assert reduced["logp_per_rollout"] == pytest.approx(FOUR_ROLLOUT_EXPECTED_REPORT)
    # per-sample-mean: every sample has total_length=12, so the average is 12.
    assert reduced["total_lengths_per_sample"] == pytest.approx(12.0)
    # mean-across-ranks: every rank's local mean is 12, so cross-rank mean is 12.
    assert reduced["rank_local_mean"] == pytest.approx(12.0)


# Keep an explicit reference to silence "unused import" complaints while
# documenting that importing the helpers module is load-bearing (it
# installs the megatron stub before slime is touched).
_ = _cp_dist_helpers


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
