"""Multi-process distributed tests for the train metric report helpers.

Spawn ``dp_size * cp_size`` workers with real ``torch.distributed`` (gloo
backend) and exercise the actual production helpers end-to-end. The
single-process numerical contracts live in ``test_metric_report.py``; this
file pins the cross-rank plumbing.

Mapping to the production train_one_step / log_rollout_data flows:

  - ``_distributed_metric_worker`` mirrors both production paths:
        per-rank reducer → ``reduce_train_step_metrics``
        (which calls ``dist.all_reduce`` over the dp-with-cp group and
        applies the cp_size cancellation for the per-token-loss path).
        per-rank reducer → ``rollout_log_metric_contribution`` →
        ``gather_and_reduce_log_dict`` (which calls ``dist.gather_object``
        and applies per-key reductions).

ALL (dp, cp) configurations must give the same reported number — that's
the contract a user touches when they flip any parallelism dial.
"""

from __future__ import annotations

import pickle

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

NUM_GPUS = 0


def _distributed_metric_worker(
    rank: int,
    world_size: int,
    cp_size: int,
    dp_size: int,
    master_port: int,
    result_path: str,
) -> None:
    """Run all three distributed report paths for one DP/CP layout."""
    import torch.distributed as _dist

    cp_rank = rank % cp_size
    dp_rank = rank // cp_size
    stub_megatron_in_worker(cp_size, cp_rank)

    dp_with_cp_group = init_worker_process_group(rank, world_size, master_port)
    try:
        # Import AFTER the megatron stub override so cp_utils still binds
        # against the pre-installed stub (which we've now pinned for this
        # worker's CP rank).
        from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
        from slime.observability.train_metric_utils import (
            gather_and_reduce_log_dict,
            reduce_train_step_metrics,
            rollout_log_metric_contribution,
        )

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

        reports = {}
        for per_token_loss, name in ((False, "per_rollout_mean"), (True, "per_token_loss")):
            if per_token_loss:
                reducer = get_sum_of_sample_mean(my_tl, my_rl, my_masks, calculate_per_token_loss=True)
                # num_tokens is computed off the full mask (not the CP chunk)
                # in loss.py, so reduce_train_step_metrics cancels by cp_size.
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
                reports[name] = reduced["metric"]

        # Run the rollout logging reductions on the same distributed group.
        reducer = get_sum_of_sample_mean(my_tl, my_rl, my_masks, my_denoms)
        per_rank_reducer_sum = reducer(x_for_rank).item()
        log_dict = {
            "logp_per_rollout": rollout_log_metric_contribution(
                per_rank_reducer_sum,
                cp_size=cp_size,
                num_rollouts_in_rollout=4,
                dp_size=dp_size,
            ),
            "total_lengths_per_sample": (float(sum(my_tl)), float(len(my_tl))),
            "rank_local_mean": float(sum(my_tl)) / len(my_tl),
        }
        reduced = gather_and_reduce_log_dict(log_dict, dp_size=world_size, dp_src_rank=0, dp_group=dp_with_cp_group)
        if rank == 0:
            reports["rollout_log"] = reduced
            with open(result_path, "wb") as f:
                pickle.dump(reports, f)
    finally:
        _dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.parametrize("dp_size,cp_size", [(dp, cp) for dp in [1, 2, 4] for cp in [1, 2, 4]])
def test_distributed_reports_match_all_dp_cp_combinations(dp_size, cp_size, tmp_path):
    """Check train and rollout reports on one real Gloo group per layout."""
    import torch.multiprocessing as mp

    world_size = dp_size * cp_size
    result_path = str(tmp_path / "reports.pkl")
    mp.spawn(
        _distributed_metric_worker,
        args=(world_size, cp_size, dp_size, free_port(), result_path),
        nprocs=world_size,
        join=True,
    )
    with open(result_path, "rb") as f:
        reports = pickle.load(f)

    # Both train-step report formulas produce 1249.875 for this fixture.
    assert reports["per_rollout_mean"] == pytest.approx(FOUR_ROLLOUT_EXPECTED_REPORT)
    assert reports["per_token_loss"] == pytest.approx(FOUR_ROLLOUT_EXPECTED_REPORT)
    reduced = reports["rollout_log"]
    # The rollout per-rollout mean matches the train-step report.
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
