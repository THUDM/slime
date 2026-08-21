from types import SimpleNamespace

import pytest

from slime.ray.rollout import RolloutManager


NUM_GPUS = 0


def test_rollout_manager_generates_exact_initial_and_aligned_replacement_counts():
    manager = object.__new__(RolloutManager.__ray_actor_class__)
    manager.args = SimpleNamespace(
        ci_test=False,
        use_fault_tolerance=False,
        rollout_batch_size=8,
        n_samples_per_prompt=2,
        rs_refill_max_rounds=2,
    )
    manager.train_parallel_config = {
        "dp_size": 2,
        "vpp_size": 2,
        "microbatch_group_size_per_vp_stage": 4,
    }
    manager._pending_rs_batches = {}
    manager.health_monitoring_resume = lambda: None
    requested_counts = []

    def generate(_rollout_id, group_count, *, known_rollout_ids=None):
        requested_counts.append((group_count, known_rollout_ids))
        return [[SimpleNamespace(), SimpleNamespace()] for _ in range(group_count)], {}, set()

    manager._call_rollout_for_group_count = generate

    manager._generate_rs_candidates(7)
    assert requested_counts == [(8, None)]

    manager._pending_rs_batches[7] = {
        "accepted": [[SimpleNamespace()] for _ in range(4)],
        "unscored": [],
        "round": 0,
        "awaiting_log_prob_indices": None,
        "seen_rollout_ids": {"initial"},
        "metrics": {},
    }
    manager.generate_rs_replacement_candidates(7)

    assert requested_counts == [(8, None), (4, {"initial"})]
    assert manager._pending_rs_batches[7]["round"] == 1


def test_rollout_manager_rejects_an_unaligned_final_effective_batch():
    manager = object.__new__(RolloutManager.__ray_actor_class__)
    manager.args = SimpleNamespace(rs_batch_refill=True, rollout_batch_size=5, n_samples_per_prompt=2)
    topology = {"dp_size": 2, "vpp_size": 2, "microbatch_group_size_per_vp_stage": 4}

    with pytest.raises(ValueError, match=r"effective sample count.*10, alignment = 8"):
        manager.set_train_parallel_config(topology)

    assert not hasattr(manager, "train_parallel_config")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
