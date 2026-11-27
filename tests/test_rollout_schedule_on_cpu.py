from slime.utils.misc import compute_rollout_schedule, should_run_periodic_action


def test_num_epoch_uses_total_prompt_count_before_division():
    num_rollout, epoch_boundaries = compute_rollout_schedule(
        dataset_size=10,
        rollout_batch_size=4,
        num_epoch=3,
    )

    assert num_rollout == 7
    assert epoch_boundaries == frozenset({2, 4})


def test_dataset_smaller_than_batch_still_runs_one_rollout():
    num_rollout, epoch_boundaries = compute_rollout_schedule(
        dataset_size=3,
        rollout_batch_size=8,
        num_epoch=2,
    )

    assert num_rollout == 1
    assert epoch_boundaries == frozenset({0})


def test_epoch_boundaries_trigger_periodic_actions_between_intervals():
    epoch_boundaries = frozenset({2, 4})

    assert not should_run_periodic_action(1, interval=100, epoch_boundaries=epoch_boundaries)
    assert should_run_periodic_action(2, interval=100, epoch_boundaries=epoch_boundaries)
    assert should_run_periodic_action(4, interval=100, epoch_boundaries=epoch_boundaries)


def test_final_rollout_still_triggers_periodic_action():
    assert should_run_periodic_action(6, interval=100, epoch_boundaries=frozenset({2, 4}), num_rollout=7)
