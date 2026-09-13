import pytest

from slime.utils.misc import should_run_periodic_action

NUM_GPUS = 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rollout_id", "interval", "num_rollout_per_epoch", "num_rollout", "skip_final_step", "expected"),
    [
        (49, None, None, 50, False, False),
        (4, 5, None, None, False, True),
        (3, 10, 4, None, False, True),
        (49, 200, None, 50, False, True),
        (49, 200, None, 50, True, False),
        (49, 50, None, 50, True, False),
        (49, 200, 50, 50, True, False),
    ],
    ids=[
        "disabled-without-interval",
        "regular-interval",
        "epoch-boundary",
        "final-step-default",
        "skip-final-step-off-cadence",
        "skip-final-step-on-interval",
        "skip-final-step-on-epoch-boundary",
    ],
)
def test_should_run_periodic_action(
    rollout_id,
    interval,
    num_rollout_per_epoch,
    num_rollout,
    skip_final_step,
    expected,
):
    assert (
        should_run_periodic_action(
            rollout_id,
            interval,
            num_rollout_per_epoch,
            num_rollout,
            skip_final_step=skip_final_step,
        )
        is expected
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
