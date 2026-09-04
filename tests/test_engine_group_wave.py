import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.utils.engine_group_wave import build_engine_group_waves, run_engine_group_waves


NUM_GPUS = 0


@pytest.mark.unit
@pytest.mark.parametrize("engine_count", [1, 2, 4, 7])
def test_zero_limit_preserves_all_at_once(engine_count):
    engines = [f"engine-{index}" for index in range(engine_count)]

    waves = build_engine_group_waves(engines, 0)

    assert waves == (tuple(enumerate(engines)),)


@pytest.mark.unit
def test_positive_limit_builds_stable_bounded_waves():
    engines = [f"engine-{index}" for index in range(5)]

    waves = build_engine_group_waves(engines, 2)

    assert waves == (
        ((0, "engine-0"), (1, "engine-1")),
        ((2, "engine-2"), (3, "engine-3")),
        ((4, "engine-4"),),
    )


@pytest.mark.unit
def test_limit_larger_than_population_is_one_wave():
    assert build_engine_group_waves(["a", "b"], 8) == (((0, "a"), (1, "b")),)


@pytest.mark.unit
def test_empty_population_has_no_waves():
    assert build_engine_group_waves([], 1) == ()


@pytest.mark.unit
def test_negative_limit_is_rejected():
    with pytest.raises(ValueError, match="must be non-negative"):
        build_engine_group_waves(["engine"], -1)


@pytest.mark.unit
def test_runner_waits_before_submitting_the_next_wave():
    events = []

    def submit(index, engine):
        events.append(("submit", index, engine))
        return f"ref-{index}"

    def wait(refs):
        events.append(("wait", tuple(refs)))

    run_engine_group_waves(["a", "b", "c", "d", "e"], 2, submit, wait)

    assert events == [
        ("submit", 0, "a"),
        ("submit", 1, "b"),
        ("wait", ("ref-0", "ref-1")),
        ("submit", 2, "c"),
        ("submit", 3, "d"),
        ("wait", ("ref-2", "ref-3")),
        ("submit", 4, "e"),
        ("wait", ("ref-4",)),
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
