import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

import slime.rollout.rm_hub as rm_hub
from slime.utils.types import Sample


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rm_type", "scorer_name", "scorer_result", "expected_reward"),
    [
        pytest.param("deepscaler", "get_deepscaler_rule_based_reward", 0.75, 0.75, id="deepscaler"),
        pytest.param("dapo", "compute_score_dapo", 0.75, 0.75, id="dapo"),
        pytest.param("math", "grade_answer_verl", True, 1, id="math-correct"),
        pytest.param("math", "grade_answer_verl", False, 0, id="math-incorrect"),
        pytest.param("f1", "f1_score", (0.75, 0.5, 1.0), 0.75, id="f1"),
        pytest.param("gpqa", "compute_gpqa_reward", 0.75, 0.75, id="gpqa"),
        pytest.param("ifbench", None, 0.75, 0.75, id="ifbench"),
    ],
)
def test_rule_based_rm_runs_scorer_in_worker_thread_and_preserves_result(
    monkeypatch, rm_type, scorer_name, scorer_result, expected_reward
):
    scorer_thread_id = None

    def scorer(*args, **kwargs):
        nonlocal scorer_thread_id
        scorer_thread_id = threading.get_ident()
        return scorer_result

    if scorer_name is None:
        ifbench = ModuleType("slime.rollout.rm_hub.ifbench")
        ifbench.compute_ifbench_reward = scorer
        monkeypatch.setitem(sys.modules, ifbench.__name__, ifbench)
    else:
        monkeypatch.setattr(rm_hub, scorer_name, scorer)

    async def run_test():
        event_loop_thread_id = threading.get_ident()
        args = SimpleNamespace(custom_rm_path=None, rm_type=rm_type)
        reward = await rm_hub.async_rm(args, Sample(response="response", label="label"))
        return event_loop_thread_id, reward

    event_loop_thread_id, reward = asyncio.run(run_test())

    assert reward == expected_reward
    assert scorer_thread_id is not None
    assert scorer_thread_id != event_loop_thread_id


@pytest.mark.unit
def test_rule_based_rm_preserves_scorer_exception(monkeypatch):
    class ScorerError(RuntimeError):
        pass

    expected_error = ScorerError("scorer failed")

    def failing_scorer(response, label):
        raise expected_error

    monkeypatch.setattr(rm_hub, "get_deepscaler_rule_based_reward", failing_scorer)

    async def run_test():
        args = SimpleNamespace(custom_rm_path=None, rm_type="deepscaler")
        with pytest.raises(ScorerError) as caught:
            await rm_hub.async_rm(args, Sample(response="response", label="label"))
        return caught.value

    actual_error = asyncio.run(run_test())

    assert actual_error is expected_error
    assert type(actual_error) is ScorerError
    assert str(actual_error) == "scorer failed"


@pytest.mark.unit
def test_rule_based_rm_keeps_event_loop_responsive(monkeypatch):
    async def run_test():
        scorer_started = threading.Event()
        release_scorer = threading.Event()

        def blocking_scorer(response, label):
            scorer_started.set()
            release_scorer.wait()
            return 0.75

        monkeypatch.setattr(rm_hub, "get_deepscaler_rule_based_reward", blocking_scorer)
        args = SimpleNamespace(custom_rm_path=None, rm_type="deepscaler")
        sample = Sample(response="response", label="label")

        # Prevent a broken implementation from hanging the test indefinitely.
        failsafe = threading.Timer(1.0, release_scorer.set)
        failsafe.start()
        reward_task = asyncio.create_task(rm_hub.async_rm(args, sample))

        try:
            deadline = asyncio.get_running_loop().time() + 2.0
            while not scorer_started.is_set():
                if asyncio.get_running_loop().time() >= deadline:
                    pytest.fail("rule-based scorer did not start")
                await asyncio.sleep(0)

            assert not reward_task.done(), "rule-based scorer blocked the event loop until it completed"
            release_scorer.set()
            assert await reward_task == 0.75
        finally:
            release_scorer.set()
            failsafe.cancel()
            await asyncio.gather(reward_task, return_exceptions=True)

    asyncio.run(run_test())
