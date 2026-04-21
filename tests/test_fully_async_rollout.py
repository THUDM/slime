from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace


def load_fully_async_module():
    sample_status = Enum("Status", ["ABORTED", "COMPLETED"])

    @dataclass
    class Sample:
        group_index: int | None = None
        index: int = 0
        prompt: str = ""
        response: str = ""
        reward: float | None = None
        label: str = ""
        metadata: dict = field(default_factory=dict)
        weight_versions: list[str] = field(default_factory=list)
        status: object = sample_status.COMPLETED

    Sample.Status = sample_status

    class FakeGenerateState:
        def __init__(self, args):
            self.args = args
            self.sampling_params = {}
            self.reset()

        def reset(self):
            self.pendings = set()
            self.aborted = False

    async def fake_abort(args, rollout_id):
        return []

    async def fake_eval_rollout(args, rollout_id):
        return {"dummy": {"rewards": [1.0]}}, []

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        return group

    def fake_run(coro):
        raise RuntimeError("run() should not be called in this unit test")

    @dataclass
    class RolloutFnTrainOutput:
        samples: list
        metrics: dict | None = None

    modules = {
        "slime": types.ModuleType("slime"),
        "slime.rollout": types.ModuleType("slime.rollout"),
        "slime.rollout.base_types": types.ModuleType("slime.rollout.base_types"),
        "slime.rollout.sglang_rollout": types.ModuleType("slime.rollout.sglang_rollout"),
        "slime.utils": types.ModuleType("slime.utils"),
        "slime.utils.async_utils": types.ModuleType("slime.utils.async_utils"),
        "slime.utils.types": types.ModuleType("slime.utils.types"),
    }
    modules["slime.rollout.base_types"].RolloutFnTrainOutput = RolloutFnTrainOutput
    modules["slime.rollout.sglang_rollout"].GenerateState = FakeGenerateState
    modules["slime.rollout.sglang_rollout"].abort = fake_abort
    modules["slime.rollout.sglang_rollout"].eval_rollout = fake_eval_rollout
    modules["slime.rollout.sglang_rollout"].generate_and_rm_group = fake_generate_and_rm_group
    modules["slime.utils.async_utils"].run = fake_run
    modules["slime.utils.types"].Sample = Sample
    modules["slime"].rollout = modules["slime.rollout"]
    modules["slime"].utils = modules["slime.utils"]

    saved_modules = {name: sys.modules.get(name) for name in modules}
    saved_test_module = sys.modules.get("fully_async_rollout_under_test")

    try:
        sys.modules.update(modules)
        spec = importlib.util.spec_from_file_location(
            "fully_async_rollout_under_test",
            Path(__file__).resolve().parents[1] / "examples" / "fully_async" / "fully_async_rollout.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, Sample, sample_status
    finally:
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
        if saved_test_module is None:
            sys.modules.pop("fully_async_rollout_under_test", None)
        else:
            sys.modules["fully_async_rollout_under_test"] = saved_test_module


def make_args(**overrides):
    defaults = dict(
        sglang_server_concurrency=2,
        rollout_batch_size=4,
        rollout_global_dataset=True,
        partial_rollout=True,
        fully_async_buffer_policy="legacy_backpressure",
        fully_async_version_window=1,
        fully_async_max_completed_samples=None,
        fully_async_eviction_policy="drop_oldest_version",
        staleness_threshold=None,
        update_weights_interval=2,
        current_policy_version=0,
        current_rollout_id=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_derive_max_stale_samples_returns_none_without_threshold():
    module, _, _ = load_fully_async_module()
    assert module._derive_max_stale_samples(make_args()) is None


def test_derive_max_stale_samples_uses_threshold_formula():
    module, _, _ = load_fully_async_module()
    args = make_args(staleness_threshold=0.5, rollout_batch_size=4, update_weights_interval=2)
    assert module._derive_max_stale_samples(args) == 12


def test_worker_before_after_weight_update_without_running_loop_are_safe():
    module, _, _ = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.5), data_buffer=None)

    before = worker.before_weight_update(policy_version=3)
    after = worker.after_weight_update(policy_version=4)

    assert before["policy_version"] == 3
    assert before["active_samples"] == 0
    assert after["policy_version"] == 4
    assert after["stale_samples"] == 0
    assert after["max_stale_samples"] == 12


def test_before_weight_update_without_running_loop_uses_current_outstanding_samples():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.0), data_buffer=None)

    worker.output_queue.put(
        (
            10,
            [Sample(group_index=10, index=10, metadata={"fully_async_sample_id": 10}, status=sample_status.COMPLETED)],
        )
    )
    worker._add_recycled_sample_ids([11])

    before = worker.before_weight_update(policy_version=3)

    assert worker.get_stale_sample_count() == 1
    assert before["stale_samples"] == 2


def test_completed_samples_do_not_refund_staleness_budget():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.0, update_weights_interval=1), data_buffer=None)

    worker._set_stale_sample_ids([10, 11])
    worker.output_queue.put(
        (
            10,
            [Sample(group_index=10, index=10, metadata={"fully_async_sample_id": 10}, status=sample_status.COMPLETED)],
        )
    )
    worker.output_queue.put(
        (
            11,
            [Sample(group_index=11, index=11, metadata={"fully_async_sample_id": 11}, status=sample_status.COMPLETED)],
        )
    )

    completed = worker.get_completed_samples()

    assert [sample_id for sample_id, _ in completed] == [10, 11]
    assert worker.get_stale_sample_count() == 2


def test_completed_samples_limit_keeps_full_staleness_budget():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.0, update_weights_interval=1), data_buffer=None)

    worker._set_stale_sample_ids([10, 11, 12])
    for sample_id in (10, 11, 12):
        worker.output_queue.put(
            (
                sample_id,
                [
                    Sample(
                        group_index=sample_id,
                        index=sample_id,
                        metadata={"fully_async_sample_id": sample_id},
                        status=sample_status.COMPLETED,
                    )
                ],
            )
        )

    completed = worker.get_completed_samples(limit=2)

    assert [sample_id for sample_id, _ in completed] == [10, 11]
    assert worker.get_queue_size() == 1
    assert worker.get_stale_sample_count() == 3


def test_pause_for_staleness_uses_budgeted_stale_samples():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(
        make_args(staleness_threshold=0.0, update_weights_interval=1, rollout_batch_size=2),
        data_buffer=None,
    )

    worker.output_queue.put(
        (
            10,
            [Sample(group_index=10, index=10, metadata={"fully_async_sample_id": 10}, status=sample_status.COMPLETED)],
        )
    )
    worker.output_queue.put(
        (
            11,
            [Sample(group_index=11, index=11, metadata={"fully_async_sample_id": 11}, status=sample_status.COMPLETED)],
        )
    )

    assert worker.get_stale_sample_count() == 0
    assert worker._should_pause_for_staleness() is False

    worker._set_stale_sample_ids([10, 11])

    assert worker._should_pause_for_staleness() is True


def test_window_evict_policy_disables_staleness_pause():
    module, _, _ = load_fully_async_module()
    worker = module.AsyncRolloutWorker(
        make_args(
            fully_async_buffer_policy="window_evict",
            staleness_threshold=0.0,
            update_weights_interval=1,
            rollout_batch_size=2,
        ),
        data_buffer=None,
    )

    worker._set_stale_sample_ids([10, 11])

    assert worker._should_pause_for_staleness() is False


def test_prepare_for_weight_update_recomputes_outstanding_after_waiting_for_active_tasks():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.0, partial_rollout=False), data_buffer=None)

    async def run_test():
        worker.task_lock = asyncio.Lock()
        group = [
            Sample(group_index=21, index=21, metadata={"fully_async_sample_id": 21}, status=sample_status.COMPLETED)
        ]
        task = asyncio.create_task(asyncio.sleep(0, result=group))
        worker.state.pendings.add(task)
        worker.task_sample_ids[task] = 21

        before = await worker._prepare_for_weight_update_async(policy_version=3)

        assert worker.get_stale_sample_count() == 0
        assert before["active_samples"] == 1
        assert before["stale_samples"] == 1

    asyncio.run(run_test())


def test_extract_sample_id_prefers_stable_sample_identity():
    module, Sample, _ = load_fully_async_module()
    group = [
        Sample(group_index=0, metadata={"fully_async_sample_id": 0}),
        Sample(group_index=0, metadata={"fully_async_sample_id": 0}),
    ]
    assert module._extract_sample_id(group) == 0


def test_record_processed_samples_tracks_stale_and_partial_metrics():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.0, current_policy_version=3), data_buffer=None)

    stale_sample = [
        Sample(
            group_index=100,
            metadata={"fully_async_sample_id": 100, "fully_async_schedule_versions": [1]},
            weight_versions=["1"],
            status=sample_status.COMPLETED,
        ),
        Sample(
            group_index=100,
            metadata={"fully_async_sample_id": 100, "fully_async_schedule_versions": [1]},
            weight_versions=["1"],
            status=sample_status.COMPLETED,
        ),
    ]
    partial_sample = [
        Sample(
            group_index=101,
            metadata={"fully_async_sample_id": 101, "fully_async_schedule_versions": [2, 3]},
            weight_versions=["2"],
            status=sample_status.COMPLETED,
        ),
        Sample(
            group_index=101,
            metadata={"fully_async_sample_id": 101, "fully_async_schedule_versions": [2, 3]},
            weight_versions=["3"],
            status=sample_status.COMPLETED,
        ),
    ]

    worker.record_processed_samples([stale_sample, partial_sample])
    metrics = worker._snapshot_processed_metrics()

    assert metrics["fully_async/count/stale_samples_processed"] == 1
    assert metrics["fully_async/count/stale_trajectory_processed"] == 2
    assert metrics["fully_async/partial/total_partial_num"] == 1
    assert metrics["fully_async/partial/partial_ratio"] == 0.5
    assert metrics["fully_async/partial/max_partial_span"] == 1


def test_summarize_processed_group_reports_sources_and_staleness():
    module, Sample, sample_status = load_fully_async_module()

    group = [
        Sample(
            group_index=123,
            metadata={"fully_async_sample_id": 123, "fully_async_schedule_versions": [1, 2], "policy_version": 2},
            weight_versions=["1"],
            status=sample_status.COMPLETED,
        ),
        Sample(
            group_index=123,
            metadata={"fully_async_sample_id": 123, "fully_async_schedule_versions": [2], "policy_version": 2},
            weight_versions=[],
            status=sample_status.COMPLETED,
        ),
    ]

    summary = module._summarize_processed_group(group, current_policy_version=3)

    assert summary["sample_id"] == 123
    assert summary["group_min_version"] == 1
    assert summary["group_max_version"] == 2
    assert summary["stale_sample"] is True
    assert summary["stale_trajectory_count"] == 2
    assert summary["partial_span"] == 1
    assert summary["source_counts"] == {"weight_versions": 1, "fully_async_schedule_versions": 1}
    assert summary["staleness_source_counts"] == {"fully_async_schedule_versions": 2}


def test_record_processed_samples_aligns_weight_version_fallback_to_policy_version():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.0, current_policy_version=1), data_buffer=None)

    worker.record_processed_samples(
        [
            [
                Sample(
                    group_index=150,
                    metadata={"fully_async_sample_id": 150},
                    weight_versions=["2"],
                    status=sample_status.COMPLETED,
                )
            ]
        ]
    )
    metrics = worker._snapshot_processed_metrics()

    assert metrics["fully_async/count/stale_samples_processed"] == 0
    assert metrics["fully_async/count/stale_trajectory_processed"] == 0


def test_after_weight_update_resets_partial_window_metrics():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.0, current_policy_version=2), data_buffer=None)

    worker.record_processed_samples(
        [
            [
                Sample(
                    group_index=200,
                    metadata={"fully_async_sample_id": 200, "fully_async_schedule_versions": [1, 2]},
                    weight_versions=["1", "2"],
                    status=sample_status.COMPLETED,
                )
            ]
        ]
    )

    after = worker.after_weight_update(policy_version=3)

    assert after["fully_async/partial/total_partial_num"] == 1
    assert after["fully_async/partial/partial_ratio"] == 1.0
    assert after["fully_async/partial/max_partial_span"] == 1

    worker.record_processed_samples(
        [
            [
                Sample(
                    group_index=201,
                    metadata={"fully_async_sample_id": 201, "fully_async_schedule_versions": [3]},
                    weight_versions=["3"],
                    status=sample_status.COMPLETED,
                )
            ]
        ]
    )
    fresh_metrics = worker._snapshot_processed_metrics()
    assert fresh_metrics["fully_async/partial/total_partial_num"] == 0
    assert fresh_metrics["fully_async/partial/partial_ratio"] == 0.0
    assert fresh_metrics["fully_async/partial/max_partial_span"] == 0


def test_flush_metrics_returns_metrics_only_when_window_has_data():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.0, current_policy_version=2), data_buffer=None)

    worker.record_processed_samples(
        [
            [
                Sample(
                    group_index=250,
                    metadata={"fully_async_sample_id": 250, "fully_async_schedule_versions": [1, 2]},
                    weight_versions=["1", "2"],
                    status=sample_status.COMPLETED,
                )
            ]
        ]
    )
    worker.worker_thread = SimpleNamespace(is_alive=lambda: True)

    saved_global_worker = module._global_worker
    module._global_worker = worker
    try:
        metrics = module.flush_metrics(args=None, data_buffer=None)
        assert metrics["fully_async/count/stale_samples_processed"] == 0
        assert metrics["fully_async/count/stale_trajectory_processed"] == 0
        assert metrics["fully_async/partial/total_partial_num"] == 1
        assert metrics["fully_async/partial/partial_ratio"] == 1.0
        assert metrics["fully_async/partial/max_partial_span"] == 1

        worker.after_weight_update(policy_version=3)
        assert module.flush_metrics(args=None, data_buffer=None) is None
    finally:
        module._global_worker = saved_global_worker


def test_shutdown_worker_stops_and_clears_global_worker():
    module, _, _ = load_fully_async_module()
    stop_calls = []
    dummy_worker = SimpleNamespace(stop=lambda: stop_calls.append("stopped"))

    saved_global_worker = module._global_worker
    module._global_worker = dummy_worker
    try:
        module.shutdown_worker(args=None, data_buffer=None)
        assert stop_calls == ["stopped"]
        assert module._global_worker is None
    finally:
        module._global_worker = saved_global_worker


def test_after_weight_update_counts_recycled_partial_samples_as_stale():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.0, current_policy_version=2), data_buffer=None)

    worker.output_queue.put(
        (300, [Sample(group_index=300, metadata={"fully_async_sample_id": 300}, status=sample_status.COMPLETED)])
    )
    worker._add_recycled_sample_ids([301, 302])

    after = worker.after_weight_update(policy_version=3)

    assert after["stale_samples"] == 3
    assert worker.get_stale_sample_count() == 3


def test_after_weight_update_evicts_completed_samples_outside_version_window():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(
        make_args(
            fully_async_buffer_policy="window_evict",
            fully_async_version_window=1,
            current_policy_version=3,
        ),
        data_buffer=None,
    )

    for sample_id, version in ((320, 1), (321, 2), (322, 3)):
        worker.output_queue.put(
            (
                sample_id,
                [
                    Sample(
                        group_index=sample_id,
                        metadata={"fully_async_sample_id": sample_id, "fully_async_schedule_versions": [version]},
                        weight_versions=[str(version)],
                        status=sample_status.COMPLETED,
                    )
                ],
            )
        )
    worker._set_stale_sample_ids([320, 321, 322])

    after = worker.after_weight_update(policy_version=4)

    assert after["fully_async/window/evicted_samples"] == 2
    assert after["fully_async/window/evicted_by_version"] == 2
    assert worker.get_queue_size() == 1
    assert worker.get_stale_sample_count() == 1
    remaining = worker.get_completed_samples(current_policy_version=4)
    assert [sample_id for sample_id, _ in remaining] == [322]


def test_continuous_worker_loop_adds_newly_scheduled_samples_to_staleness_budget():
    module, Sample, sample_status = load_fully_async_module()

    class FakeDataBuffer:
        def __init__(self):
            self.returned = False

        def get_samples(self, num_samples):
            assert num_samples == 1
            if self.returned:
                return []
            self.returned = True
            return [
                [
                    Sample(
                        group_index=350,
                        index=350,
                        metadata={"fully_async_sample_id": 350},
                        status=sample_status.COMPLETED,
                    )
                ]
            ]

    worker = module.AsyncRolloutWorker(
        make_args(staleness_threshold=0.0, update_weights_interval=1, rollout_batch_size=1),
        data_buffer=FakeDataBuffer(),
    )

    async def run_test():
        loop_task = asyncio.create_task(worker.continuous_worker_loop())
        await asyncio.sleep(0.05)
        worker.running = False
        await loop_task

    asyncio.run(run_test())

    assert worker.get_stale_sample_count() == 1


def test_collect_task_result_clears_recycled_tracking_for_resumed_samples():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.0), data_buffer=None)

    async def complete_recycled_task():
        group = [
            Sample(group_index=400, metadata={"fully_async_sample_id": 400}, status=sample_status.COMPLETED),
        ]
        task = asyncio.create_task(asyncio.sleep(0, result=group))
        worker.task_sample_ids[task] = 400
        worker._add_recycled_sample_ids([400])
        await task
        worker._collect_task_result(task)

    asyncio.run(complete_recycled_task())

    assert 400 not in worker._snapshot_recycled_sample_ids()
    completed = worker.get_completed_samples()
    assert [sample_id for sample_id, _ in completed] == [400]


def test_window_evict_policy_evicts_oldest_version_on_completed_store_overflow():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(
        make_args(
            fully_async_buffer_policy="window_evict",
            fully_async_version_window=10,
            fully_async_max_completed_samples=2,
            current_policy_version=2,
        ),
        data_buffer=None,
    )

    for sample_id, version in ((410, 0), (411, 1), (412, 2)):
        worker.output_queue.put(
            (
                sample_id,
                [
                    Sample(
                        group_index=sample_id,
                        index=sample_id,
                        metadata={"fully_async_sample_id": sample_id, "fully_async_schedule_versions": [version]},
                        weight_versions=[str(version)],
                        status=sample_status.COMPLETED,
                    )
                ],
            )
        )

    remaining = worker.get_completed_samples(current_policy_version=2)

    assert [sample_id for sample_id, _ in remaining] == [411, 412]


def test_generate_rollout_async_only_drains_needed_completed_groups():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.0, rollout_batch_size=2), data_buffer=None)
    worker.worker_thread = SimpleNamespace(is_alive=lambda: True)

    for sample_id in (500, 501, 502):
        worker.output_queue.put(
            (
                sample_id,
                [
                    Sample(
                        group_index=sample_id,
                        index=sample_id,
                        metadata={"fully_async_sample_id": sample_id},
                        weight_versions=["1"],
                        status=sample_status.COMPLETED,
                    )
                ],
            )
        )

    saved_global_worker = module._global_worker
    module._global_worker = worker
    try:
        output = asyncio.run(
            module.generate_rollout_async(make_args(rollout_batch_size=2), rollout_id=0, data_buffer=None)
        )
    finally:
        module._global_worker = saved_global_worker

    assert [group[0].group_index for group in output.samples] == [500, 501]
    assert worker.get_queue_size() == 1


def test_generate_rollout_async_consumed_samples_refund_staleness_budget():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(make_args(staleness_threshold=0.0, rollout_batch_size=2), data_buffer=None)
    worker.worker_thread = SimpleNamespace(is_alive=lambda: True)

    for sample_id in (520, 521, 522):
        worker.output_queue.put(
            (
                sample_id,
                [
                    Sample(
                        group_index=sample_id,
                        index=sample_id,
                        metadata={"fully_async_sample_id": sample_id},
                        weight_versions=["1"],
                        status=sample_status.COMPLETED,
                    )
                ],
            )
        )
    worker._set_stale_sample_ids([520, 521, 522])

    saved_global_worker = module._global_worker
    module._global_worker = worker
    try:
        output = asyncio.run(
            module.generate_rollout_async(make_args(rollout_batch_size=2), rollout_id=0, data_buffer=None)
        )
    finally:
        module._global_worker = saved_global_worker

    assert [group[0].group_index for group in output.samples] == [520, 521]
    assert worker.get_queue_size() == 1
    assert worker.get_stale_sample_count() == 1


def test_generate_rollout_async_window_evict_skips_out_of_window_completed_groups():
    module, Sample, sample_status = load_fully_async_module()
    worker = module.AsyncRolloutWorker(
        make_args(
            fully_async_buffer_policy="window_evict",
            fully_async_version_window=1,
            current_policy_version=3,
            rollout_batch_size=2,
        ),
        data_buffer=None,
    )
    worker.worker_thread = SimpleNamespace(is_alive=lambda: True)

    for sample_id, version in ((540, 1), (541, 2), (542, 3)):
        worker.output_queue.put(
            (
                sample_id,
                [
                    Sample(
                        group_index=sample_id,
                        index=sample_id,
                        metadata={"fully_async_sample_id": sample_id, "fully_async_schedule_versions": [version]},
                        weight_versions=[str(version)],
                        status=sample_status.COMPLETED,
                    )
                ],
            )
        )

    saved_global_worker = module._global_worker
    module._global_worker = worker
    try:
        output = asyncio.run(
            module.generate_rollout_async(
                make_args(
                    fully_async_buffer_policy="window_evict",
                    fully_async_version_window=1,
                    current_policy_version=3,
                    rollout_batch_size=2,
                ),
                rollout_id=0,
                data_buffer=None,
            )
        )
    finally:
        module._global_worker = saved_global_worker

    assert [group[0].group_index for group in output.samples] == [541, 542]
    assert worker.get_queue_size() == 0


def test_generate_rollout_async_recycled_samples_stay_in_staleness_budget():
    module, Sample, sample_status = load_fully_async_module()

    class FakeDataBuffer:
        def __init__(self):
            self.samples = []

        def add_samples(self, samples):
            self.samples.extend(samples)

    data_buffer = FakeDataBuffer()
    worker = module.AsyncRolloutWorker(
        make_args(staleness_threshold=0.0, rollout_batch_size=1), data_buffer=data_buffer
    )
    worker.worker_thread = SimpleNamespace(is_alive=lambda: True)

    worker.output_queue.put(
        (
            530,
            [
                Sample(
                    group_index=530,
                    index=530,
                    metadata={"fully_async_sample_id": 530},
                    weight_versions=["1"],
                    status=sample_status.ABORTED,
                )
            ],
        )
    )
    worker.output_queue.put(
        (
            531,
            [
                Sample(
                    group_index=531,
                    index=531,
                    metadata={"fully_async_sample_id": 531},
                    weight_versions=["1"],
                    status=sample_status.COMPLETED,
                )
            ],
        )
    )
    worker._set_stale_sample_ids([530, 531])

    saved_global_worker = module._global_worker
    module._global_worker = worker
    try:
        output = asyncio.run(
            module.generate_rollout_async(make_args(rollout_batch_size=1), rollout_id=0, data_buffer=data_buffer)
        )
    finally:
        module._global_worker = saved_global_worker

    assert [group[0].group_index for group in output.samples] == [531]
    assert [group[0].group_index for group in data_buffer.samples] == [530]
    assert worker.get_stale_sample_count() == 1
