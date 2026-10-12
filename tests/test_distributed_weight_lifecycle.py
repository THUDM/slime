"""Control-plane checks for disaggregated weight publication and peer replacement.

These tests exercise the updater while replacing Ray RPCs and process-group
operations. They establish lifecycle ordering and failure propagation, rather
than claiming that a failed NCCL communicator can recover in place.
"""

import importlib.util
import sys
import types
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest

NUM_GPUS = 0


class RemoteMethod:
    def __init__(self, events, actor_id, name):
        self.events, self.actor_id, self.name = events, actor_id, name

    def remote(self, *args, **kwargs):
        operation = (self.actor_id, self.name, args, kwargs)
        self.events.append(operation)
        return operation


class Engine:
    def __init__(self, events, actor_id):
        self._actor_id = actor_id
        for name in (
            "pause_generation",
            "flush_cache",
            "continue_generation",
            "init_weights_update_group",
            "destroy_weights_update_group",
            "acquire",
            "release",
        ):
            setattr(self, name, RemoteMethod(events, actor_id, name))


@pytest.fixture
def lifecycle(monkeypatch):
    mpu = types.SimpleNamespace(
        get_data_parallel_rank=lambda **kwargs: 0,
        get_tensor_model_parallel_rank=lambda: 0,
        get_pipeline_model_parallel_rank=lambda: 0,
    )
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    core.mpu = mpu
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    common = types.ModuleType("slime.backends.megatron_utils.update_weight.common")
    common.all_gather_param = lambda name, param: param
    common.named_params_and_buffers = lambda *args: ()
    monkeypatch.setitem(sys.modules, common.__name__, common)
    converter = types.ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    converter.convert_to_hf = lambda *args: []
    monkeypatch.setitem(sys.modules, converter.__name__, converter)

    path = (
        Path(__file__).resolve().parents[1]
        / "slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py"
    )
    spec = importlib.util.spec_from_file_location(
        "slime.backends.megatron_utils.update_weight._distributed_lifecycle_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events, gets, destroyed, created = [], [], [], []

    def get(refs, **kwargs):
        gets.append((refs, kwargs))
        return True

    module.ray = types.SimpleNamespace(
        get=get,
        _private=types.SimpleNamespace(services=types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")),
    )
    module.dist = types.SimpleNamespace(
        get_rank=lambda: 0,
        barrier=lambda **kwargs: None,
        destroy_process_group=lambda group: destroyed.append(group),
    )
    module.get_gloo_group = lambda: "gloo"
    module.accelerator = types.SimpleNamespace(weight_update_backend=lambda: "nccl")

    def init_group(**kwargs):
        created.append(kwargs)
        return "new-group"

    module.init_process_group = init_group
    return types.SimpleNamespace(module=module, events=events, gets=gets, destroyed=destroyed, created=created)


def make_updater(state, backend="sglang-omni", **overrides):
    args = Namespace(
        rollout_backend=backend,
        distributed_timeout_minutes=1,
        rollout_num_gpus_per_engine=1,
        flush_cache_interval=1,
        update_weight_start_version=0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    updater = state.module.UpdateWeightFromDistributed(
        args, [], lambda: {}, model_name="prism", quantization_config=None
    )
    updater._is_pp_src_rank = True
    updater._group_name = "slime-pp_0"
    updater._model_update_groups = "old-group"
    return updater


def operation_names(state):
    return [event[1] for event in state.events]


def test_reconnect_resets_only_retained_actor_ids(lifecycle):
    updater = make_updater(lifecycle)
    updater.rollout_engines = [Engine(lifecycle.events, "gone"), Engine(lifecycle.events, "retained")]
    # Ray can deserialize a distinct Python handle for the same surviving actor.
    current = [Engine(lifecycle.events, "new"), Engine(lifecycle.events, "retained")]
    updater.connect_rollout_engines(current, Engine(lifecycle.events, "lock"))

    resets = [event[0] for event in lifecycle.events if event[1] == "destroy_weights_update_group"]
    assert resets == ["retained"]
    assert lifecycle.destroyed == ["old-group"]
    assert updater.rollout_engines is current
    assert updater._model_update_groups == "new-group"
    assert operation_names(lifecycle) == [
        "destroy_weights_update_group",
        "init_weights_update_group",
        "init_weights_update_group",
    ]


def test_reconnect_failure_cannot_publish_through_destroyed_group(lifecycle, monkeypatch):
    updater = make_updater(lifecycle)
    updater.rollout_engines = [Engine(lifecycle.events, "old")]
    error = RuntimeError("replacement did not join")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(lifecycle.module, "connect_rollout_engines_from_distributed", fail)
    with pytest.raises(RuntimeError, match="replacement did not join") as raised:
        updater.connect_rollout_engines([Engine(lifecycle.events, "new")], Engine(lifecycle.events, "lock"))
    assert raised.value is error
    assert updater._model_update_groups is None
    assert lifecycle.destroyed == ["old-group"]
    with pytest.raises(RuntimeError, match="without a connected"):
        updater.update_weights()
    assert "pause_generation" not in operation_names(lifecycle)


def test_teardown_failure_stops_reconnect_and_invalidates_local_group(lifecycle, monkeypatch):
    updater = make_updater(lifecycle)
    updater.rollout_engines = [Engine(lifecycle.events, "retained")]
    error = TimeoutError("old peer teardown exceeded its deadline")

    def fail(refs, **kwargs):
        assert kwargs == {"timeout": 120.0}
        raise error

    monkeypatch.setattr(lifecycle.module.ray, "get", fail)
    with pytest.raises(TimeoutError, match="old peer teardown") as raised:
        updater.connect_rollout_engines([Engine(lifecycle.events, "retained")], Engine(lifecycle.events, "lock"))
    assert raised.value is error
    assert lifecycle.destroyed == ["old-group"]
    assert lifecycle.created == []
    assert updater._model_update_groups is None


def test_omni_group_join_has_finite_budget_and_distinct_replica_ranks(lifecycle):
    updater = make_updater(lifecycle, distributed_timeout_minutes=2)
    updater._model_update_groups = None
    updater.connect_rollout_engines(
        [Engine(lifecycle.events, "first"), Engine(lifecycle.events, "second")],
        Engine(lifecycle.events, "lock"),
        engine_gpu_counts=[1, 2],
    )
    assert lifecycle.created[0]["timeout"] == timedelta(minutes=2)
    assert lifecycle.created[0]["world_size"] == 4
    calls = [event[3] for event in lifecycle.events if event[1] == "init_weights_update_group"]
    assert [call["rank_offset"] for call in calls] == [1, 2]
    assert all(call["world_size"] == 4 for call in calls)
    assert lifecycle.gets[0][1] == {"timeout": 240.0}


def test_failed_engine_join_destroys_created_local_group(lifecycle, monkeypatch):
    updater = make_updater(lifecycle)
    updater._model_update_groups = None

    def fail(refs, **kwargs):
        raise TimeoutError("engine join failed")

    monkeypatch.setattr(lifecycle.module.ray, "get", fail)
    with pytest.raises(TimeoutError, match="engine join failed"):
        updater.connect_rollout_engines([Engine(lifecycle.events, "new")], Engine(lifecycle.events, "lock"))
    assert lifecycle.destroyed == ["new-group"]
    assert updater._model_update_groups is None


def test_native_group_join_keeps_original_default_timeouts(lifecycle):
    updater = make_updater(lifecycle, backend="sglang")
    updater._model_update_groups = None
    updater.connect_rollout_engines([Engine(lifecycle.events, "text")], Engine(lifecycle.events, "lock"))
    assert "timeout" not in lifecycle.created[0]
    assert lifecycle.gets[0][1] == {}


@pytest.mark.parametrize(
    ("backend", "flush_interval", "expected"),
    [
        ("sglang-omni", 1, ["pause_generation", "continue_generation"]),
        ("sglang", 1, ["pause_generation", "flush_cache", "continue_generation"]),
        ("sglang-omni", 0, ["pause_generation", "continue_generation"]),
    ],
)
def test_omni_flush_is_delegated_to_each_weight_bucket(lifecycle, backend, flush_interval, expected):
    updater = make_updater(lifecycle, backend=backend, flush_cache_interval=flush_interval)
    updater.weight_version = 1  # Check the interval after the mandatory initial flush.
    updater.rollout_engines = [Engine(lifecycle.events, "student")]
    updater._send_weights = lambda pbar: None
    updater.update_weights()
    assert operation_names(lifecycle) == expected
    mode = lifecycle.events[0][3]["mode"]
    assert mode == ("abort" if flush_interval else "in_place")
    assert all(kwargs == ({"timeout": 120.0} if backend == "sglang-omni" else {}) for _, kwargs in lifecycle.gets)


def test_publication_failure_never_resumes_generation(lifecycle):
    updater = make_updater(lifecycle)
    updater.rollout_engines = [Engine(lifecycle.events, "student")]

    def fail(pbar):
        raise RuntimeError("NCCL transfer failed")

    updater._send_weights = fail
    with pytest.raises(RuntimeError, match="NCCL transfer failed"):
        updater.update_weights()
    assert operation_names(lifecycle) == ["pause_generation"]


def test_failed_bucket_releases_lock_without_clearing_failed_payload(lifecycle, monkeypatch):
    updater = make_updater(lifecycle)
    updater.rollout_engines = [Engine(lifecycle.events, "student")]
    updater.rollout_engine_lock = Engine(lifecycle.events, "lock")
    monkeypatch.setattr(lifecycle.module, "update_weights_from_distributed", lambda *args, **kwargs: "transfer")

    def get(refs, **kwargs):
        if refs == "transfer":
            raise TimeoutError("weight bucket did not complete")
        return True

    monkeypatch.setattr(lifecycle.module.ray, "get", get)
    payload = [("weight", object())]
    with pytest.raises(TimeoutError, match="weight bucket did not complete"):
        updater._update_bucket_weights_from_distributed(payload)
    assert operation_names(lifecycle) == ["acquire", "release"]
    assert len(payload) == 1


def test_lock_contention_has_a_finite_omni_deadline(lifecycle, monkeypatch):
    updater = make_updater(lifecycle)
    updater.rollout_engine_lock = Engine(lifecycle.events, "lock")
    monkeypatch.setattr(lifecycle.module.ray, "get", lambda refs, **kwargs: False)
    times = iter([0.0, 120.0])
    monkeypatch.setattr(lifecycle.module.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="acquiring the rollout"):
        updater._update_bucket_weights_from_distributed([])
    assert operation_names(lifecycle) == ["acquire"]


def test_disconnect_failure_preserves_error_and_invalidates_group(lifecycle, monkeypatch):
    updater = make_updater(lifecycle)
    updater.rollout_engines = [Engine(lifecycle.events, "student")]
    error = RuntimeError("NCCL communicator destruction failed")

    def destroy(group):
        raise error

    monkeypatch.setattr(lifecycle.module.dist, "destroy_process_group", destroy)
    with pytest.raises(RuntimeError, match="communicator destruction failed") as raised:
        updater.disconnect_rollout_engines()
    assert raised.value is error
    assert updater._model_update_groups is None
