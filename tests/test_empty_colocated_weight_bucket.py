import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeFlattenedTensorBucket:
    supports_multi_dtypes = True

    def __init__(self, *, named_tensors=None, flattened_tensor=None, metadata=None):
        if named_tensors is not None:
            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")
            self._flattened_tensor = ("flattened", tuple(name for name, _ in named_tensors))
            self._metadata = tuple(name for name, _ in named_tensors)
            return

        self._flattened_tensor = flattened_tensor
        self._metadata = metadata

    def get_flattened_tensor(self):
        return self._flattened_tensor

    def get_metadata(self):
        return self._metadata


class _FakeMultiprocessingSerializer:
    @staticmethod
    def serialize(value, output_str):
        assert output_str is True
        return value


class _FakeRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return f"ref-{len(self.calls)}"


class _FakeEngine:
    def __init__(self):
        self.update_weights_from_tensor = _FakeRemoteMethod()


def _install_fake_deps(monkeypatch):
    dist_state = types.SimpleNamespace(
        rank=0,
        world_size=2,
        gathered=None,
        local_object=None,
        broadcast_value=None,
    )

    slime_pkg = types.ModuleType("slime")
    slime_pkg.__path__ = [str(REPO_ROOT / "slime")]
    slime_backends_pkg = types.ModuleType("slime.backends")
    slime_backends_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends")]
    megatron_utils_pkg = types.ModuleType("slime.backends.megatron_utils")
    megatron_utils_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends" / "megatron_utils")]
    update_weight_pkg = types.ModuleType("slime.backends.megatron_utils.update_weight")
    update_weight_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight")]
    slime_utils_pkg = types.ModuleType("slime.utils")
    slime_utils_pkg.__path__ = [str(REPO_ROOT / "slime" / "utils")]
    accelerator_mod = types.ModuleType("slime.utils.accelerator")
    accelerator_mod.device = lambda: "cuda:0"
    accelerator_mod.current_device = lambda: "cuda:0"
    accelerator_mod.ipc_collect = lambda: None
    accelerator_mod.empty_cache = lambda: None

    dist_mod = types.ModuleType("torch.distributed")

    def gather_object(obj, object_gather_list, dst, group):
        dist_state.local_object = obj
        if object_gather_list is not None:
            object_gather_list[:] = dist_state.gathered(obj)

    def broadcast_object_list(status, src, group):
        assert group is not None
        if dist_state.rank == src:
            dist_state.broadcast_value = status[0]
        else:
            status[0] = dist_state.broadcast_value

    dist_mod.get_rank = lambda: dist_state.rank
    dist_mod.get_world_size = lambda group=None: dist_state.world_size
    dist_mod.gather_object = gather_object
    dist_mod.broadcast_object_list = broadcast_object_list

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.dtype = object
    torch_mod.uint8 = "uint8"
    torch_mod.distributed = dist_mod
    torch_mod.empty = lambda size, dtype, device: {"size": size, "dtype": dtype, "device": device}
    torch_mod.no_grad = lambda: (lambda fn: fn)
    torch_mod.cuda = types.SimpleNamespace(current_device=lambda: "cuda:0", ipc_collect=lambda: None)
    torch_mod.nn = types.SimpleNamespace(Module=object)

    ray_mod = types.ModuleType("ray")
    ray_mod.ObjectRef = object
    ray_mod.get = lambda refs: refs
    ray_actor_mod = types.ModuleType("ray.actor")
    ray_actor_mod.ActorHandle = object

    mpu_mod = types.ModuleType("megatron.core.mpu")
    megatron_mod = types.ModuleType("megatron")
    megatron_core_mod = types.ModuleType("megatron.core")
    megatron_core_mod.mpu = mpu_mod

    sglang_mod = types.ModuleType("slime.backends.megatron_utils.sglang")
    sglang_mod.FlattenedTensorBucket = _FakeFlattenedTensorBucket
    sglang_mod.MultiprocessingSerializer = _FakeMultiprocessingSerializer

    megatron_to_hf_mod = types.ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    megatron_to_hf_mod.convert_to_hf = lambda *args, **kwargs: []

    expert_routing_mod = types.ModuleType("slime.backends.megatron_utils.update_weight.expert_routing")
    expert_routing_mod.configure_expert_routing = lambda *args, **kwargs: (None, [])

    hf_weight_iterator_direct_mod = types.ModuleType(
        "slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct"
    )
    hf_weight_iterator_direct_mod.HfWeightIteratorDirect = lambda *args, **kwargs: None

    slime_utils_types_mod = types.ModuleType("slime.utils.types")
    slime_utils_types_mod.ParamInfo = type("ParamInfo", (), {})

    distributed_utils_mod = types.ModuleType("slime.utils.distributed_utils")
    distributed_utils_mod.get_gloo_group = lambda: object()

    update_from_distributed_mod = types.ModuleType(
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    )
    update_from_distributed_mod.connect_rollout_engines_from_distributed = lambda *args, **kwargs: None
    update_from_distributed_mod.disconnect_rollout_engines_from_distributed = lambda *args, **kwargs: None
    update_from_distributed_mod.launch_weights_from_distributed = lambda *args, **kwargs: ([], [])
    update_from_distributed_mod.post_process_weights = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "slime", slime_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends", slime_backends_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils", megatron_utils_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.update_weight", update_weight_pkg)
    monkeypatch.setitem(sys.modules, "slime.utils", slime_utils_pkg)
    monkeypatch.setitem(sys.modules, "slime.utils.accelerator", accelerator_mod)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.actor", ray_actor_mod)
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.megatron_to_hf", megatron_to_hf_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.update_weight.expert_routing", expert_routing_mod)
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct",
        hf_weight_iterator_direct_mod,
    )
    monkeypatch.setitem(sys.modules, "slime.utils.types", slime_utils_types_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.distributed_utils", distributed_utils_mod)
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed",
        update_from_distributed_mod,
    )

    return dist_state


def _load_update_weight_module(monkeypatch):
    dist_state = _install_fake_deps(monkeypatch)

    module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_tensor"
    sys.modules.pop(module_name, None)
    module_path = (
        REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight" / "update_weight_from_tensor.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, dist_state


def test_empty_colocated_bucket_still_participates_in_gather(monkeypatch):
    module, dist_state = _load_update_weight_module(monkeypatch)
    dist_state.gathered = lambda local: [local, []]
    engine = _FakeEngine()

    refs, long_lived_tensors = module._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=3,
    )

    assert dist_state.local_object == []
    assert refs == []
    assert long_lived_tensors == []
    assert engine.update_weights_from_tensor.calls == []


def test_source_rank_pads_empty_colocated_bucket_entries(monkeypatch):
    module, dist_state = _load_update_weight_module(monkeypatch)
    remote_serialized_bucket = {"flattened_tensor": ("remote",), "metadata": ("remote_weight",)}
    dist_state.gathered = lambda local: [local, [remote_serialized_bucket]]
    engine = _FakeEngine()

    refs, long_lived_tensors = module._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=7,
    )

    assert refs == ["ref-1"]
    assert len(long_lived_tensors) == 1
    empty_bucket = long_lived_tensors[0]
    assert empty_bucket["metadata"] == []
    assert empty_bucket["flattened_tensor"] == {"size": 0, "dtype": "uint8", "device": "cuda:0"}

    assert engine.update_weights_from_tensor.calls == [
        {
            "serialized_named_tensors": [empty_bucket, remote_serialized_bucket],
            "load_format": "flattened_bucket",
            "weight_version": "7",
        }
    ]


def test_tensor_credit_window_launches_all_admitted_buckets_before_waiting(monkeypatch):
    module, _ = _load_update_weight_module(monkeypatch)
    events = []

    class _Tensor:
        def __init__(self, size):
            self.size = size

        def numel(self):
            return self.size

        def element_size(self):
            return 1

    class _Handle:
        def __init__(self, name):
            self.name = name

        def wait(self):
            events.append(("wait", self.name))

    updater = object.__new__(module.UpdateWeightFromTensor)
    updater._weight_sync_credit = module.WeightSyncCreditController(
        max_inflight_buckets=2,
        max_inflight_bytes=10,
    )
    updater._weight_sync_credit.begin_version(4)
    updater._weight_bucket_bytes = lambda bucket: sum(tensor.numel() for _, tensor in bucket)
    updater._ipc_gather_group = None
    updater._ipc_gather_src = None
    updater.rank = 0

    def send_hf_params(bucket):
        name = bucket[0][0]
        events.append(("launch", name))
        return [f"ref-{name}"], [_Handle(name)], object()

    updater._send_hf_params = send_hf_params
    module.ray.get = lambda refs: events.append(("get", refs[0]))
    buckets = [[("b0", _Tensor(6))], [("b1", _Tensor(4))], [("b2", _Tensor(5))]]

    updater._send_weight_bucket_windows(iter(buckets))
    updater._weight_sync_credit.commit_version(4)

    assert events == [
        ("launch", "b0"),
        ("launch", "b1"),
        ("wait", "b0"),
        ("get", "ref-b0"),
        ("wait", "b1"),
        ("get", "ref-b1"),
        ("launch", "b2"),
        ("wait", "b2"),
        ("get", "ref-b2"),
    ]
    assert buckets == [[], [], []]


def test_tensor_byte_credit_uses_the_largest_trainer_rank_bucket(monkeypatch):
    module, _ = _load_update_weight_module(monkeypatch)

    class _Tensor:
        def numel(self):
            return 5

        def element_size(self):
            return 1

    class _Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    updater = object.__new__(module.UpdateWeightFromTensor)
    updater._weight_sync_credit = module.WeightSyncCreditController(max_inflight_bytes=16)
    monkeypatch.setattr(module.torch, "int64", "int64", raising=False)
    monkeypatch.setattr(module.torch, "tensor", lambda value, **_: _Scalar(value), raising=False)
    monkeypatch.setattr(module.dist, "ReduceOp", types.SimpleNamespace(MAX="max"), raising=False)

    def all_reduce(value, *, op, group):
        assert op == "max"
        assert group is not None
        value.value = 12

    monkeypatch.setattr(module.dist, "all_reduce", all_reduce, raising=False)

    assert updater._weight_bucket_bytes([("local", _Tensor())]) == 12


def test_colocated_source_broadcasts_load_failure_before_credit_release(monkeypatch):
    module, dist_state = _load_update_weight_module(monkeypatch)
    error = RuntimeError("engine load failed")
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater.rank = 0
    updater._ipc_gather_src = 0
    updater._ipc_gather_group = object()
    updater._weight_sync_credit = module.WeightSyncCreditController(max_inflight_buckets=1)
    updater._weight_sync_credit.begin_version(6)
    reservation = updater._weight_sync_credit.reserve(8)
    assert reservation is not None
    updater._weight_sync_credit.mark_launched(
        reservation,
        transport_bytes=8,
        staging_bytes=16,
        consumer_objects=1,
    )
    module.ray.get = lambda _refs: (_ for _ in ()).throw(error)

    with pytest.raises(RuntimeError, match="engine load failed") as raised:
        updater._wait_for_bucket_completion(reservation, [], ["engine-ref"])
    assert raised.value is error
    assert dist_state.broadcast_value == "RuntimeError: engine load failed"
    assert updater._weight_sync_credit.snapshot.inflight_buckets == 1
    assert updater._weight_sync_credit.snapshot.pending_consumer_objects == 1

    updater._weight_sync_credit.fail_version(6, error)
    with pytest.raises(RuntimeError, match="cannot commit failed weight version"):
        updater._weight_sync_credit.commit_version(6)


def test_colocated_non_source_observes_source_failure(monkeypatch):
    module, dist_state = _load_update_weight_module(monkeypatch)
    dist_state.rank = 1
    dist_state.broadcast_value = "RuntimeError: engine load failed"
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater.rank = 1
    updater._ipc_gather_src = 0
    updater._ipc_gather_group = object()
    updater._weight_sync_credit = module.WeightSyncCreditController(max_inflight_buckets=1)
    updater._weight_sync_credit.begin_version(6)
    reservation = updater._weight_sync_credit.reserve(8)
    assert reservation is not None
    updater._weight_sync_credit.mark_launched(
        reservation,
        transport_bytes=8,
        staging_bytes=16,
        consumer_objects=0,
    )

    with pytest.raises(RuntimeError, match="consumer failed on engine source rank"):
        updater._wait_for_bucket_completion(reservation, [], [])
    assert updater._weight_sync_credit.snapshot.inflight_buckets == 1
    assert updater._weight_sync_credit.snapshot.transport_outstanding_bytes == 8


def _make_empty_update_updater(module, events, continue_result="continue-ref"):
    class _EventRemote:
        def __init__(self, name, result):
            self.name = name
            self.result = result

        def remote(self):
            events.append(self.name)
            return self.result

    colocated_engine = types.SimpleNamespace(
        pause_generation=_EventRemote("pause", "pause-ref"),
        flush_cache=_EventRemote("flush", "flush-ref"),
        continue_generation=_EventRemote("continue", continue_result),
    )
    distributed_engine = types.SimpleNamespace(
        pause_generation=_EventRemote("pause-distributed", "pause-distributed-ref"),
        flush_cache=_EventRemote("flush-distributed", "flush-distributed-ref"),
        continue_generation=_EventRemote("continue-distributed", "continue-distributed-ref"),
    )
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater.rank = 0
    updater.weight_version = 0
    updater.rollout_engines = [colocated_engine]
    updater._all_rollout_engines = [colocated_engine, distributed_engine]
    updater.quantization_config = None
    updater._weight_sync_credit = module.WeightSyncCreditController(max_inflight_buckets=1)
    updater.weights_getter = lambda: {}
    updater._expert_transfer_plan = []
    updater._non_expert_param_info_buckets = None
    updater._full_param_info_buckets = ()
    updater._hf_weight_iterator = types.SimpleNamespace(get_hf_weight_chunks=lambda *args, **kwargs: iter(()))
    updater.update_weight_metrics = {}
    return updater


def test_tensor_version_commits_only_after_consumers_resume(monkeypatch):
    module, _ = _load_update_weight_module(monkeypatch)
    events = []
    updater = _make_empty_update_updater(module, events)
    module.dist.barrier = lambda **_: events.append("barrier")
    original_commit = updater._weight_sync_credit.commit_version

    def commit_version(version):
        assert "continue" in events
        events.append("commit")
        original_commit(version)

    updater._weight_sync_credit.commit_version = commit_version
    updater.update_weights()

    assert events[-1] == "commit"
    assert "continue-distributed" in events
    assert updater._weight_sync_credit.snapshot.active_version is None
    assert updater.update_weight_metrics["perf/update_weights_sync_seconds"] >= 0


def test_tensor_resume_failure_poisons_version(monkeypatch):
    module, _ = _load_update_weight_module(monkeypatch)
    events = []
    updater = _make_empty_update_updater(module, events, continue_result="failed-ref")
    module.dist.barrier = lambda **_: events.append("barrier")
    error = RuntimeError("resume failed")

    def ray_get(refs):
        if "failed-ref" in refs:
            raise error
        return refs

    module.ray.get = ray_get
    with pytest.raises(RuntimeError, match="resume failed") as raised:
        updater.update_weights()
    assert raised.value is error
    assert updater._weight_sync_credit.snapshot.active_version == 1
    assert updater._weight_sync_credit.snapshot.failed_reason == "RuntimeError: resume failed"
    with pytest.raises(RuntimeError, match="cannot commit failed weight version"):
        updater._weight_sync_credit.commit_version(1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
