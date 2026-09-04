import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NUM_GPUS = 0


def _load_distributed_update_module(monkeypatch):
    slime_pkg = types.ModuleType("slime")
    slime_pkg.__path__ = [str(REPO_ROOT / "slime")]
    slime_utils_pkg = types.ModuleType("slime.utils")
    slime_utils_pkg.__path__ = [str(REPO_ROOT / "slime" / "utils")]
    slime_backends_pkg = types.ModuleType("slime.backends")
    slime_backends_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends")]
    megatron_utils_pkg = types.ModuleType("slime.backends.megatron_utils")
    megatron_utils_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends" / "megatron_utils")]
    update_weight_pkg = types.ModuleType("slime.backends.megatron_utils.update_weight")
    update_weight_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight")]

    ray_mod = types.ModuleType("ray")
    ray_mod.ObjectRef = object
    ray_mod.get = lambda refs: refs
    ray_actor_mod = types.ModuleType("ray.actor")
    ray_actor_mod.ActorHandle = object

    dist_mod = types.ModuleType("torch.distributed")
    dist_mod.ProcessGroup = object
    dist_mod.Work = object
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.nn = types.SimpleNamespace(Module=object)
    torch_mod.no_grad = lambda: (lambda function: function)
    torch_mod.distributed = dist_mod

    megatron_mod = types.ModuleType("megatron")
    megatron_core_mod = types.ModuleType("megatron.core")
    mpu_mod = types.ModuleType("megatron.core.mpu")
    megatron_core_mod.mpu = mpu_mod

    accelerator_mod = types.ModuleType("slime.utils.accelerator")
    distributed_utils_mod = types.ModuleType("slime.utils.distributed_utils")
    distributed_utils_mod.get_gloo_group = lambda: object()
    distributed_utils_mod.init_process_group = lambda **kwargs: object()
    http_utils_mod = types.ModuleType("slime.utils.http_utils")
    http_utils_mod._wrap_ipv6 = lambda address: address
    megatron_to_hf_mod = types.ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    megatron_to_hf_mod.convert_to_hf = lambda *args, **kwargs: []
    common_mod = types.ModuleType("slime.backends.megatron_utils.update_weight.common")
    common_mod.all_gather_param = lambda _name, param: param
    common_mod.named_params_and_buffers = lambda *_args, **_kwargs: []
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = object

    fake_modules = {
        "slime": slime_pkg,
        "slime.utils": slime_utils_pkg,
        "slime.backends": slime_backends_pkg,
        "slime.backends.megatron_utils": megatron_utils_pkg,
        "slime.backends.megatron_utils.update_weight": update_weight_pkg,
        "ray": ray_mod,
        "ray.actor": ray_actor_mod,
        "torch": torch_mod,
        "torch.distributed": dist_mod,
        "megatron": megatron_mod,
        "megatron.core": megatron_core_mod,
        "megatron.core.mpu": mpu_mod,
        "slime.utils.accelerator": accelerator_mod,
        "slime.utils.distributed_utils": distributed_utils_mod,
        "slime.utils.http_utils": http_utils_mod,
        "slime.backends.megatron_utils.megatron_to_hf": megatron_to_hf_mod,
        "slime.backends.megatron_utils.update_weight.common": common_mod,
        "tqdm": tqdm_mod,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    sys.modules.pop(module_name, None)
    module_path = (
        REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight" / "update_weight_from_distributed.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.parametrize("limit", [0, 3, 8])
def test_unbounded_distributed_topology_keeps_one_aggregate_group(monkeypatch, limit):
    module = _load_distributed_update_module(monkeypatch)
    calls = []

    def connect(_args, name, engines, engine_gpu_counts=None):
        calls.append((name, tuple(engines), tuple(engine_gpu_counts)))
        return f"pg-{name}"

    monkeypatch.setattr(module, "connect_rollout_engines_from_distributed", connect)
    engines = ["engine-0", "engine-1", "engine-2"]

    groups = module.connect_rollout_engine_groups_from_distributed(
        Namespace(rollout_num_gpus_per_engine=1),
        "slime",
        engines,
        engine_gpu_counts=[1, 2, 3],
        max_inflight_engine_groups=limit,
    )

    assert calls == [("slime", tuple(engines), (1, 2, 3))]
    assert len(groups) == 1
    assert groups[0].engine_indices == (0, 1, 2)
    assert groups[0].rollout_engines == tuple(engines)


@pytest.mark.unit
def test_partial_group_setup_is_torn_down_on_failure(monkeypatch):
    module = _load_distributed_update_module(monkeypatch)
    disconnected = []

    def connect(_args, name, _engines, engine_gpu_counts=None):
        assert engine_gpu_counts is not None
        if name.endswith("-1"):
            raise RuntimeError("engine setup failed")
        return f"pg-{name}"

    monkeypatch.setattr(module, "connect_rollout_engines_from_distributed", connect)
    monkeypatch.setattr(
        module,
        "disconnect_rollout_engine_groups_from_distributed",
        lambda groups: disconnected.extend(groups),
    )

    with pytest.raises(RuntimeError, match="engine setup failed"):
        module.connect_rollout_engine_groups_from_distributed(
            Namespace(rollout_num_gpus_per_engine=1),
            "slime",
            ["engine-a", "engine-b", "engine-c"],
            max_inflight_engine_groups=1,
        )

    assert len(disconnected) == 1
    assert disconnected[0].group_name == "slime-engine-0"


@pytest.mark.unit
def test_distributed_launch_starts_all_bucket_broadcasts_asynchronously(monkeypatch):
    module = _load_distributed_update_module(monkeypatch)
    remote_calls = []
    broadcasts = []

    class RemoteMethod:
        def remote(self, **kwargs):
            remote_calls.append(kwargs)
            return "engine-ref"

    class Engine:
        update_weights_from_distributed = RemoteMethod()

    class Tensor:
        def __init__(self, name, dtype, shape):
            self.data = f"data-{name}"
            self.dtype = dtype
            self.shape = shape

    def broadcast(data, source, *, group, async_op):
        broadcasts.append((data, source, group, async_op))
        return f"work-{data}"

    monkeypatch.setattr(module.dist, "broadcast", broadcast, raising=False)
    named_tensors = [
        ("weight-a", Tensor("a", "bf16", (2, 4))),
        ("weight-b", Tensor("b", "fp32", (3,))),
    ]

    refs, works = module.launch_weights_from_distributed(
        "slime-engine-0",
        "pg-0",
        17,
        [Engine()],
        named_tensors,
        load_format="flattened_bucket",
    )

    assert refs == ["engine-ref"]
    assert works == ["work-data-a", "work-data-b"]
    assert broadcasts == [
        ("data-a", 0, "pg-0", True),
        ("data-b", 0, "pg-0", True),
    ]
    assert remote_calls == [
        {
            "names": ["weight-a", "weight-b"],
            "dtypes": ["bf16", "fp32"],
            "shapes": [(2, 4), (3,)],
            "group_name": "slime-engine-0",
            "weight_version": "17",
            "load_format": "flattened_bucket",
        }
    ]


@pytest.mark.unit
def test_bounded_distributed_topology_builds_one_group_per_engine(monkeypatch):
    module = _load_distributed_update_module(monkeypatch)
    calls = []

    def connect(_args, name, engines, engine_gpu_counts=None):
        calls.append((name, tuple(engines), tuple(engine_gpu_counts)))
        return f"pg-{name}"

    monkeypatch.setattr(module, "connect_rollout_engines_from_distributed", connect)

    groups = module.connect_rollout_engine_groups_from_distributed(
        Namespace(rollout_num_gpus_per_engine=1),
        "slime",
        ["engine-a", "engine-b", "engine-c"],
        engine_gpu_counts=[1, 3, 2],
        max_inflight_engine_groups=2,
        total_engine_groups=5,
        engine_index_offset=2,
    )

    assert calls == [
        ("slime-engine-2", ("engine-a",), (1,)),
        ("slime-engine-3", ("engine-b",), (3,)),
        ("slime-engine-4", ("engine-c",), (2,)),
    ]
    assert [group.engine_indices for group in groups] == [(2,), (3,), (4,)]


@pytest.mark.unit
def test_distributed_wave_waits_before_admitting_next_engines(monkeypatch):
    module = _load_distributed_update_module(monkeypatch)
    events = []

    class Work:
        def __init__(self, engine_index):
            self.engine_index = engine_index

        def wait(self):
            events.append(("work.wait", self.engine_index))

    groups = [
        module.DistributedWeightUpdateGroup(
            engine_indices=(index,),
            group_name=f"group-{index}",
            process_group=f"pg-{index}",
            rollout_engines=(f"engine-{index}",),
        )
        for index in range(5)
    ]

    def launch(group_name, _group, _version, _engines, _tensors, load_format=None):
        engine_index = int(group_name.rsplit("-", 1)[1])
        events.append(("launch", engine_index, load_format))
        return [f"ref-{engine_index}"], [Work(engine_index)]

    monkeypatch.setattr(module, "launch_weights_from_distributed", launch)
    monkeypatch.setattr(module.ray, "get", lambda refs: events.append(("ray.get", tuple(refs))))

    module.update_weights_in_engine_group_waves(
        groups,
        weight_version=9,
        converted_named_tensors=[],
        max_inflight_engine_groups=2,
        load_format="test",
    )

    assert events == [
        ("launch", 0, "test"),
        ("launch", 1, "test"),
        ("work.wait", 0),
        ("work.wait", 1),
        ("ray.get", ("ref-0", "ref-1")),
        ("launch", 2, "test"),
        ("launch", 3, "test"),
        ("work.wait", 2),
        ("work.wait", 3),
        ("ray.get", ("ref-2", "ref-3")),
        ("launch", 4, "test"),
        ("work.wait", 4),
        ("ray.get", ("ref-4",)),
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
