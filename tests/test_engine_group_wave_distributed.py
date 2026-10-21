import ast
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
@pytest.mark.parametrize("limit", [0, 2, 4])
def test_unbounded_distributed_topology_keeps_one_aggregate_group(monkeypatch, limit):
    module = _load_distributed_update_module(monkeypatch)
    calls = []

    def connect(_args, name, engines, engine_gpu_counts=None):
        calls.append((name, tuple(engines), tuple(engine_gpu_counts)))
        return f"pg-{name}"

    monkeypatch.setattr(module, "connect_rollout_engines_from_distributed", connect)
    engines = ["engine-0", "engine-1"]

    groups = module.connect_rollout_engine_groups_from_distributed(
        Namespace(rollout_num_gpus_per_engine=1),
        "slime",
        engines,
        engine_gpu_counts=[1, 2],
        max_inflight_engine_groups=limit,
    )

    assert calls == [("slime", tuple(engines), (1, 2))]
    assert len(groups) == 1
    assert groups[0].engine_indices == (0, 1)
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
            ["engine-a", "engine-b"],
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
        ["engine-a", "engine-b"],
        engine_gpu_counts=[1, 2],
        max_inflight_engine_groups=2,
        total_engine_groups=4,
        engine_index_offset=2,
    )

    assert calls == [
        ("slime-engine-2", ("engine-a",), (1,)),
        ("slime-engine-3", ("engine-b",), (2,)),
    ]
    assert [group.engine_indices for group in groups] == [(2,), (3,)]


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
        for index in range(4)
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
        max_inflight_engine_groups=3,
        load_format="test",
    )

    assert events == [
        ("launch", 0, "test"),
        ("launch", 1, "test"),
        ("launch", 2, "test"),
        ("work.wait", 0),
        ("work.wait", 1),
        ("work.wait", 2),
        ("ray.get", ("ref-0", "ref-1", "ref-2")),
        ("launch", 3, "test"),
        ("work.wait", 3),
        ("ray.get", ("ref-3",)),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["full", "delta"])
@pytest.mark.parametrize("limit", [0, 2])
@pytest.mark.parametrize("version,flush", [(2, False), (3, True)])
@pytest.mark.parametrize("fail_reload", [False, True])
def test_disk_waves_preserve_cache_policy_and_wait_before_resume(transport, limit, version, flush, fail_reload):
    from slime.observability.communication_timeline import communication_phase
    from slime.utils.engine_group_wave import run_engine_group_waves
    from slime.utils.weight_sync import should_flush_cache

    events = []

    class RemoteMethod:
        def __init__(self, operation, index):
            self.operation, self.index = operation, index

        def remote(self, *args, **kwargs):
            events.append(("call", self.operation, self.index, args, kwargs))
            return self.operation, self.index

    engines = [
        types.SimpleNamespace(
            **{
                name: RemoteMethod(name, index)
                for name in (
                    "pull_weights",
                    "pause_generation",
                    "flush_cache",
                    "update_weights_from_disk",
                    "continue_generation",
                )
            }
        )
        for index in range(5)
    ]

    def wait(refs):
        if isinstance(refs, list):
            events.append(("wait", tuple(refs)))
            if fail_reload and refs and refs[0][0] == "update_weights_from_disk":
                raise RuntimeError("reload failed")
        return refs

    relative, method = (
        ("slime/ray/actor_group.py", "_reload_rollout_weights_from_disk")
        if transport == "full"
        else (
            "slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py",
            "_reload_engines",
        )
    )
    path = REPO_ROOT / relative
    function = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == method
    )
    namespace = {
        "ray": types.SimpleNamespace(get=wait),
        "dist": types.SimpleNamespace(get_rank=lambda: 0, barrier=lambda **kwargs: None),
        "get_gloo_group": lambda: None,
        "run_engine_group_waves": run_engine_group_waves,
        "should_flush_cache": should_flush_cache,
        "communication_phase": communication_phase,
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace
    )
    updater = types.SimpleNamespace(
        args=Namespace(
            offload_rollout=False,
            update_weight_disk_keep_files=True,
            update_weight_max_inflight_engine_groups=limit,
            update_weight_local_checkpoint_dir="local-checkpoint",
            flush_cache_interval=2,
            ci_test=False,
        ),
        _rollout_manager=types.SimpleNamespace(
            get_updatable_engines_and_lock=types.SimpleNamespace(remote=lambda: (engines,))
        ),
        rollout_engines=engines,
        weight_version=version,
        _post_write_hook=None,
        _communication_timeline=None,
        wire_bytes=16,
    )
    arguments = (updater, "published-checkpoint", str(version)) if transport == "full" else (updater,)
    if fail_reload:
        with pytest.raises(RuntimeError, match="reload failed"):
            namespace[method](*arguments)
    else:
        namespace[method](*arguments)

    calls = [event for event in events if event[0] == "call"]
    assert [event[1] for event in calls[:5]] == ["pull_weights"] * 5
    pauses = [event for event in calls if event[1] == "pause_generation"]
    assert len(pauses) == 5
    assert all(event[4] == {"mode": "abort" if flush else "in_place"} for event in pauses)
    assert len([event for event in calls if event[1] == "flush_cache"]) == (5 if flush else 0)
    reloads = [event for event in calls if event[1] == "update_weights_from_disk"]
    assert len(reloads) == ((limit or 5) if fail_reload else 5)
    assert all(
        event[4] == {"model_path": "local-checkpoint", "weight_version": str(version), "flush_cache": flush}
        for event in reloads
    )
    waits = [
        event[1]
        for event in events
        if event[0] == "wait" and event[1] and event[1][0][0] in ("pull_weights", "update_weights_from_disk")
    ]
    assert all(len(refs) <= (limit or 5) for refs in waits)
    resumes = [event for event in calls if event[1] == "continue_generation"]
    assert len(resumes) == (0 if fail_reload else 5)
    if not fail_reload:
        assert events[-7][0] == "wait" and events[-7][1][-1] == ("update_weights_from_disk", 4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
