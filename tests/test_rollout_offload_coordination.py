import importlib
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _ObjectRef:
    def __init__(self, engine, method: str, result=None):
        self.engine = engine
        self.method = method
        self.result = result


class _RemoteMethod:
    def __init__(self, engine, method: str):
        self.engine = engine
        self.method = method

    def remote(self, **kwargs):
        self.engine.log.append(("call", self.engine.name, self.method, kwargs))
        return _ObjectRef(self.engine, self.method, result=f"{self.engine.name}:{self.method}")


class _Engine:
    def __init__(self, name: str, log: list):
        self.name = name
        self.log = log

    def __getattr__(self, method: str):
        return _RemoteMethod(self, method)


def _install_rollout_import_stubs(monkeypatch):
    numpy = types.ModuleType("numpy")
    numpy.ndarray = type("ndarray", (), {})
    monkeypatch.setitem(sys.modules, "numpy", numpy)

    torch = types.ModuleType("torch")
    dtype = type("dtype", (), {})
    torch.dtype = dtype
    torch.Size = tuple
    torch.Tensor = type("Tensor", (), {})
    torch.long = dtype()
    torch.int = dtype()
    torch.float32 = dtype()
    torch.int32 = dtype()
    monkeypatch.setitem(sys.modules, "torch", torch)

    external = types.ModuleType("slime.backends.sglang_utils.external")
    external.start_external_rollout_servers = lambda *args, **kwargs: ({}, [])
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.external", external)

    sglang_config = types.ModuleType("slime.backends.sglang_utils.sglang_config")
    sglang_config.ModelConfig = type("ModelConfig", (), {})
    sglang_config.ServerGroupConfig = type("ServerGroupConfig", (), {})
    sglang_config.SglangConfig = type("SglangConfig", (), {})
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.sglang_config", sglang_config)

    http_utils = types.ModuleType("slime.utils.http_utils")
    http_utils._wrap_ipv6 = lambda host: host
    http_utils.find_available_port = lambda base_port: base_port
    http_utils.get_host_info = lambda: ("localhost", "127.0.0.1")
    http_utils.init_http_client = lambda *args, **kwargs: None
    http_utils.is_port_available = lambda port: True
    monkeypatch.setitem(sys.modules, "slime.utils.http_utils", http_utils)

    logging_utils = types.ModuleType("slime.utils.logging_utils")
    logging_utils.configure_logger = lambda *args, **kwargs: None
    logging_utils.finish_tracking = lambda *args, **kwargs: None
    logging_utils.init_tracking = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils)

    base_types = types.ModuleType("slime.rollout.base_types")
    base_types.call_rollout_fn = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "slime.rollout.base_types", base_types)

    data = types.ModuleType("slime.utils.data")
    data.get_source = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "slime.utils.data", data)

    dp_schedule = types.ModuleType("slime.utils.dp_schedule")
    dp_schedule.build_dp_schedule = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "slime.utils.dp_schedule", dp_schedule)

    health_monitor = types.ModuleType("slime.utils.health_monitor")

    class _RolloutHealthMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    health_monitor.RolloutHealthMonitor = _RolloutHealthMonitor
    monkeypatch.setitem(sys.modules, "slime.utils.health_monitor", health_monitor)

    metric_utils = types.ModuleType("slime.utils.metric_utils")
    metric_utils.compute_pass_rate = lambda *args, **kwargs: None
    metric_utils.compute_rollout_step = lambda *args, **kwargs: None
    metric_utils.compute_statistics = lambda *args, **kwargs: None
    metric_utils.dict_add_prefix = lambda prefix, values: {f"{prefix}{key}": value for key, value in values.items()}
    metric_utils.has_repetition = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, "slime.utils.metric_utils", metric_utils)

    misc = types.ModuleType("slime.utils.misc")
    misc.Box = dict
    misc.group_by = lambda values, key: {}
    misc.load_function = lambda path: None
    monkeypatch.setitem(sys.modules, "slime.utils.misc", misc)

    types_mod = types.ModuleType("slime.utils.types")
    types_mod.Sample = type("Sample", (), {})
    monkeypatch.setitem(sys.modules, "slime.utils.types", types_mod)

    rollout_validation = types.ModuleType("slime.ray.rollout_validation")
    rollout_validation.validate_server_group_gpu_indices = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "slime.ray.rollout_validation", rollout_validation)

    ray_utils = types.ModuleType("slime.ray.utils")
    ray_utils.NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = []
    ray_utils.add_default_ray_env_vars = lambda env_vars=None: env_vars or {}
    ray_utils.Lock = types.SimpleNamespace(
        options=lambda *args, **kwargs: types.SimpleNamespace(remote=lambda *args, **kwargs: object())
    )
    monkeypatch.setitem(sys.modules, "slime.ray.utils", ray_utils)

    ray = types.ModuleType("ray")

    def get(refs):
        if isinstance(refs, _ObjectRef):
            refs = [refs]
        results = []
        for ref in refs:
            ref.engine.log.append(("get", ref.engine.name, ref.method, {}))
            results.append(ref.result)
        return results

    def remote(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorate(target):
            return target

        return decorate

    ray.get = get
    ray.remote = remote

    ray_util = types.ModuleType("ray.util")
    scheduling = types.ModuleType("ray.util.scheduling_strategies")
    scheduling.PlacementGroupSchedulingStrategy = object
    ray_util.scheduling_strategies = scheduling
    ray.util = ray_util

    sglang = types.ModuleType("sglang")
    sglang_srt = types.ModuleType("sglang.srt")
    constants = types.ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
    constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"

    sglang_engine = types.ModuleType("slime.backends.sglang_utils.sglang_engine")
    sglang_engine.SGLangEngine = object

    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", scheduling)
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", sglang_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.constants", constants)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.sglang_engine", sglang_engine)
    sys.modules.pop("slime.ray.rollout", None)


@pytest.fixture
def rollout(monkeypatch):
    _install_rollout_import_stubs(monkeypatch)
    return importlib.import_module("slime.ray.rollout")


def _server_group(rollout, engines, *, needs_offload=True):
    return rollout.ServerGroup(
        args=types.SimpleNamespace(num_gpus_per_node=8, debug_train_only=False),
        pg=None,
        all_engines=list(engines),
        num_gpus_per_engine=1,
        num_new_engines=0,
        needs_offload=needs_offload,
    )


@pytest.mark.unit
def test_offload_waits_for_all_groups_to_pause_and_flush_before_release(rollout):
    log = []
    server = rollout.RolloutServer(
        server_groups=[
            _server_group(rollout, [_Engine("e0", log), _Engine("e1", log), None]),
            _server_group(rollout, [_Engine("e2", log)]),
        ],
    )

    results = server.offload()

    assert log == [
        ("call", "e0", "pause_generation", {}),
        ("call", "e1", "pause_generation", {}),
        ("call", "e2", "pause_generation", {}),
        ("get", "e0", "pause_generation", {}),
        ("get", "e1", "pause_generation", {}),
        ("get", "e2", "pause_generation", {}),
        ("call", "e0", "flush_cache", {}),
        ("call", "e1", "flush_cache", {}),
        ("call", "e2", "flush_cache", {}),
        ("get", "e0", "flush_cache", {}),
        ("get", "e1", "flush_cache", {}),
        ("get", "e2", "flush_cache", {}),
        ("call", "e0", "release_memory_occupation", {}),
        ("call", "e1", "release_memory_occupation", {}),
        ("call", "e2", "release_memory_occupation", {}),
        ("get", "e0", "release_memory_occupation", {}),
        ("get", "e1", "release_memory_occupation", {}),
        ("get", "e2", "release_memory_occupation", {}),
    ]
    assert results == [
        "e0:release_memory_occupation",
        "e1:release_memory_occupation",
        "e2:release_memory_occupation",
    ]


@pytest.mark.unit
def test_offload_and_continue_skip_non_overlapping_groups(rollout):
    log = []
    group = _server_group(rollout, [_Engine("e0", log)], needs_offload=False)

    assert group.pause_generation() == []
    assert group.flush_cache() == []
    assert group.continue_generation() == []
    assert log == []


@pytest.mark.unit
def test_onload_kv_resumes_generation_after_kv_restore(rollout):
    log = []
    group = _server_group(rollout, [_Engine("e0", log), _Engine("e1", log)])
    server = rollout.RolloutServer(server_groups=[group])

    results = server.onload_kv()

    assert results == ["e0:resume_memory_occupation", "e1:resume_memory_occupation"]
    assert log == [
        (
            "call",
            "e0",
            "resume_memory_occupation",
            {"tags": ["kv_cache", "cuda_graph"]},
        ),
        (
            "call",
            "e1",
            "resume_memory_occupation",
            {"tags": ["kv_cache", "cuda_graph"]},
        ),
        ("get", "e0", "resume_memory_occupation", {}),
        ("get", "e1", "resume_memory_occupation", {}),
        ("call", "e0", "continue_generation", {}),
        ("call", "e1", "continue_generation", {}),
        ("get", "e0", "continue_generation", {}),
        ("get", "e1", "continue_generation", {}),
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
