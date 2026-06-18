"""Unit tests for rollout offload/onload coordination.

``slime.ray.rollout`` imports GPU/runtime dependencies at module scope.  These
tests exercise the CPU-only onload wait contract by installing lightweight
``sys.modules`` stubs before importing the module under test.
"""

import importlib
import sys
import types

import pytest

NUM_GPUS = 0


class _TransientActorUnavailable(Exception):
    pass


class _FakeObjectRef:
    def __init__(self, engine, method):
        self.engine = engine
        self.method = method


class _FakeRemoteMethod:
    def __init__(self, engine, method):
        self._engine = engine
        self._method = method

    def remote(self, *args, **kwargs):
        self._engine.submissions.append((self._method, args, kwargs))
        return _FakeObjectRef(self._engine, self._method)


class FakeEngine:
    def __init__(self, name, log):
        self.name = name
        self.log = log
        self.submissions = []

    def __getattr__(self, method):
        return _FakeRemoteMethod(self, method)


_MISSING = object()


def _install_stub(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _make_fake_ray():
    fake_ray = types.ModuleType("ray")

    def get(refs):
        if isinstance(refs, _FakeObjectRef):
            refs = [refs]
        results = []
        for ref in refs:
            ref.engine.log.append((ref.engine.name, ref.method))
            results.append(None)
        return results

    def remote(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(cls):
            return cls

        return _decorator

    fake_ray.get = get
    fake_ray.remote = remote
    fake_ray.ObjectRef = _FakeObjectRef
    return fake_ray


class _FakeTensor:
    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self


def _import_rollout(monkeypatch):
    monkeypatch.delitem(sys.modules, "slime.ray.rollout", raising=False)

    monkeypatch.setitem(sys.modules, "ray", _make_fake_ray())
    _install_stub(monkeypatch, "ray.exceptions", ActorUnavailableError=_TransientActorUnavailable)
    _install_stub(monkeypatch, "ray.util")
    _install_stub(monkeypatch, "ray.util.scheduling_strategies", PlacementGroupSchedulingStrategy=object)
    _install_stub(monkeypatch, "numpy", ndarray=object)
    _install_stub(
        monkeypatch,
        "torch",
        Tensor=_FakeTensor,
        as_tensor=lambda *args, **kwargs: _FakeTensor(),
        dtype=object,
        float32="float32",
        int="int",
        long="long",
    )
    _install_stub(monkeypatch, "sglang")
    _install_stub(monkeypatch, "sglang.srt")
    _install_stub(
        monkeypatch,
        "sglang.srt.constants",
        GPU_MEMORY_TYPE_CUDA_GRAPH="cuda_graph",
        GPU_MEMORY_TYPE_KV_CACHE="kv_cache",
        GPU_MEMORY_TYPE_WEIGHTS="weights",
    )

    _install_stub(
        monkeypatch, "slime.backends.sglang_utils.external", start_external_rollout_servers=lambda *a, **k: None
    )
    _install_stub(
        monkeypatch,
        "slime.backends.sglang_utils.sglang_config",
        ModelConfig=object,
        ServerGroupConfig=object,
        SglangConfig=object,
    )
    _install_stub(monkeypatch, "slime.backends.sglang_utils.sglang_engine", SGLangEngine=object)
    _install_stub(monkeypatch, "slime.rollout.base_types", call_rollout_fn=lambda *a, **k: None)
    _install_stub(monkeypatch, "slime.utils.dp_schedule", build_dp_schedule=lambda *a, **k: None)
    _install_stub(monkeypatch, "slime.utils.health_monitor", RolloutHealthMonitor=object)
    _install_stub(
        monkeypatch,
        "slime.utils.http_utils",
        _wrap_ipv6=lambda *a, **k: None,
        find_available_port=lambda *a, **k: None,
        get_host_info=lambda *a, **k: None,
        init_http_client=lambda *a, **k: None,
    )
    _install_stub(
        monkeypatch,
        "slime.utils.logging_utils",
        configure_logger=lambda *a, **k: None,
        init_tracking=lambda *a, **k: None,
    )
    _install_stub(
        monkeypatch,
        "slime.utils.metric_utils",
        compute_pass_rate=lambda *a, **k: None,
        compute_rollout_step=lambda *a, **k: None,
        compute_statistics=lambda *a, **k: None,
        dict_add_prefix=lambda *a, **k: None,
        has_repetition=lambda *a, **k: None,
    )
    _install_stub(
        monkeypatch, "slime.utils.misc", Box=object, group_by=lambda *a, **k: None, load_function=lambda *a, **k: None
    )
    _install_stub(monkeypatch, "slime.utils.types", Sample=object)
    _install_stub(monkeypatch, "slime.ray.rollout_validation", validate_server_group_gpu_indices=lambda *a, **k: None)
    _install_stub(
        monkeypatch,
        "slime.ray.utils",
        Lock=object,
        NOSET_VISIBLE_DEVICES_ENV_VARS_LIST=[],
        add_default_ray_env_vars=lambda env_vars=None: env_vars or {},
    )

    return importlib.import_module("slime.ray.rollout")


@pytest.fixture
def rollout(monkeypatch):
    ray_pkg_before = sys.modules.get("slime.ray")
    previous_rollout_attr = getattr(ray_pkg_before, "rollout", _MISSING) if ray_pkg_before is not None else _MISSING

    module = _import_rollout(monkeypatch)
    try:
        yield module
    finally:
        ray_pkg = sys.modules.get("slime.ray")
        if ray_pkg is not None:
            if previous_rollout_attr is _MISSING:
                if getattr(ray_pkg, "rollout", None) is module:
                    delattr(ray_pkg, "rollout")
            else:
                ray_pkg.rollout = previous_rollout_attr


def _make_group(rollout, engines, *, needs_offload=True):
    args = types.SimpleNamespace(num_gpus_per_node=8)
    return rollout.ServerGroup(
        args=args,
        pg=None,
        all_engines=list(engines),
        num_gpus_per_engine=1,
        num_new_engines=0,
        needs_offload=needs_offload,
    )


@pytest.mark.unit
@pytest.mark.parametrize("method_name", ["onload", "onload_weights", "onload_kv"])
def test_onload_waits_retry_transient_actor_unavailable_without_resubmitting(rollout, monkeypatch, method_name):
    monkeypatch.setattr(rollout.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(rollout.random, "random", lambda: 0.0)

    log = []
    engines = [FakeEngine("e0", log), FakeEngine("e1", log)]
    group = _make_group(rollout, engines)
    server = rollout.RolloutServer(server_groups=[group])
    original_get = rollout.ray.get
    get_calls = []

    def flaky_get(refs):
        refs = list(refs)
        get_calls.append([(id(ref), ref.engine.name, ref.method) for ref in refs])
        if len(get_calls) == 1:
            raise rollout.ActorUnavailableError("actor direct RPC temporarily unavailable")
        return original_get(refs)

    monkeypatch.setattr(rollout.ray, "get", flaky_get)

    getattr(server, method_name)()

    expected_tags = {
        "onload": None,
        "onload_weights": ["weights"],
        "onload_kv": ["kv_cache", "cuda_graph"],
    }[method_name]
    assert len(get_calls) == 2
    assert get_calls[0] == get_calls[1]
    assert log == [("e0", "resume_memory_occupation"), ("e1", "resume_memory_occupation")]
    assert [engine.submissions for engine in engines] == [
        [("resume_memory_occupation", (), {"tags": expected_tags})],
        [("resume_memory_occupation", (), {"tags": expected_tags})],
    ]


@pytest.mark.unit
def test_onload_wait_does_not_retry_non_actor_unavailable_errors(rollout, monkeypatch):
    log = []
    engines = [FakeEngine("e0", log)]
    group = _make_group(rollout, engines)
    server = rollout.RolloutServer(server_groups=[group])
    get_calls = []

    def failing_get(refs):
        refs = list(refs)
        get_calls.append([(ref.engine.name, ref.method) for ref in refs])
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(rollout.ray, "get", failing_get)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        server.onload_weights()

    assert get_calls == [[("e0", "resume_memory_occupation")]]
    assert log == []
    assert engines[0].submissions == [("resume_memory_occupation", (), {"tags": ["weights"]})]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
