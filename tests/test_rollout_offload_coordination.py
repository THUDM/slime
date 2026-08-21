import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_module(name: str, **attrs):
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules.setdefault(parent, types.ModuleType(parent)), child, module)


def _ensure_import_stubs():
    _ensure_module("wandb")
    _ensure_module("sglang")
    _ensure_module("sglang.srt")
    _ensure_module(
        "sglang.srt.constants",
        GPU_MEMORY_TYPE_CUDA_GRAPH="cuda_graph",
        GPU_MEMORY_TYPE_KV_CACHE="kv_cache",
        GPU_MEMORY_TYPE_WEIGHTS="weights",
    )
    _ensure_module("sglang.srt.server_args", ServerArgs=type("ServerArgs", (), {}))
    _ensure_module("sglang.srt.utils", kill_process_tree=lambda *args, **kwargs: None)


class _Ref:
    def __init__(self, name: str, method: str, kwargs: dict, log: list):
        self.name = name
        self.method = method
        self.kwargs = kwargs
        self.log = log
        self.result = f"{name}:{method}"


class _Remote:
    def __init__(self, name: str, method: str, log: list):
        self._name = name
        self._method = method
        self._log = log

    def remote(self, **kwargs):
        self._log.append(("call", self._name, self._method, kwargs))
        return _Ref(self._name, self._method, kwargs, self._log)


class _Engine:
    def __init__(self, name: str, log: list):
        self.pause_generation = _Remote(name, "pause_generation", log)
        self.flush_cache = _Remote(name, "flush_cache", log)
        self.release_memory_occupation = _Remote(name, "release_memory_occupation", log)
        self.resume_memory_occupation = _Remote(name, "resume_memory_occupation", log)
        self.continue_generation = _Remote(name, "continue_generation", log)


@pytest.fixture
def rollout(monkeypatch):
    _ensure_import_stubs()
    import slime.ray.rollout as rollout_mod

    def get(refs):
        if isinstance(refs, _Ref):
            refs = [refs]
        results = []
        for ref in refs:
            ref.log.append(("ack", ref.name, ref.method, ref.kwargs))
            results.append(ref.result)
        return results

    monkeypatch.setattr(rollout_mod.ray, "get", get)
    return rollout_mod


def _group(rollout, names, log, *, needs_offload=True, extra_engines=()):
    engines = [_Engine(name, log) for name in names]
    engines.extend(extra_engines)
    return rollout.ServerGroup(
        args=types.SimpleNamespace(num_gpus_per_node=8),
        pg=None,
        all_engines=engines,
        num_gpus_per_engine=1,
        num_new_engines=0,
        needs_offload=needs_offload,
    )


def _indices(log, kind, method):
    return [i for i, (k, _name, m, _kwargs) in enumerate(log) if k == kind and m == method]


@pytest.mark.unit
def test_offload_acks_pause_and_flush_before_any_release(rollout):
    log = []
    server = rollout.RolloutServer(
        server_groups=[
            _group(rollout, ["e0", "e1"], log, extra_engines=[None]),
            _group(rollout, ["e2"], log),
        ]
    )

    results = server.offload()

    pause_calls = _indices(log, "call", "pause_generation")
    flush_calls = _indices(log, "call", "flush_cache")
    release_calls = _indices(log, "call", "release_memory_occupation")
    assert len(pause_calls) == len(flush_calls) == len(release_calls) == 3
    assert max(_indices(log, "ack", "pause_generation")) < min(flush_calls)
    assert max(_indices(log, "ack", "flush_cache")) < min(release_calls)
    assert sorted(results) == [
        "e0:release_memory_occupation",
        "e1:release_memory_occupation",
        "e2:release_memory_occupation",
    ]


@pytest.mark.unit
def test_offload_skips_groups_that_do_not_need_offload(rollout):
    log = []
    group = _group(rollout, ["e0"], log, needs_offload=False)

    assert group.pause_generation() == []
    assert group.flush_cache() == []
    assert group.continue_generation() == []
    assert rollout.RolloutServer(server_groups=[group]).offload() == []
    assert log == []


@pytest.mark.unit
def test_onload_kv_resumes_generation_after_kv_restore(rollout):
    log = []
    server = rollout.RolloutServer(server_groups=[_group(rollout, ["e0", "e1"], log)])

    results = server.onload_kv()

    resume_calls = _indices(log, "call", "resume_memory_occupation")
    continue_calls = _indices(log, "call", "continue_generation")
    assert len(resume_calls) == len(continue_calls) == 2
    assert max(_indices(log, "ack", "resume_memory_occupation")) < min(continue_calls)
    assert all(log[i][3] == {"tags": ["kv_cache", "cuda_graph"]} for i in resume_calls)
    assert results == ["e0:resume_memory_occupation", "e1:resume_memory_occupation"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
