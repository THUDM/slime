"""CPU wiring test: rollout engine bringup retries transient Ray unavailability.

``tests/test_retry.py`` covers the generic helper; these tests pin the
*call-site wiring* in ``slime.ray.rollout.start_rollout_servers`` — they fail
if the ``ray.get`` on the engine-init handles is no longer wrapped in the
transient-only retry. Ray/sglang/torch are stubbed at import time so the real
``start_rollout_servers`` runs on CPU with no GPU, Ray cluster, or sglang.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.unit


def _install_rollout_import_stubs(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    torch = types.ModuleType("torch")
    torch.dtype = type("dtype", (), {})
    torch.Size = tuple
    torch.Tensor = type("Tensor", (), {})
    monkeypatch.setitem(sys.modules, "torch", torch)

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

    ray = types.ModuleType("ray")
    ray.get = lambda refs: refs

    def remote(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorate(target):
            return target

        return decorate

    ray.remote = remote

    ray_exceptions = types.ModuleType("ray.exceptions")
    ray_exceptions.ActorUnavailableError = type("ActorUnavailableError", (Exception,), {})
    ray.exceptions = ray_exceptions

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
    monkeypatch.setitem(sys.modules, "ray.exceptions", ray_exceptions)
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


def _bringup_args():
    """Minimal args for the default (non-EPD) start_rollout_servers path."""
    return types.SimpleNamespace(
        rollout_external=False,
        sglang_config=None,
        prefill_num_servers=None,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        hf_checkpoint="/ckpt",
        offload_rollout=False,
        colocate=True,
        debug_train_only=False,
        debug_rollout_only=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
    )


@pytest.fixture
def bringup(rollout, monkeypatch):
    """Drive start_rollout_servers with patched router/engine startup.

    Returns (run, gets): ``run(ray_get)`` installs ``ray_get`` as the module's
    ``ray.get`` binding and runs the bringup; ``gets`` logs each ray.get call.
    """
    init_handles = [object()]
    monkeypatch.setattr(rollout.ServerGroup, "start_engines", lambda self, port_cursors=None: (list(init_handles), {}))
    monkeypatch.setattr(rollout, "_start_router", lambda args, **kwargs: ("127.0.0.1", 10000))
    # Backoff must be instantaneous in tests.
    monkeypatch.setattr(sys.modules["slime.utils.retry"].time, "sleep", lambda _s: None)

    gets = []

    def run(ray_get):
        def logged_get(refs):
            gets.append(refs)
            return ray_get(refs)

        monkeypatch.setattr(rollout.ray, "get", logged_get)
        return rollout.start_rollout_servers(_bringup_args(), pg=None)

    return run, gets


def test_bringup_retries_transient_actor_unavailable(rollout, bringup):
    """The first ray.get raising ActorUnavailableError must not fail bringup."""
    run, gets = bringup

    def flaky_get(refs):
        if len(gets) == 1:
            raise rollout.ActorUnavailableError("heartbeat miss: gRPC UNAVAILABLE")
        return refs

    servers = run(flaky_get)

    assert len(gets) == 2  # first wait raised, retried wait succeeded
    assert gets[0] is gets[1]  # the SAME init handles are re-awaited, not recreated
    assert list(servers) == ["default"]


def test_bringup_propagates_non_transient_error_immediately(rollout, bringup):
    """A real init failure (e.g. CUDA OOM) is never retried or masked."""
    run, gets = bringup

    def failing_get(refs):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run(failing_get)

    assert len(gets) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
