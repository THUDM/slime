"""CPU tests for optional SGLang CUDA-graph memory tags.

Official AMD images ship an sglang whose ``sglang.srt.constants`` module has
``GPU_MEMORY_TYPE_KV_CACHE`` / ``GPU_MEMORY_TYPE_WEIGHTS`` but not
``GPU_MEMORY_TYPE_CUDA_GRAPH``. Importing ``engine_group`` must still succeed,
and KV onload must omit the missing tag.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_GROUP_PATH = REPO_ROOT / "slime" / "backends" / "sglang_utils" / "engine_group.py"

KV_CACHE = "kv_cache"
WEIGHTS = "weights"
CUDA_GRAPH = "cuda_graph"


def _install_import_stubs(monkeypatch, *, include_cuda_graph: bool) -> None:
    """Stub ray / sglang so ``engine_group`` can be loaded without those runtimes."""
    ray_mod = types.ModuleType("ray")
    ray_mod.get = lambda refs: refs
    ray_util = types.ModuleType("ray.util")
    ray_sched = types.ModuleType("ray.util.scheduling_strategies")
    ray_sched.PlacementGroupSchedulingStrategy = type("PlacementGroupSchedulingStrategy", (), {})
    ray_mod.util = ray_util

    sglang = types.ModuleType("sglang")
    sglang.__path__ = []
    sglang_srt = types.ModuleType("sglang.srt")
    sglang_srt.__path__ = []
    constants = types.ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_TYPE_KV_CACHE = KV_CACHE
    constants.GPU_MEMORY_TYPE_WEIGHTS = WEIGHTS
    if include_cuda_graph:
        constants.GPU_MEMORY_TYPE_CUDA_GRAPH = CUDA_GRAPH
    sglang.srt = sglang_srt
    sglang_srt.constants = constants

    sglang_engine = types.ModuleType("slime.backends.sglang_utils.sglang_engine")
    sglang_engine.SGLangEngine = type("SGLangEngine", (), {})

    ray_utils = types.ModuleType("slime.ray.utils")
    ray_utils.NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = []
    ray_utils.add_default_ray_env_vars = lambda env: env

    sglang_config = types.ModuleType("slime.backends.sglang_utils.sglang_config")
    sglang_config.ServerGroupConfig = type("ServerGroupConfig", (), {})

    for name, module in (
        ("ray", ray_mod),
        ("ray.util", ray_util),
        ("ray.util.scheduling_strategies", ray_sched),
        ("sglang", sglang),
        ("sglang.srt", sglang_srt),
        ("sglang.srt.constants", constants),
        ("slime.backends.sglang_utils.sglang_engine", sglang_engine),
        ("slime.backends.sglang_utils.sglang_config", sglang_config),
        ("slime.ray.utils", ray_utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _load_engine_group(monkeypatch, *, include_cuda_graph: bool):
    _install_import_stubs(monkeypatch, include_cuda_graph=include_cuda_graph)
    module_name = f"test_engine_group_memory_tags_{include_cuda_graph}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, ENGINE_GROUP_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _kv_onload_tags(module):
    captured = []
    group = types.SimpleNamespace(onload=lambda tags=None: captured.append(tags) or [])
    module.RolloutServer(server_groups=[group]).onload_kv()
    assert len(captured) == 1
    return captured[0]


@pytest.mark.unit
def test_import_tolerates_missing_cuda_graph_tag(monkeypatch):
    module = _load_engine_group(monkeypatch, include_cuda_graph=False)

    assert module.GPU_MEMORY_TYPE_KV_CACHE == KV_CACHE
    assert module.GPU_MEMORY_TYPE_WEIGHTS == WEIGHTS
    assert module.GPU_MEMORY_TYPE_CUDA_GRAPH is None
    assert _kv_onload_tags(module) == [KV_CACHE]


@pytest.mark.unit
def test_onload_kv_includes_cuda_graph_when_present(monkeypatch):
    module = _load_engine_group(monkeypatch, include_cuda_graph=True)

    assert module.GPU_MEMORY_TYPE_CUDA_GRAPH == CUDA_GRAPH
    assert _kv_onload_tags(module) == [KV_CACHE, CUDA_GRAPH]
