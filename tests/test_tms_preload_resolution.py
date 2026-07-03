"""Unit tests for actor_group._resolve_tms_preload_lib.

Loads slime.ray.actor_group with heavy deps (ray) stubbed, then exercises the
torch_memory_saver preload .so selection: resolver-first, then a CUDA-major
keyed existence fallback, and finally a dlopen probe. Mirrors the module-stub
pattern in test_megatron_argument_validation.py.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_actor_group(monkeypatch):
    ray_mod = types.ModuleType("ray")
    pg_mod = types.ModuleType("ray.util.placement_group")
    sched_mod = types.ModuleType("ray.util.scheduling_strategies")
    pg_mod.PlacementGroup = object
    sched_mod.PlacementGroupSchedulingStrategy = object
    utils_mod = types.ModuleType("slime.ray.utils")
    utils_mod.NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = []
    utils_mod.add_default_ray_env_vars = lambda *a, **k: None

    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", pg_mod)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", sched_mod)
    monkeypatch.setitem(sys.modules, "slime.ray.utils", utils_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "ray" / "actor_group.py"
    module_name = "test_actor_group_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_tms(tmp_path, *, with_resolver, resolver_ret=None, resolver_raises=None):
    """A fake torch_memory_saver module rooted so its .so dir is tmp_path."""
    pkg = tmp_path / "torch_memory_saver"
    pkg.mkdir(exist_ok=True)
    tms = types.ModuleType("torch_memory_saver")
    tms.__file__ = str(pkg / "__init__.py")
    utils = types.ModuleType("torch_memory_saver.utils")
    if with_resolver:

        def get_binary_path_from_package(stem):
            if resolver_raises is not None:
                raise resolver_raises
            return resolver_ret

        utils.get_binary_path_from_package = get_binary_path_from_package
    return tms, utils


def _touch(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"")
    return str(p)


def test_prefers_library_resolver(monkeypatch, tmp_path):
    ag = load_actor_group(monkeypatch)
    want = str(tmp_path / "torch_memory_saver_hook_mode_preload_cu13.abi3.so")
    tms, utils = _fake_tms(tmp_path, with_resolver=True, resolver_ret=want)
    monkeypatch.setitem(sys.modules, "torch_memory_saver.utils", utils)
    assert ag._resolve_tms_preload_lib(tms) == want


def test_resolver_error_propagates(monkeypatch, tmp_path):
    # A resolver that exists but raises must NOT be masked by the fallback.
    ag = load_actor_group(monkeypatch)
    tms, utils = _fake_tms(tmp_path, with_resolver=True, resolver_raises=RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "torch_memory_saver.utils", utils)
    with pytest.raises(RuntimeError, match="boom"):
        ag._resolve_tms_preload_lib(tms)


def test_fallback_keys_on_cuda_major_by_existence(monkeypatch, tmp_path):
    # No resolver: pick cu<major> by existence (no dlopen), so it works on a
    # GPU-less driver. torch.version.cuda drives the major.
    ag = load_actor_group(monkeypatch)
    tms, _ = _fake_tms(tmp_path, with_resolver=False)
    monkeypatch.delitem(sys.modules, "torch_memory_saver.utils", raising=False)
    torch_mod = types.ModuleType("torch")
    torch_mod.version = types.SimpleNamespace(cuda="13.0")
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    # Only the cu13 build exists; the fallback must select it by name.
    _touch(tmp_path, "torch_memory_saver_hook_mode_preload_cu12.abi3.so")
    want = _touch(tmp_path, "torch_memory_saver_hook_mode_preload_cu13.abi3.so")
    assert ag._resolve_tms_preload_lib(tms) == want


def test_fallback_cuda12_selects_cu12(monkeypatch, tmp_path):
    # CUDA 12 build: the cu12 variant is selected by existence, unchanged.
    ag = load_actor_group(monkeypatch)
    tms, _ = _fake_tms(tmp_path, with_resolver=False)
    monkeypatch.delitem(sys.modules, "torch_memory_saver.utils", raising=False)
    torch_mod = types.ModuleType("torch")
    torch_mod.version = types.SimpleNamespace(cuda="12.4")
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    want = _touch(tmp_path, "torch_memory_saver_hook_mode_preload_cu12.abi3.so")
    _touch(tmp_path, "torch_memory_saver_hook_mode_preload_cu13.abi3.so")
    assert ag._resolve_tms_preload_lib(tms) == want
