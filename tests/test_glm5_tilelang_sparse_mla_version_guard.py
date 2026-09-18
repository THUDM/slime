import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0


class _PassConfigKey:
    TL_DISABLE_TMA_LOWER = "disable_tma_lower"
    TL_DISABLE_WARP_SPECIALIZED = "disable_warp_specialized"
    TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE = "enable_aggressive_shared_memory_merge"


def _load_sparse_mla_bwd(monkeypatch, tilelang_version):
    captured_jit_options = {}
    tilelang = types.ModuleType("tilelang")
    tilelang.__version__ = tilelang_version
    tilelang.PassConfigKey = _PassConfigKey

    def jit(*jit_args, **jit_kwargs):
        assert not jit_args

        def decorate(function):
            if function.__name__ == "bwd":
                captured_jit_options.update(jit_kwargs)
            return function

        return decorate

    tilelang.jit = jit
    tilelang.language = types.SimpleNamespace(bfloat16="bfloat16", float32="float32", int32="int32")
    monkeypatch.setitem(sys.modules, "tilelang", tilelang)

    module_name = f"tilelang_sparse_mla_bwd_test_{id(tilelang)}"
    source = Path(__file__).parents[1] / "slime_plugins/models/glm5/ops/tilelang_sparse_mla_bwd.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, captured_jit_options


@pytest.mark.parametrize(
    ("tilelang_version", "expected"),
    [
        ("0.1.8", True),
        ("0.1.8.post1", True),
        ("0.1.9.dev0", False),
        ("0.1.9", False),
        ("0.1.10+cu128", False),
        ("0.1.11.dev0", False),
        ("0.1.11", True),
        ("0.1.12", True),
        (None, False),
        ("", False),
        ("unknown", False),
        ("0.0.dev0", False),
    ],
)
def test_aggressive_shared_memory_merge_version_guard(monkeypatch, tilelang_version, expected):
    module, jit_options = _load_sparse_mla_bwd(monkeypatch, tilelang_version)

    assert module._enable_aggressive_shared_memory_merge(tilelang_version) is expected
    assert jit_options["pass_configs"][_PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE] is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
