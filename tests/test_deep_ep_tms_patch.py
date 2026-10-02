import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0


def _load_megatron_utils_init(monkeypatch, buffer_cls, tms_impl, module_name):
    deep_ep = types.ModuleType("deep_ep")
    deep_ep.Buffer = buffer_cls
    monkeypatch.setitem(sys.modules, "deep_ep", deep_ep)

    torch_memory_saver_module = types.ModuleType("torch_memory_saver")
    torch_memory_saver_module.torch_memory_saver = types.SimpleNamespace(_impl=tms_impl)
    monkeypatch.setitem(sys.modules, "torch_memory_saver", torch_memory_saver_module)
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))

    package_path = Path(__file__).parents[1] / "slime" / "backends" / "megatron_utils"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)


@pytest.mark.unit
def test_deep_ep_init_restores_original_tms_region(monkeypatch):
    events = []

    class FakeCdll:
        def __init__(self):
            self.interesting_region = False

        def tms_get_interesting_region(self):
            events.append(("get", self.interesting_region))
            return self.interesting_region

        def tms_set_interesting_region(self, enabled):
            self.interesting_region = enabled
            events.append(("set", enabled))

    cdll = FakeCdll()

    class FakeBuffer:
        def __init__(self):
            events.append(("init", cdll.interesting_region))

    tms_impl = types.SimpleNamespace(_binary_wrapper=types.SimpleNamespace(cdll=cdll))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append(("sync", cdll.interesting_region)))

    _load_megatron_utils_init(monkeypatch, FakeBuffer, tms_impl, "_test_megatron_utils_tms_restore")
    FakeBuffer()

    assert events == [
        ("get", False),
        ("set", False),
        ("init", False),
        ("sync", False),
        ("set", False),
    ]
    assert cdll.interesting_region is False


@pytest.mark.unit
def test_deep_ep_init_restores_tms_region_after_failure(monkeypatch):
    events = []

    class FakeCdll:
        interesting_region = True

        def tms_get_interesting_region(self):
            return self.interesting_region

        def tms_set_interesting_region(self, enabled):
            self.interesting_region = enabled
            events.append(("set", enabled))

    cdll = FakeCdll()

    class FailingBuffer:
        def __init__(self):
            events.append(("init", cdll.interesting_region))
            raise RuntimeError("buffer init failed")

    tms_impl = types.SimpleNamespace(_binary_wrapper=types.SimpleNamespace(cdll=cdll))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append(("sync", cdll.interesting_region)))

    _load_megatron_utils_init(monkeypatch, FailingBuffer, tms_impl, "_test_megatron_utils_tms_failure")
    with pytest.raises(RuntimeError, match="buffer init failed"):
        FailingBuffer()

    assert events == [("set", False), ("init", False), ("set", True)]
    assert cdll.interesting_region is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
