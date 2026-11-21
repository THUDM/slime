import ctypes
from types import SimpleNamespace

import pytest

from slime.utils import memory_utils

NUM_GPUS = 0


@pytest.mark.unit
@pytest.mark.parametrize("cuda,hip,initialized", [("12.0", "6.0", True), ("12.0", None, False), (None, None, True)])
def test_stack_reset_does_not_initialize_unsupported_cuda(monkeypatch, cuda, hip, initialized):
    monkeypatch.setattr(memory_utils.torch.version, "cuda", cuda)
    monkeypatch.setattr(memory_utils.torch.version, "hip", hip)
    monkeypatch.setattr(memory_utils.torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(memory_utils, "_cuda_stack_api", lambda: pytest.fail("must not load CUDA"))

    memory_utils.reset_cuda_stack_size()


@pytest.mark.unit
def test_stack_reset_restores_cuda_default(monkeypatch):
    calls = []

    class Driver:
        def cuCtxGetLimit(self, output, limit):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_size_t)).contents.value = 32768
            calls.append(("get", limit))
            return 0

        def cuCtxSetLimit(self, limit, value):
            calls.append(("set", limit, value))
            return 0

    monkeypatch.setattr(memory_utils.torch, "version", SimpleNamespace(cuda="12.0", hip=None))
    monkeypatch.setattr(memory_utils.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(memory_utils.torch.cuda, "synchronize", lambda: calls.append(("sync",)))
    monkeypatch.setattr(memory_utils, "_cuda_stack_api", lambda: Driver())

    memory_utils.reset_cuda_stack_size()

    assert calls == [("sync",), ("get", 0), ("set", 0, 1024)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
