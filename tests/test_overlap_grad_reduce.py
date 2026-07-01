"""CPU unit test for ``configure_overlap_grad_reduce`` idempotency (#2066, #1779).

``config`` persists across ``train()`` calls, so the constant sync funcs must be
set only once -- re-running on a later step previously tripped an
``assert config.no_sync_func is None`` and crashed. The CPU CI image has no
megatron, so we stub the single symbol the helper imports
(``megatron.core.distributed.DistributedDataParallel``) and load the helper file
directly, mirroring ``test_megatron_argument_validation.py``.
"""

import importlib.util
import sys
import types
from pathlib import Path

NUM_GPUS = 0


def _load_helper():
    dist_mod = types.ModuleType("megatron.core.distributed")

    class DistributedDataParallel:  # megatron DDP, stubbed
        pass

    dist_mod.DistributedDataParallel = DistributedDataParallel
    core_mod = types.ModuleType("megatron.core")
    core_mod.distributed = dist_mod
    megatron_mod = types.ModuleType("megatron")
    megatron_mod.core = core_mod
    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.distributed"] = dist_mod

    path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "grad_reduce.py"
    spec = importlib.util.spec_from_file_location("slime_grad_reduce_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, DistributedDataParallel


def _make_model(DDP):
    class Chunk(DDP):
        def no_sync(self): ...

        def start_grad_sync(self): ...

    return [Chunk()]


def test_set_once_then_idempotent():
    helper, DDP = _load_helper()
    model = _make_model(DDP)
    config = types.SimpleNamespace(no_sync_func=None, grad_sync_func=None)
    args = types.SimpleNamespace(overlap_grad_reduce=True, align_grad_reduce=True)

    helper.configure_overlap_grad_reduce(model, config, args)  # step 1
    assert config.no_sync_func is not None
    assert config.grad_sync_func is not None
    no_sync, grad_sync = config.no_sync_func, config.grad_sync_func

    # step 2 must be a no-op, not crash (the #2066 regression)
    helper.configure_overlap_grad_reduce(model, config, args)
    assert config.no_sync_func is no_sync
    assert config.grad_sync_func is grad_sync


def test_skipped_when_disabled_or_not_ddp():
    helper, DDP = _load_helper()
    config = types.SimpleNamespace(no_sync_func=None, grad_sync_func=None)

    # overlap_grad_reduce off -> untouched
    helper.configure_overlap_grad_reduce(
        _make_model(DDP),
        config,
        types.SimpleNamespace(overlap_grad_reduce=False, align_grad_reduce=True),
    )
    assert config.no_sync_func is None

    # model is not DDP-wrapped -> untouched
    helper.configure_overlap_grad_reduce(
        [object()],
        config,
        types.SimpleNamespace(overlap_grad_reduce=True, align_grad_reduce=True),
    )
    assert config.no_sync_func is None
