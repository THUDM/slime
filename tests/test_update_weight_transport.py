import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]


class _RemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class _Engine:
    def __init__(self):
        self.update_weights_from_distributed = _RemoteMethod()


class _Group:
    def rank(self):
        return 0


def _load_update_weight_module(monkeypatch):
    slime_pkg = types.ModuleType("slime")
    slime_pkg.__path__ = [str(REPO_ROOT / "slime")]
    backends_pkg = types.ModuleType("slime.backends")
    backends_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends")]
    megatron_utils_pkg = types.ModuleType("slime.backends.megatron_utils")
    megatron_utils_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends" / "megatron_utils")]
    update_weight_pkg = types.ModuleType("slime.backends.megatron_utils.update_weight")
    update_weight_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight")]
    utils_pkg = types.ModuleType("slime.utils")
    utils_pkg.__path__ = [str(REPO_ROOT / "slime" / "utils")]

    megatron_mod = types.ModuleType("megatron")
    megatron_core_mod = types.ModuleType("megatron.core")
    megatron_core_mod.mpu = types.ModuleType("megatron.core.mpu")

    ray_mod = types.ModuleType("ray")
    ray_mod.ObjectRef = object
    ray_actor_mod = types.ModuleType("ray.actor")
    ray_actor_mod.ActorHandle = object

    accelerator_mod = types.ModuleType("slime.utils.accelerator")
    distributed_utils_mod = types.ModuleType("slime.utils.distributed_utils")
    distributed_utils_mod.get_gloo_group = lambda: None
    distributed_utils_mod.init_process_group = lambda **kwargs: None
    http_utils_mod = types.ModuleType("slime.utils.http_utils")
    http_utils_mod._wrap_ipv6 = lambda address: address

    megatron_to_hf_mod = types.ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    megatron_to_hf_mod.convert_to_hf = lambda *args, **kwargs: []
    common_mod = types.ModuleType("slime.backends.megatron_utils.update_weight.common")
    common_mod.all_gather_param = lambda *args, **kwargs: None
    common_mod.named_params_and_buffers = lambda *args, **kwargs: []

    fake_modules = {
        "slime": slime_pkg,
        "slime.backends": backends_pkg,
        "slime.backends.megatron_utils": megatron_utils_pkg,
        "slime.backends.megatron_utils.megatron_to_hf": megatron_to_hf_mod,
        "slime.backends.megatron_utils.update_weight": update_weight_pkg,
        "slime.backends.megatron_utils.update_weight.common": common_mod,
        "slime.utils": utils_pkg,
        "slime.utils.accelerator": accelerator_mod,
        "slime.utils.distributed_utils": distributed_utils_mod,
        "slime.utils.http_utils": http_utils_mod,
        "megatron": megatron_mod,
        "megatron.core": megatron_core_mod,
        "ray": ray_mod,
        "ray.actor": ray_actor_mod,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    module_path = (
        REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight" / "update_weight_from_distributed.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_p2p_weight_update_matches_engine_receive_protocol(monkeypatch):
    update_weight_from_distributed = _load_update_weight_module(monkeypatch)
    monkeypatch.setenv("UPDATE_MODE", "p2p-broadcast")
    monkeypatch.setattr(update_weight_from_distributed.dist, "get_global_rank", lambda group, rank: rank)

    sends = []
    monkeypatch.setattr(
        update_weight_from_distributed.dist,
        "send",
        lambda tensor, dst, group, tag: sends.append((tensor, dst, group, tag)),
    )
    monkeypatch.setattr(
        update_weight_from_distributed.dist,
        "broadcast",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected collective broadcast")),
    )

    group = _Group()
    engine = _Engine()
    tensors = [("a", torch.ones(2)), ("b", torch.ones(3))]

    refs = update_weight_from_distributed.update_weights_from_distributed("slime-pp_0", group, 1, [engine], tensors)

    assert refs == engine.update_weights_from_distributed.calls
    assert [(dst, send_group, tag) for _, dst, send_group, tag in sends] == [
        (1, group, 0),
        (1, group, 1),
    ]


def test_default_weight_update_keeps_async_broadcast(monkeypatch):
    update_weight_from_distributed = _load_update_weight_module(monkeypatch)
    monkeypatch.delenv("UPDATE_MODE", raising=False)

    waited = []

    class _Handle:
        def wait(self):
            waited.append(True)

    broadcasts = []
    monkeypatch.setattr(
        update_weight_from_distributed.dist,
        "broadcast",
        lambda tensor, src, group, async_op: broadcasts.append((tensor, src, group, async_op)) or _Handle(),
    )

    group = _Group()
    tensors = [("a", torch.ones(2)), ("b", torch.ones(3))]
    update_weight_from_distributed.update_weights_from_distributed("slime-pp_0", group, 1, [_Engine()], tensors)

    assert [(src, broadcast_group, async_op) for _, src, broadcast_group, async_op in broadcasts] == [
        (0, group, True),
        (0, group, True),
    ]
    assert waited == [True, True]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
