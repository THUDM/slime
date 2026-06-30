import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeFlattenedTensorBucket:
    supports_multi_dtypes = True

    def __init__(self, *, named_tensors=None, flattened_tensor=None, metadata=None):
        if named_tensors is not None:
            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")
            self._flattened_tensor = ("flattened", tuple(name for name, _ in named_tensors))
            self._metadata = tuple(name for name, _ in named_tensors)
            return

        self._flattened_tensor = flattened_tensor
        self._metadata = metadata

    def get_flattened_tensor(self):
        return self._flattened_tensor

    def get_metadata(self):
        return self._metadata


class _FakeMultiprocessingSerializer:
    @staticmethod
    def serialize(value, output_str):
        assert output_str is True
        return value


class _FakeRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return f"ref-{len(self.calls)}"


class _FakeEngine:
    def __init__(self):
        self.update_weights_from_tensor = _FakeRemoteMethod()


def _install_fake_deps(monkeypatch):
    dist_state = types.SimpleNamespace(rank=0, world_size=2, gathered=None, local_object=None)

    slime_pkg = types.ModuleType("slime")
    slime_pkg.__path__ = [str(REPO_ROOT / "slime")]
    slime_backends_pkg = types.ModuleType("slime.backends")
    slime_backends_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends")]
    megatron_utils_pkg = types.ModuleType("slime.backends.megatron_utils")
    megatron_utils_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends" / "megatron_utils")]
    update_weight_pkg = types.ModuleType("slime.backends.megatron_utils.update_weight")
    update_weight_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight")]
    slime_utils_pkg = types.ModuleType("slime.utils")
    slime_utils_pkg.__path__ = [str(REPO_ROOT / "slime" / "utils")]

    dist_mod = types.ModuleType("torch.distributed")

    def gather_object(obj, object_gather_list, dst, group):
        dist_state.local_object = obj
        if object_gather_list is not None:
            object_gather_list[:] = dist_state.gathered(obj)

    dist_mod.get_rank = lambda: dist_state.rank
    dist_mod.get_world_size = lambda group=None: dist_state.world_size
    dist_mod.gather_object = gather_object

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.uint8 = "uint8"
    torch_mod.distributed = dist_mod
    torch_mod.empty = lambda size, dtype, device: {"size": size, "dtype": dtype, "device": device}
    torch_mod.no_grad = lambda: (lambda fn: fn)
    torch_mod.cuda = types.SimpleNamespace(current_device=lambda: "cuda:0", ipc_collect=lambda: None)
    torch_mod.nn = types.SimpleNamespace(Module=object)

    ray_mod = types.ModuleType("ray")
    ray_mod.ObjectRef = object
    ray_actor_mod = types.ModuleType("ray.actor")
    ray_actor_mod.ActorHandle = object

    mpu_mod = types.ModuleType("megatron.core.mpu")
    megatron_mod = types.ModuleType("megatron")
    megatron_core_mod = types.ModuleType("megatron.core")
    megatron_core_mod.mpu = mpu_mod

    sglang_mod = types.ModuleType("slime.backends.megatron_utils.sglang")
    sglang_mod.FlattenedTensorBucket = _FakeFlattenedTensorBucket
    sglang_mod.MultiprocessingSerializer = _FakeMultiprocessingSerializer

    distributed_utils_mod = types.ModuleType("slime.utils.distributed_utils")
    distributed_utils_mod.get_gloo_group = lambda: object()

    update_from_distributed_mod = types.ModuleType(
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    )
    update_from_distributed_mod.connect_rollout_engines_from_distributed = lambda *args, **kwargs: None
    update_from_distributed_mod.disconnect_rollout_engines_from_distributed = lambda *args, **kwargs: None
    update_from_distributed_mod.post_process_weights = lambda *args, **kwargs: None
    update_from_distributed_mod.update_weights_from_distributed = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "slime", slime_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends", slime_backends_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils", megatron_utils_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.update_weight", update_weight_pkg)
    monkeypatch.setitem(sys.modules, "slime.utils", slime_utils_pkg)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.actor", ray_actor_mod)
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.distributed_utils", distributed_utils_mod)
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed",
        update_from_distributed_mod,
    )

    return dist_state


def _load_update_weight_module(monkeypatch):
    dist_state = _install_fake_deps(monkeypatch)

    module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_tensor"
    sys.modules.pop(module_name, None)
    module_path = (
        REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight" / "update_weight_from_tensor.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, dist_state


def test_empty_colocated_bucket_still_participates_in_gather(monkeypatch):
    module, dist_state = _load_update_weight_module(monkeypatch)
    dist_state.gathered = lambda local: [local, []]
    engine = _FakeEngine()

    refs, long_lived_tensors = module._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=3,
    )

    assert dist_state.local_object == []
    assert refs == []
    assert long_lived_tensors == []
    assert engine.update_weights_from_tensor.calls == []


def test_source_rank_pads_empty_colocated_bucket_entries(monkeypatch):
    module, dist_state = _load_update_weight_module(monkeypatch)
    remote_serialized_bucket = {"flattened_tensor": ("remote",), "metadata": ("remote_weight",)}
    dist_state.gathered = lambda local: [local, [remote_serialized_bucket]]
    engine = _FakeEngine()

    refs, long_lived_tensors = module._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=7,
    )

    assert refs == ["ref-1"]
    assert len(long_lived_tensors) == 1
    empty_bucket = long_lived_tensors[0]
    assert empty_bucket["metadata"] == []
    assert empty_bucket["flattened_tensor"] == {"size": 0, "dtype": "uint8", "device": "cuda:0"}

    assert engine.update_weights_from_tensor.calls == [
        {
            "serialized_named_tensors": [empty_bucket, remote_serialized_bucket],
            "load_format": "flattened_bucket",
            "weight_version": "7",
        }
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
