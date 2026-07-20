import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    return module


def _load_module(monkeypatch, name, path):
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_update_modules(monkeypatch):
    slime_pkg = _package("slime", REPO_ROOT / "slime")
    backends_pkg = _package("slime.backends", REPO_ROOT / "slime" / "backends")
    megatron_utils_pkg = _package("slime.backends.megatron_utils", REPO_ROOT / "slime" / "backends" / "megatron_utils")
    update_weight_pkg = _package(
        "slime.backends.megatron_utils.update_weight",
        REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight",
    )
    slime_utils_pkg = _package("slime.utils", REPO_ROOT / "slime" / "utils")

    ray_mod = types.ModuleType("ray")
    ray_mod.get = lambda refs: refs
    ray_mod.ObjectRef = object
    ray_actor_mod = types.ModuleType("ray.actor")
    ray_actor_mod.ActorHandle = object

    megatron_mod = types.ModuleType("megatron")
    megatron_core_mod = types.ModuleType("megatron.core")
    mpu_mod = types.ModuleType("megatron.core.mpu")
    megatron_core_mod.mpu = mpu_mod

    distributed_utils_mod = types.ModuleType("slime.utils.distributed_utils")
    distributed_utils_mod.get_gloo_group = lambda: None
    distributed_utils_mod.init_process_group = lambda *_args, **_kwargs: None
    http_utils_mod = types.ModuleType("slime.utils.http_utils")
    http_utils_mod._wrap_ipv6 = lambda host: host
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.ParamInfo = object

    megatron_to_hf_mod = types.ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    megatron_to_hf_mod.convert_to_hf = lambda *_args, **_kwargs: []
    common_mod = types.ModuleType("slime.backends.megatron_utils.update_weight.common")
    common_mod.all_gather_param = lambda _name, param: param
    common_mod.named_params_and_buffers = lambda *_args, **_kwargs: []

    sglang_mod = types.ModuleType("slime.backends.megatron_utils.sglang")
    sglang_mod.FlattenedTensorBucket = object
    sglang_mod.MultiprocessingSerializer = object
    iterator_mod = types.ModuleType("slime.backends.megatron_utils.update_weight.hf_weight_iterator_base")
    iterator_mod.HfWeightIteratorBase = object
    expert_routing_mod = types.ModuleType("slime.backends.megatron_utils.update_weight.expert_routing")
    expert_routing_mod.configure_expert_routing = lambda **_kwargs: (None, [])

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = type("tqdm", (), {})

    modules = {
        "slime": slime_pkg,
        "slime.backends": backends_pkg,
        "slime.backends.megatron_utils": megatron_utils_pkg,
        "slime.backends.megatron_utils.update_weight": update_weight_pkg,
        "slime.utils": slime_utils_pkg,
        "ray": ray_mod,
        "ray.actor": ray_actor_mod,
        "megatron": megatron_mod,
        "megatron.core": megatron_core_mod,
        "megatron.core.mpu": mpu_mod,
        "slime.utils.distributed_utils": distributed_utils_mod,
        "slime.utils.http_utils": http_utils_mod,
        "slime.utils.types": types_mod,
        "slime.backends.megatron_utils.megatron_to_hf": megatron_to_hf_mod,
        "slime.backends.megatron_utils.update_weight.common": common_mod,
        "slime.backends.megatron_utils.sglang": sglang_mod,
        "slime.backends.megatron_utils.update_weight.hf_weight_iterator_base": iterator_mod,
        "slime.backends.megatron_utils.update_weight.expert_routing": expert_routing_mod,
        "tqdm": tqdm_mod,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    update_weight_dir = REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight"
    distributed_name = "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    distributed_module = _load_module(
        monkeypatch,
        distributed_name,
        update_weight_dir / "update_weight_from_distributed.py",
    )
    tensor_module = _load_module(
        monkeypatch,
        "slime.backends.megatron_utils.update_weight.update_weight_from_tensor",
        update_weight_dir / "update_weight_from_tensor.py",
    )
    return SimpleNamespace(tensor=tensor_module, distributed=distributed_module)


@pytest.fixture
def update_modules(monkeypatch):
    return _load_update_modules(monkeypatch)


class _RemoteMethod:
    def __init__(self, event_log, name):
        self._event_log = event_log
        self._name = name

    def remote(self):
        self._event_log.append(self._name)
        return self._name


class _Engine:
    def __init__(self, event_log):
        self.pause_generation = _RemoteMethod(event_log, "pause")
        self.flush_cache = _RemoteMethod(event_log, "flush")
        self.continue_generation = _RemoteMethod(event_log, "continue")


def _patch_collectives(monkeypatch, module):
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(module.ray, "get", lambda refs: refs)


def _record_post_process(events, calls):
    def post_process(**kwargs):
        calls.append(kwargs)
        events.append("restore" if kwargs["restore_weights_before_load"] else "post_process")

    return post_process


def _make_tensor_updater(tensor_module, engine, quantization_config=None):
    updater = object.__new__(tensor_module.UpdateWeightFromTensor)
    updater.rank = 0
    updater.weight_version = 0
    updater.rollout_engines = [engine]
    updater.quantization_config = quantization_config
    updater.weights_getter = lambda: {}
    updater._full_param_info_buckets = None
    updater._non_expert_param_info_buckets = None
    updater._expert_transfer_plan = []
    updater._hf_weight_iterator = SimpleNamespace(get_hf_weight_chunks=lambda _weights, **_kwargs: [[]])
    updater._send_hf_params = lambda _chunk: ([], None)
    return updater


@pytest.mark.unit
def test_tensor_weight_update_post_processes_bf16_engines(monkeypatch, update_modules):
    tensor_module = update_modules.tensor
    events = []
    calls = []
    engine = _Engine(events)
    _patch_collectives(monkeypatch, tensor_module)
    monkeypatch.setattr(tensor_module.torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(
        tensor_module,
        "post_process_weights",
        _record_post_process(events, calls),
    )

    updater = _make_tensor_updater(tensor_module, engine)

    updater.update_weights()

    assert events == ["pause", "flush", "post_process", "continue"]
    assert calls == [
        {
            "restore_weights_before_load": False,
            "post_process_quantization": True,
            "rollout_engines": [engine],
        }
    ]


@pytest.mark.unit
def test_tensor_weight_update_preserves_compressed_tensor_processing(monkeypatch, update_modules):
    tensor_module = update_modules.tensor
    events = []
    calls = []
    engine = _Engine(events)
    _patch_collectives(monkeypatch, tensor_module)
    monkeypatch.setattr(tensor_module.torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(
        tensor_module,
        "post_process_weights",
        _record_post_process(events, calls),
    )

    updater = _make_tensor_updater(tensor_module, engine, {"quant_method": "compressed-tensors"})

    updater.update_weights()

    assert events == ["pause", "flush", "restore", "post_process", "continue"]
    assert [(call["restore_weights_before_load"], call["post_process_quantization"]) for call in calls] == [
        (True, False),
        (False, True),
    ]


@pytest.mark.unit
def test_tensor_weight_update_does_not_resume_after_post_process_failure(monkeypatch, update_modules):
    tensor_module = update_modules.tensor
    events = []
    engine = _Engine(events)
    _patch_collectives(monkeypatch, tensor_module)
    monkeypatch.setattr(tensor_module.torch.cuda, "ipc_collect", lambda: None)

    def fail_post_process(**_kwargs):
        raise RuntimeError("post-process failed")

    monkeypatch.setattr(tensor_module, "post_process_weights", fail_post_process)
    updater = _make_tensor_updater(tensor_module, engine)

    with pytest.raises(RuntimeError, match="post-process failed"):
        updater.update_weights()

    assert events == ["pause", "flush"]


@pytest.mark.unit
def test_distributed_weight_update_post_processes_bf16_engines(monkeypatch, update_modules):
    distributed_module = update_modules.distributed
    events = []
    calls = []
    engine = _Engine(events)
    _patch_collectives(monkeypatch, distributed_module)
    monkeypatch.setattr(
        distributed_module,
        "post_process_weights",
        _record_post_process(events, calls),
    )

    updater = object.__new__(distributed_module.UpdateWeightFromDistributed)
    updater.weight_version = 0
    updater.rollout_engines = [engine]
    updater.quantization_config = None
    updater._group_name = "test"
    updater._is_pp_src_rank = False
    updater._send_weights = lambda _pbar: events.append("send")

    updater.update_weights()

    assert events == ["pause", "flush", "send", "post_process", "continue"]
    assert calls == [
        {
            "restore_weights_before_load": False,
            "post_process_quantization": True,
            "rollout_engines": [engine],
        }
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
