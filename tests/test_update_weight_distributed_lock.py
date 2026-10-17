import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_NAME = "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
MODULE_PATH = (
    REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight" / "update_weight_from_distributed.py"
)


class _FakeActorMethod:
    """Actor-method handle: .remote() runs the method inline and returns its result.
    (Real ray returns an ObjectRef; the fake ray.get below is identity, so values pass through.)"""

    def __init__(self, fn):
        self._fn = fn

    def remote(self):
        return self._fn()


class _FakeLock:
    """Stands in for the ray Lock actor (slime/ray/utils.py): first acquire wins,
    release asserts the lock is held — so a release without a successful acquire
    fails the test, mirroring the real actor."""

    def __init__(self, events):
        self._locked = False
        self.events = events
        self.acquire = _FakeActorMethod(self._acquire)
        self.release = _FakeActorMethod(self._release)

    def _acquire(self):
        self.events.append("acquire")
        if not self._locked:
            self._locked = True
            return True
        return False

    def _release(self):
        assert self._locked, "Lock is not acquired, cannot release."
        self._locked = False
        self.events.append("release")


class _FakePbar:
    """Records tqdm update calls into the shared event log."""

    def __init__(self, events):
        self.events = events
        self.updates = []

    def update(self, n):
        self.updates.append(n)
        self.events.append("pbar")


def _install_fake_deps(monkeypatch):
    """Seed sys.modules with fakes for every non-stdlib import of the target module.

    Package fakes carry __path__ to the real directories (relative-import resolution);
    leaf fakes carry the attributes the target module imports. The target module itself
    is NOT faked — _load_module loads the real file.
    """
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

    # imported by the target module; not exercised by these tests, kept callable
    accelerator_mod = types.ModuleType("slime.utils.accelerator")
    accelerator_mod.current_device = lambda: "cuda:0"
    accelerator_mod.weight_update_backend = lambda *args, **kwargs: "nccl"

    distributed_utils_mod = types.ModuleType("slime.utils.distributed_utils")
    distributed_utils_mod.get_gloo_group = lambda: object()
    distributed_utils_mod.init_process_group = lambda **kwargs: object()

    http_utils_mod = types.ModuleType("slime.utils.http_utils")
    http_utils_mod._wrap_ipv6 = lambda host: host

    megatron_to_hf_mod = types.ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    megatron_to_hf_mod.convert_to_hf = lambda *args, **kwargs: []

    common_mod = types.ModuleType("slime.backends.megatron_utils.update_weight.common")
    common_mod.all_gather_param = lambda name, param: param
    common_mod.named_params_and_buffers = lambda args, model: iter([])

    dist_mod = types.ModuleType("torch.distributed")

    torch_mod = types.ModuleType("torch")
    torch_mod.distributed = dist_mod  # binds `import torch.distributed as dist`
    torch_mod.no_grad = lambda: (lambda fn: fn)  # applied at class-body exec (@torch.no_grad)

    ray_mod = types.ModuleType("ray")
    ray_mod.ObjectRef = object
    ray_mod.get = lambda value: value
    ray_actor_mod = types.ModuleType("ray.actor")
    ray_actor_mod.ActorHandle = object

    mpu_mod = types.ModuleType("megatron.core.mpu")
    megatron_core_mod = types.ModuleType("megatron.core")
    megatron_core_mod.mpu = mpu_mod
    megatron_mod = types.ModuleType("megatron")

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "slime", slime_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends", slime_backends_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils", megatron_utils_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.update_weight", update_weight_pkg)
    monkeypatch.setitem(sys.modules, "slime.utils", slime_utils_pkg)
    monkeypatch.setitem(sys.modules, "slime.utils.accelerator", accelerator_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.distributed_utils", distributed_utils_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.http_utils", http_utils_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.megatron_to_hf", megatron_to_hf_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.update_weight.common", common_mod)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.actor", ray_actor_mod)
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu_mod)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)


def _load_module(monkeypatch):
    """Load the real update_weight_from_distributed.py under its dotted module name,
    so its relative imports resolve against the fake packages above."""
    _install_fake_deps(monkeypatch)

    sys.modules.pop(MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE_NAME, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_updater(module, lock):
    """Real UpdateWeightFromDistributed plus the attributes connect_rollout_engines
    sets on a PP source rank."""
    updater = module.UpdateWeightFromDistributed(
        types.SimpleNamespace(),
        [],
        lambda: {},
        model_name="test-model",
        quantization_config=None,
    )
    updater.rollout_engines = []
    updater.rollout_engine_lock = lock
    updater._group_name = "slime-pp_0"
    updater._model_update_groups = object()
    updater.weight_version = 1
    return updater


def test_transfer_failure_releases_rollout_engine_lock(monkeypatch):
    module = _load_module(monkeypatch)
    events = []
    lock = _FakeLock(events)
    updater = _make_updater(module, lock)
    converted = [("mlp.weight", "tensor")]
    pbar = _FakePbar(events)

    def fake_transfer(*args, **kwargs):
        events.append("transfer")
        raise RuntimeError("nccl broadcast failed")

    monkeypatch.setattr(module, "update_weights_from_distributed", fake_transfer)

    with pytest.raises(RuntimeError, match="nccl broadcast failed"):
        updater._update_bucket_weights_from_distributed(converted, pbar=pbar)

    assert events == ["acquire", "transfer", "release"]
    assert pbar.updates == []
    assert converted == [("mlp.weight", "tensor")]  # cleared only after a successful apply


def test_engine_apply_failure_releases_rollout_engine_lock(monkeypatch):
    module = _load_module(monkeypatch)
    events = []
    lock = _FakeLock(events)
    updater = _make_updater(module, lock)
    converted = [("mlp.weight", "tensor")]
    pbar = _FakePbar(events)
    engine_refs = ["engine-ref-1"]

    def fake_transfer(*args, **kwargs):
        events.append("transfer")
        return engine_refs

    monkeypatch.setattr(module, "update_weights_from_distributed", fake_transfer)

    def fake_get(value):
        if value is engine_refs:
            raise RuntimeError("engine apply failed")
        return value

    monkeypatch.setattr(module.ray, "get", fake_get)

    with pytest.raises(RuntimeError, match="engine apply failed"):
        updater._update_bucket_weights_from_distributed(converted, pbar=pbar)

    assert events == ["acquire", "transfer", "release"]
    assert pbar.updates == []
    assert converted == [("mlp.weight", "tensor")]


def test_success_releases_rollout_engine_lock_and_updates_pbar(monkeypatch):
    module = _load_module(monkeypatch)
    events = []
    lock = _FakeLock(events)
    updater = _make_updater(module, lock)
    converted = [("mlp.weight", "tensor")]
    pbar = _FakePbar(events)
    transfer_calls = []

    def fake_transfer(group_name, groups, weight_version, engines, named_tensors, load_format=None):
        transfer_calls.append((group_name, weight_version, load_format))
        events.append("transfer")
        return ["engine-ref-1"]

    monkeypatch.setattr(module, "update_weights_from_distributed", fake_transfer)

    updater._update_bucket_weights_from_distributed(converted, pbar=pbar)

    assert transfer_calls == [("slime-pp_0", 1, None)]
    assert converted == []
    assert events == ["acquire", "transfer", "release", "pbar"]
    assert pbar.updates == [1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
