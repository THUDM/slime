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


def _load_common(monkeypatch):
    slime_pkg = _package("slime", REPO_ROOT / "slime")
    backends_pkg = _package("slime.backends", REPO_ROOT / "slime" / "backends")
    megatron_utils_pkg = _package("slime.backends.megatron_utils", REPO_ROOT / "slime" / "backends" / "megatron_utils")
    update_weight_pkg = _package(
        "slime.backends.megatron_utils.update_weight",
        REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight",
    )
    slime_utils_pkg = _package("slime.utils", REPO_ROOT / "slime" / "utils")

    misc_utils_mod = types.ModuleType("slime.backends.megatron_utils.misc_utils")
    misc_utils_mod.strip_param_name_prefix = lambda name: name.removeprefix("module.")
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.ParamInfo = object

    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    mpu_mod = types.ModuleType("megatron.core.mpu")
    transformer_mod = types.ModuleType("megatron.core.transformer")
    transformer_layer_mod = types.ModuleType("megatron.core.transformer.transformer_layer")
    transformer_layer_mod.get_transformer_layer_offset = lambda *_args, **_kwargs: 0
    core_mod.mpu = mpu_mod

    modules = {
        "slime": slime_pkg,
        "slime.backends": backends_pkg,
        "slime.backends.megatron_utils": megatron_utils_pkg,
        "slime.backends.megatron_utils.update_weight": update_weight_pkg,
        "slime.backends.megatron_utils.misc_utils": misc_utils_mod,
        "slime.utils": slime_utils_pkg,
        "slime.utils.types": types_mod,
        "megatron": megatron_mod,
        "megatron.core": core_mod,
        "megatron.core.mpu": mpu_mod,
        "megatron.core.transformer": transformer_mod,
        "megatron.core.transformer.transformer_layer": transformer_layer_mod,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "test_grouped_moe_glu_rechunk_common"
    module_path = REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight" / "common.py"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def common(monkeypatch):
    return _load_common(monkeypatch)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "shape", "expected"),
    [
        ("decoder.layers.0.mlp.linear_fc1.weight", (1024, 2048), True),
        ("decoder.layers.0.mlp.linear_fc1.bias", (1024,), True),
        ("decoder.layers.0.mlp.experts.linear_fc1.weight", (64, 1024, 2048), False),
        ("decoder.layers.0.mlp.experts.linear_fc1.bias", (64, 1024), False),
        ("decoder.layers.0.mlp.experts.linear_fc2.weight", (64, 2048, 512), False),
    ],
)
def test_glu_rechunk_only_applies_to_non_grouped_fc1_tensors(common, name, shape, expected):
    assert common._requires_glu_rechunk(name, torch.empty(shape)) is expected


class _ParallelParam(torch.nn.Parameter):
    def __new__(cls, data, partition_dim):
        param = super().__new__(cls, data, requires_grad=False)
        param.partition_dim = partition_dim
        param.partition_stride = 1
        param.tensor_model_parallel = True
        return param


class _CompletedWork:
    def wait(self):
        return None


def _patch_two_rank_all_gather(monkeypatch, common):
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    monkeypatch.setattr(common.mpu, "get_expert_tensor_parallel_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(common.mpu, "get_expert_tensor_parallel_group", lambda: None, raising=False)

    def all_gather(outputs, _input, group, async_op=False):
        del group
        for rank, output in enumerate(outputs):
            output.fill_(rank + 1)
        return _CompletedWork() if async_op else None

    monkeypatch.setattr(common.dist, "all_gather", all_gather)


@pytest.mark.unit
def test_sync_all_gather_preserves_grouped_moe_expert_axis(monkeypatch, common):
    _patch_two_rank_all_gather(monkeypatch, common)
    param = _ParallelParam(torch.empty(2, 4, 3), partition_dim=2)

    gathered = common.all_gather_param("decoder.layers.0.mlp.experts.linear_fc1.weight", param)

    assert gathered.shape == (2, 4, 6)
    assert torch.equal(gathered[..., :3], torch.ones(2, 4, 3))
    assert torch.equal(gathered[..., 3:], torch.full((2, 4, 3), 2.0))


@pytest.mark.unit
def test_async_all_gather_preserves_grouped_moe_expert_axis(monkeypatch, common):
    _patch_two_rank_all_gather(monkeypatch, common)
    param = _ParallelParam(torch.empty(2, 4, 3), partition_dim=2)
    info = SimpleNamespace(name="decoder.layers.0.mlp.experts.linear_fc1.weight")

    (gathered,) = common.all_gather_params_async([(info, param)])

    assert gathered.shape == (2, 4, 6)
    assert torch.equal(gathered[..., :3], torch.ones(2, 4, 3))
    assert torch.equal(gathered[..., 3:], torch.full((2, 4, 3), 2.0))


@pytest.mark.unit
def test_sync_all_gather_keeps_2d_glu_rechunk_order(monkeypatch, common):
    _patch_two_rank_all_gather(monkeypatch, common)
    param = _ParallelParam(torch.empty(4, 3), partition_dim=0)

    gathered = common.all_gather_param("decoder.layers.0.mlp.linear_fc1.weight", param)

    expected = torch.cat(
        [
            torch.ones(2, 3),
            torch.full((2, 3), 2.0),
            torch.ones(2, 3),
            torch.full((2, 3), 2.0),
        ]
    )
    assert torch.equal(gathered, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
