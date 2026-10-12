import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_common(monkeypatch):
    expert_group = object()
    tensor_group = object()

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

    mpu_mod = types.ModuleType("megatron.core.mpu")
    mpu_mod.get_expert_tensor_parallel_world_size = lambda: 2
    mpu_mod.get_expert_tensor_parallel_group = lambda: expert_group
    mpu_mod.get_tensor_model_parallel_world_size = lambda: 2
    mpu_mod.get_tensor_model_parallel_group = lambda: tensor_group

    megatron_mod = types.ModuleType("megatron")
    megatron_core_mod = types.ModuleType("megatron.core")
    megatron_core_mod.mpu = mpu_mod
    transformer_pkg = types.ModuleType("megatron.core.transformer")
    transformer_layer_mod = types.ModuleType("megatron.core.transformer.transformer_layer")
    transformer_layer_mod.get_transformer_layer_offset = lambda *args, **kwargs: 0

    misc_utils_mod = types.ModuleType("slime.backends.megatron_utils.misc_utils")
    misc_utils_mod.strip_param_name_prefix = lambda name: name
    slime_types_mod = types.ModuleType("slime.utils.types")
    slime_types_mod.ParamInfo = object

    modules = {
        "slime": slime_pkg,
        "slime.backends": slime_backends_pkg,
        "slime.backends.megatron_utils": megatron_utils_pkg,
        "slime.backends.megatron_utils.update_weight": update_weight_pkg,
        "slime.backends.megatron_utils.misc_utils": misc_utils_mod,
        "slime.utils": slime_utils_pkg,
        "slime.utils.types": slime_types_mod,
        "megatron": megatron_mod,
        "megatron.core": megatron_core_mod,
        "megatron.core.mpu": mpu_mod,
        "megatron.core.transformer": transformer_pkg,
        "megatron.core.transformer.transformer_layer": transformer_layer_mod,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "slime.backends.megatron_utils.update_weight.common"
    sys.modules.pop(module_name, None)
    module_path = REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight" / "common.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, expert_group, tensor_group


def _install_fake_all_gather(monkeypatch, module):
    calls = []
    handles = []

    class _Handle:
        waited = False

        def wait(self):
            self.waited = True

    def _all_gather(outputs, value, *, group, async_op=False):
        outputs[0].copy_(value)
        outputs[1].copy_(value + 2)
        calls.append((group, async_op))
        if async_op:
            handle = _Handle()
            handles.append(handle)
            return handle
        return None

    monkeypatch.setattr(module.dist, "all_gather", _all_gather)
    return calls, handles


def _param(value, **attrs):
    param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32), requires_grad=False)
    for name, attr_value in attrs.items():
        setattr(param, name, attr_value)
    return param


def test_sync_gathers_legacy_grouped_fc1_without_tp_metadata(monkeypatch):
    module, expert_group, _ = _load_common(monkeypatch)
    calls, _ = _install_fake_all_gather(monkeypatch, module)
    param = _param([1, 2])

    result = module.all_gather_param(
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0",
        param,
    )

    torch.testing.assert_close(result, torch.tensor([1, 3, 2, 4], dtype=torch.float32))
    assert calls == [(expert_group, False)]


def test_sync_gathers_legacy_grouped_fc2_along_input_dimension(monkeypatch):
    module, expert_group, _ = _load_common(monkeypatch)
    calls, _ = _install_fake_all_gather(monkeypatch, module)
    param = _param([[1, 2]])

    result = module.all_gather_param(
        "module.module.decoder.layers.0.mlp.experts.linear_fc2.weight0",
        param,
    )

    torch.testing.assert_close(result, torch.tensor([[1, 2, 3, 4]], dtype=torch.float32))
    assert calls == [(expert_group, False)]


def test_legacy_grouped_row_bias_and_unrecognized_expert_stay_replicated(monkeypatch):
    module, _, _ = _load_common(monkeypatch)
    calls, _ = _install_fake_all_gather(monkeypatch, module)
    row_bias = _param([1, 2])
    router = _param([3, 4], tensor_model_parallel=False)

    row_result = module.all_gather_param(
        "module.module.decoder.layers.0.mlp.experts.linear_fc2.bias0",
        row_bias,
    )
    router_result = module.all_gather_param(
        "module.module.decoder.layers.0.mlp.experts.router.weight",
        router,
    )

    torch.testing.assert_close(row_result, row_bias)
    torch.testing.assert_close(router_result, router)
    assert calls == []


def test_unrecognized_parameter_without_tp_metadata_still_fails_fast(monkeypatch):
    module, _, _ = _load_common(monkeypatch)
    param = _param([1, 2])

    with pytest.raises(AssertionError, match="does not have tensor_model_parallel attribute"):
        module.all_gather_param("module.module.decoder.layers.0.mlp.router.weight", param)


def test_existing_tensor_parallel_metadata_keeps_regular_tp_group(monkeypatch):
    module, _, tensor_group = _load_common(monkeypatch)
    calls, _ = _install_fake_all_gather(monkeypatch, module)
    param = _param([1, 2], tensor_model_parallel=True, partition_dim=0, partition_stride=1)

    result = module.all_gather_param("module.module.decoder.layers.0.self_attention.linear_qkv.weight", param)

    torch.testing.assert_close(result, torch.tensor([1, 2, 3, 4], dtype=torch.float32))
    assert calls == [(tensor_group, False)]


def test_async_gathers_legacy_grouped_expert_with_false_metadata(monkeypatch):
    module, expert_group, _ = _load_common(monkeypatch)
    calls, handles = _install_fake_all_gather(monkeypatch, module)
    param = _param([1, 2], tensor_model_parallel=False, partition_dim=-1, partition_stride=1)
    info = types.SimpleNamespace(name="module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0")

    result = module.all_gather_params_async([(info, param)])

    torch.testing.assert_close(result[0], torch.tensor([1, 3, 2, 4], dtype=torch.float32))
    assert calls == [(expert_group, True)]
    assert len(handles) == 1 and handles[0].waited
