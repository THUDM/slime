"""Native GDN import and serving export preserve heterogeneous TP sections."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NUM_GPUS = 0


def load_functions(relative, names, **namespace):
    path = ROOT / relative
    definitions = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec("from __future__ import annotations\n" + "\n\n".join(ast.unparse(n) for n in definitions), namespace)
    return namespace


def converter_modules():
    import re

    namespace = load_functions(
        "slime/backends/megatron_utils/hf_to_megatron/qwen3_5.py",
        {"qwen3_5_hf_tensor", "_merge_qkv"},
        torch=torch,
        re=re,
        strip_mcore_wrappers=lambda name: name.removeprefix("module.module."),
    )
    exported = load_functions(
        "slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py",
        {"convert_qwen3_5_to_hf", "_convert_mtp_layer"},
        torch=torch,
        re=re,
    )
    shard = load_functions(
        "slime/backends/megatron_utils/hf_to_megatron/common.py", {"_tensor_parallel_shard"}, torch=torch
    )
    return namespace["qwen3_5_hf_tensor"], exported["convert_qwen3_5_to_hf"], shard["_tensor_parallel_shard"]


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_native_checkpoint_to_shards_to_serving_roundtrip(tp_size, mode):
    load, export, shard = converter_modules()
    prefix = "model.language_model.layers.0."
    shapes = {
        "input_layernorm.weight": (8,),
        "linear_attn.in_proj_qkv.weight": (20, 8),
        "linear_attn.in_proj_z.weight": (12, 8),
        "linear_attn.in_proj_b.weight": (4, 8),
        "linear_attn.in_proj_a.weight": (4, 8),
        "linear_attn.conv1d.weight": (20, 1, 4),
        "linear_attn.A_log": (4,),
        "linear_attn.dt_bias": (4,),
        "linear_attn.norm.weight": (3,),
        "linear_attn.out_proj.weight": (8, 12),
    }
    tensors = {
        prefix + name: (torch.arange(torch.tensor(shape).prod()).reshape(shape) / 16).bfloat16()
        for name, shape in shapes.items()
    }
    # These directly stored norm scales must survive without a lossy +/-1 transform.
    tensors[prefix + "linear_attn.norm.weight"] = torch.tensor([0.003, 0.123, 1.75], dtype=torch.bfloat16)
    reader = SimpleNamespace(get_tensor=tensors.__getitem__)
    config = SimpleNamespace()
    args = SimpleNamespace(hidden_size=8, num_attention_heads=4, num_query_groups=2, kv_channels=2)
    parameters = {
        "in_proj.weight": (0, (4, 4, 12, 12, 4, 4)),
        "in_proj.layer_norm_weight": (None, None),
        "conv1d.weight": (0, (4, 4, 12)),
        "A_log": (0, None),
        "dt_bias": (0, None),
        "out_norm.weight": (None, None),
        "out_proj.weight": (1, None),
    }
    restored = {}
    for suffix, (dim, sections) in parameters.items():
        name = "module.module.decoder.layers.0.self_attention." + suffix
        full = load(name, reader, config)
        parts = [
            (
                shard(
                    name,
                    full,
                    parallel_size=tp_size,
                    parallel_rank=rank,
                    partition_dim=dim,
                    partition_stride=1,
                    partition_sections=sections,
                )
                if dim is not None
                else full.clone()
            )
            for rank in range(tp_size)
        ]
        local = torch.nn.Parameter(parts[0])
        local.tensor_model_parallel = dim is not None and tp_size > 1
        local.partition_dim = dim
        local.partition_stride = 1
        local.partition_sections = sections
        waits = []

        def all_gather(destinations, source, *, group, async_op=False, parts=parts, waits=waits):
            assert torch.equal(source, parts[0])
            for dst, src in zip(destinations, parts, strict=True):
                dst.copy_(src)
            return SimpleNamespace(wait=lambda: waits.append(True)) if async_op else None

        gather = load_functions(
            "slime/backends/megatron_utils/update_weight/common.py",
            {"_merge_param_partitions", "all_gather_param", "all_gather_params_async"},
            torch=torch,
            dist=SimpleNamespace(all_gather=all_gather),
            mpu=SimpleNamespace(
                get_tensor_model_parallel_world_size=lambda: tp_size, get_tensor_model_parallel_group=lambda: None
            ),
        )
        if mode == "sync":
            result = gather["all_gather_param"](name, local)
        else:
            info = SimpleNamespace(name=name, attrs={"partition_sections": sections})
            (result,) = gather["all_gather_params_async"]([(info, local)])
            assert len(waits) == int(tp_size > 1 and dim is not None)
        assert torch.equal(result, full)
        restored.update(export(args, name, result))
    assert restored.keys() == tensors.keys()
    for name in tensors:
        assert torch.equal(restored[name], tensors[name]), name


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize(
    "name,shape,dim,stride",
    [
        ("linear_fc1.weight", (16, 8), 0, 2),
        ("linear_fc1.bias", (16,), 0, 2),
        ("linear_fc2.weight", (8, 16), 0, 1),
        ("linear_qkv.weight", (16, 8), 0, 1),
    ],
)
def test_existing_tp_layouts_still_roundtrip(tp_size, name, shape, dim, stride):
    _, _, shard = converter_modules()
    merge = load_functions(
        "slime/backends/megatron_utils/update_weight/common.py", {"_merge_param_partitions"}, torch=torch
    )["_merge_param_partitions"]
    full = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    parts = [
        shard(name, full, parallel_size=tp_size, parallel_rank=rank, partition_dim=dim, partition_stride=stride)
        for rank in range(tp_size)
    ]
    assert torch.equal(merge(name, parts, dim), full)


def test_native_output_norm_uses_direct_scale(monkeypatch):
    import types

    class GDN(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config
            self.tp_size = 2
            self.qk_dim, self.v_dim = 4, 12
            self.in_proj_split_sections = (2, 2, 6, 6, 2, 2)
            self.in_proj = torch.nn.Linear(8, 20)
            self.conv1d = torch.nn.Conv1d(10, 10, 4, groups=10)
            self.out_norm = torch.nn.LayerNorm(3)
            self.out_norm.zero_centered_gamma = True

    module = types.ModuleType("megatron.core.ssm.gated_delta_net.gdn")
    module.GatedDeltaNet = GDN
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location("native_qwen_test", ROOT / "slime_plugins/models/qwen3_5_native.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    config = SimpleNamespace(
        layernorm_zero_centered_gamma=True,
        linear_num_value_heads=32,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
    )
    layer = native.Qwen3_5NativeAttention(config, {"linear_num_value_heads": 48})
    assert layer.config.linear_num_value_heads == 48 and config.linear_num_value_heads == 32
    assert layer.out_norm.zero_centered_gamma is False
    assert torch.equal(layer.out_norm.weight, torch.ones(3))
    assert layer.in_proj.weight.partition_sections == (4, 4, 12, 12, 4, 4)
    assert layer.conv1d.weight.partition_sections == (4, 4, 12)
    config.sequence_parallel = False
    with pytest.raises(ValueError, match="sequence-parallel"):
        native.Qwen3_5NativeAttention(config, {"linear_num_value_heads": 48})
    config.sequence_parallel = True
    config.layernorm_zero_centered_gamma = False
    with pytest.raises(ValueError, match="apply-layernorm-1p"):
        native.Qwen3_5NativeAttention(config, {"linear_num_value_heads": 48})


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_native_packed_validation_reuses_offsets_and_rechecks_edits(monkeypatch, padded, inference):
    import types

    calls = []

    class GDN(torch.nn.Module):
        def _resolve_cu_seqlens(self, cu_padded, cu_actual, total, name, cp_size=1):
            cu = cu_padded if cu_padded is not None else cu_actual
            calls.append((total, cp_size))
            if int(cu[-1]) != total:
                raise ValueError("total length mismatch")
            if (cu.diff() % cp_size != 0).any():
                raise ValueError("length is not divisible")
            return cu

    module = types.ModuleType("megatron.core.ssm.gated_delta_net.gdn")
    module.GatedDeltaNet = GDN
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location(
        "native_offset_test", ROOT / "slime_plugins/models/qwen3_5_native.py"
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    layers = [native.Qwen3_5NativeAttention.__new__(native.Qwen3_5NativeAttention) for _ in range(2)]
    for layer in layers:
        torch.nn.Module.__init__(layer)
    with torch.inference_mode(inference):
        cu = torch.tensor([0, 8, 24])
        actual = torch.tensor([0, 5, 17]) if padded else cu
        padding = cu if padded else None
        for layer in layers * 3:
            assert layer._resolve_cu_seqlens(padding, actual, 24, "q", 2) is cu
        assert len(calls) == (6 if inference else 1)
        # A changed interpretation must still reject otherwise cached data.
        with pytest.raises(ValueError, match="total length"):
            layers[0]._resolve_cu_seqlens(padding, actual, 28, "kv", 2)
        # Edits through an alias share the version counter.
        cu.view(-1)[-1] = 28
        with pytest.raises(ValueError, match="total length"):
            layers[0]._resolve_cu_seqlens(padding, actual, 24, "q", 2)
        assert layers[1]._resolve_cu_seqlens(padding, actual, 28, "kv", 2) is cu
        with pytest.raises(ValueError, match="divisible"):
            layers[0]._resolve_cu_seqlens(padding, actual, 28, "q", 8)
        # A new tensor must undergo its own validation.
        before = len(calls)
        other = cu.clone()
        assert layers[0]._resolve_cu_seqlens(None, other, 28, "q", 2) is other
        assert len(calls) == before + 1


@pytest.mark.parametrize("implementation", ["replicated", "megatron"])
def test_spec_changes_only_linear_attention_layers(monkeypatch, implementation):
    import copy
    import types

    native_module = types.ModuleType("slime_plugins.models.qwen3_5_native")
    native_module.Qwen3_5NativeAttention = type("NativeAttention", (), {})
    monkeypatch.setitem(sys.modules, native_module.__name__, native_module)
    factory = types.ModuleType("megatron.core.models.gpt.experimental_attention_variant_module_specs")
    factory.get_gated_delta_net_module_spec = lambda config: SimpleNamespace(module=None, params={})
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    ordinary, replicated = object(), type("ReplicatedAttention", (), {})
    text_config = SimpleNamespace(
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    namespace = load_functions(
        "slime_plugins/models/qwen3_5.py",
        {"get_qwen3_5_spec"},
        __package__="slime_plugins.models",
        copy=copy,
        _load_hf_config=lambda path: text_config,
        _get_text_config=lambda value: value,
        get_num_layers_to_build=lambda config, vp_stage: 2,
        get_transformer_layer_offset=lambda config, vp_stage: 0,
        get_gpt_decoder_block_spec=lambda *args, **kwargs: SimpleNamespace(
            layer_specs=[SimpleNamespace(submodules=SimpleNamespace(self_attention=ordinary)) for _ in range(2)]
        ),
        Attention=replicated,
        ModuleSpec=SimpleNamespace,
    )
    args = SimpleNamespace(
        num_experts=None, hf_checkpoint="unused", qwen_gdn_implementation=implementation, qwen_gdn_backend="fla"
    )
    config = SimpleNamespace(num_layers=2, pipeline_model_parallel_layout=None)
    spec = namespace["get_qwen3_5_spec"](args, config, None)
    assert spec.layer_specs[1].submodules.self_attention is ordinary
    attention = spec.layer_specs[0].submodules.self_attention
    if implementation == "megatron":
        assert attention.module is native_module.Qwen3_5NativeAttention
        assert attention.params["linear_config"]["linear_num_value_heads"] == 48
    else:
        assert attention.module is replicated and attention.params["args"] is args


def test_invalid_projection_sections_are_rejected():
    _, export, shard = converter_modules()
    with pytest.raises(ValueError, match="divisible"):
        shard(
            "in_proj.weight",
            torch.zeros(8, 2),
            parallel_size=2,
            parallel_rank=0,
            partition_dim=0,
            partition_stride=1,
            partition_sections=(3, 5),
        )
    args = SimpleNamespace(hidden_size=8, num_attention_heads=4, num_query_groups=2, kv_channels=2)
    with pytest.raises(ValueError, match="partition sections"):
        export(args, "module.module.decoder.layers.0.self_attention.in_proj.weight", torch.zeros(8, 2))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
