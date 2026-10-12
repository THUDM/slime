import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_module(module_name, relative_path):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


from slime.backends import qwen3_5_expert_layout as layout


qwen3_5 = _load_module(
    "test_qwen35_megatron_to_hf_qwen3_5",
    "slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py",
)
converter = _load_module("test_qwen35_int4_converter", "tools/convert_hf_to_int4_direct.py")


@pytest.fixture(autouse=True)
def _use_cpu_for_converter_metadata(monkeypatch):
    monkeypatch.setattr(converter.accelerator, "memory_allocated", lambda: 0)
    monkeypatch.setattr(converter.accelerator, "device", lambda: torch.device("cpu"))


def _install_megatron_stubs():
    megatron = types.ModuleType("megatron")
    megatron.__path__ = []
    core = types.ModuleType("megatron.core")
    core.__path__ = []
    mpu = types.ModuleType("megatron.core.mpu")
    mpu.get_expert_model_parallel_world_size = lambda: 1
    mpu.get_expert_model_parallel_rank = lambda: 0
    transformer = types.ModuleType("megatron.core.transformer")
    transformer.__path__ = []
    transformer_layer = types.ModuleType("megatron.core.transformer.transformer_layer")
    transformer_layer.get_transformer_layer_offset = lambda config, *args: 0
    core.mpu = mpu
    megatron.core = core
    sys.modules.update(
        {
            "megatron": megatron,
            "megatron.core": core,
            "megatron.core.mpu": mpu,
            "megatron.core.transformer": transformer,
            "megatron.core.transformer.transformer_layer": transformer_layer,
        }
    )


NUM_EXPERTS = 256
EP_SIZE = 4
LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE
FFN = 8
HIDDEN = 4


def _args(*, is_qwen3_5_moe=True):
    return SimpleNamespace(
        num_experts=NUM_EXPERTS,
        kv_channels=None,
        hidden_size=HIDDEN,
        num_attention_heads=2,
        num_query_groups=1,
        is_qwen3_5_moe=is_qwen3_5_moe,
    )


def _ep_offset(ep_rank):
    return ep_rank * NUM_EXPERTS // EP_SIZE


def _expert_ids(named_params, suffix):
    ids = []
    for name, _ in named_params:
        marker = ".mlp.experts."
        if suffix in name and marker in name:
            ids.append(int(name.split(marker, 1)[1].split(".", 1)[0]))
    return sorted(ids)


@pytest.mark.integration
def test_native_bridge_global_name_carries_ep_offset(monkeypatch):
    try:
        has_megatron = importlib.util.find_spec("megatron.core") is not None
    except ModuleNotFoundError:
        has_megatron = False
    if not has_megatron:
        original_modules = {name: sys.modules.get(name) for name in list(sys.modules) if name.startswith("megatron")}
        _install_megatron_stubs()
    else:
        original_modules = None

    try:
        update_weight_common = _load_module(
            "test_qwen35_update_weight_common",
            "slime/backends/megatron_utils/update_weight/common.py",
        )
    finally:
        if original_modules is not None:
            for name in [name for name in list(sys.modules) if name.startswith("megatron")]:
                sys.modules.pop(name, None)
            sys.modules.update(original_modules)

    monkeypatch.setattr(update_weight_common.mpu, "get_expert_model_parallel_world_size", lambda: EP_SIZE)
    monkeypatch.setattr(update_weight_common.mpu, "get_expert_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(update_weight_common, "get_transformer_layer_offset", lambda config, *args: 3)

    param = torch.nn.Parameter(torch.zeros(LOCAL_EXPERTS, 2 * FFN, HIDDEN))

    class Model:
        config = SimpleNamespace()

        def named_parameters(self):
            return iter(
                [
                    (
                        "module.module.language_model.decoder.layers.1.mlp.experts.linear_fc1",
                        param,
                    )
                ]
            )

        def named_buffers(self):
            return iter([])

    [(name, emitted_param)] = list(update_weight_common.named_params_and_buffers(_args(), [Model()]))

    assert name == ("module.module.language_model.decoder.layers.4.mlp.experts.linear_fc1.__ep_offset128")
    assert emitted_param is param

    [(other_name, other_param)] = list(
        update_weight_common.named_params_and_buffers(_args(is_qwen3_5_moe=False), [Model()])
    )
    assert other_name == "module.module.language_model.decoder.layers.4.mlp.experts.linear_fc1"
    assert other_param is param

    unsupported_args = _args()
    unsupported_args.expert_tensor_parallel_size = 2
    with pytest.raises(ValueError, match="expert tensor parallel size 1"):
        list(update_weight_common.named_params_and_buffers(unsupported_args, [Model()]))


@pytest.mark.unit
def test_runtime_fused_experts_keep_ep_global_ids():
    args = _args()
    all_gate_ids = []
    all_down_ids = []

    for ep_rank in range(EP_SIZE):
        offset = _ep_offset(ep_rank)
        fc1_name = f"module.module.language_model.decoder.layers.0.mlp.experts.linear_fc1.__ep_offset{offset}"
        fc2_name = f"module.module.language_model.decoder.layers.0.mlp.experts.linear_fc2.__ep_offset{offset}"

        fc1_out = qwen3_5.convert_qwen3_5_to_hf(args, fc1_name, torch.zeros(LOCAL_EXPERTS, 2 * FFN, HIDDEN))
        fc2_out = qwen3_5.convert_qwen3_5_to_hf(args, fc2_name, torch.zeros(LOCAL_EXPERTS, HIDDEN, FFN))

        expected = list(range(offset, offset + LOCAL_EXPERTS))
        assert _expert_ids(fc1_out, "gate_proj.weight") == expected
        assert _expert_ids(fc2_out, "down_proj.weight") == expected
        all_gate_ids.extend(expected)
        all_down_ids.extend(expected)

    assert sorted(all_gate_ids) == list(range(NUM_EXPERTS))
    assert sorted(all_down_ids) == list(range(NUM_EXPERTS))


@pytest.mark.integration
def test_runtime_fused_expert_split_matches_offline_split():
    args = _args()
    fc1 = torch.arange(LOCAL_EXPERTS * 2 * FFN * HIDDEN, dtype=torch.float32).view(LOCAL_EXPERTS, 2 * FFN, HIDDEN)
    fc2 = torch.arange(LOCAL_EXPERTS * HIDDEN * FFN, dtype=torch.float32).view(LOCAL_EXPERTS, HIDDEN, FFN)

    runtime = dict(
        qwen3_5.convert_qwen3_5_to_hf(
            args,
            "module.module.language_model.decoder.layers.0.mlp.experts.linear_fc1.__ep_offset0",
            fc1,
        )
        + qwen3_5.convert_qwen3_5_to_hf(
            args,
            "module.module.language_model.decoder.layers.0.mlp.experts.linear_fc2.__ep_offset0",
            fc2,
        )
    )
    prefix = "model.language_model.layers.0.mlp.experts"
    offline = {
        f"{prefix}.{expert_id}.{projection}.weight": weight
        for fused_weight, fused_projection in (
            (fc1, layout.GATE_UP_PROJ),
            (fc2, layout.DOWN_PROJ),
        )
        for expert_id, projection, weight in layout.iter_fused_expert_projections(
            fused_weight,
            fused_projection,
        )
    }

    assert runtime.keys() == offline.keys()
    assert all(torch.equal(runtime[name], offline[name]) for name in runtime)


@pytest.mark.unit
def test_fused_gate_up_rejects_an_odd_projection_dimension():
    with pytest.raises(ValueError, match="output dimension must be even"):
        list(layout.iter_fused_expert_projections(torch.zeros(2, 7, HIDDEN), layout.GATE_UP_PROJ))


@pytest.mark.unit
def test_fused_expert_mode_is_scoped_to_qwen35_configs():
    exact_config = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    }
    assert layout.is_qwen3_5_moe_config(exact_config)
    assert layout.is_qwen3_5_moe_config(SimpleNamespace(**exact_config))
    assert not layout.is_qwen3_5_moe_config({"model_type": "qwen3_5_moe"})
    assert not layout.is_qwen3_5_moe_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
        }
    )
    assert not layout.is_qwen3_5_moe_config(
        {
            "model_type": "other_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        }
    )


@pytest.mark.unit
def test_offline_converter_fused_mode_quantizes_only_split_routed_experts(monkeypatch):
    def fake_pack_layer(weight, group_size, sym=True):
        packed = torch.full((weight.shape[0], 1), fill_value=weight.shape[0], dtype=torch.int32)
        scale = torch.ones(weight.shape[0], 1, dtype=weight.dtype)
        return packed, scale, None

    monkeypatch.setattr(converter, "pack_layer", fake_pack_layer)

    weights = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 2 * FFN, HIDDEN),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(2, HIDDEN, FFN),
        "model.language_model.layers.0.linear_attn.out_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(2, HIDDEN),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(HIDDEN, FFN),
        "model.visual.patch_embed.weight": torch.randn(HIDDEN, HIDDEN),
        "mtp.layers.0.mlp.down_proj.weight": torch.randn(HIDDEN, FFN),
    }

    q_weights = converter.convert_weights(
        weights,
        group_size=128,
        is_symmetric=True,
        ignore_rules=converter.QWEN35_IGNORE_RULES,
        qwen35_moe=True,
    )

    for expert_id in range(2):
        assert f"model.language_model.layers.0.mlp.experts.{expert_id}.gate_proj.weight_packed" in q_weights
        assert f"model.language_model.layers.0.mlp.experts.{expert_id}.up_proj.weight_packed" in q_weights
        assert f"model.language_model.layers.0.mlp.experts.{expert_id}.down_proj.weight_packed" in q_weights

    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" not in q_weights
    assert "model.language_model.layers.0.mlp.experts.down_proj" not in q_weights
    for name in weights.keys() - {
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    }:
        assert name in q_weights
        assert name.replace(".weight", ".weight_packed") not in q_weights


@pytest.mark.unit
def test_non_qwen_fused_key_is_not_split(monkeypatch):
    def fail_pack_layer(weight, group_size, sym=True):
        raise AssertionError("a non-Qwen fused key must not enter Qwen3.5 split/pack")

    monkeypatch.setattr(converter, "pack_layer", fail_pack_layer)

    name = "model.layers.0.mlp.experts.gate_up_proj"
    tensor = torch.randn(2, 2 * FFN, HIDDEN)
    result = converter.convert_weights(
        {name: tensor},
        group_size=128,
        is_symmetric=True,
        ignore_rules=[],
        qwen35_moe=False,
    )

    assert list(result) == [name]
    assert result[name] is tensor


@pytest.mark.unit
def test_qwen35_effective_quantization_config_keeps_non_experts_bf16():
    effective_ignore = converter.QWEN35_IGNORE_RULES

    assert "re:.*linear_attn.*" in effective_ignore
    assert "re:.*visual.*" in effective_ignore
    assert "re:.*shared_expert.*" in effective_ignore
    assert "re:.*mtp.*" in effective_ignore


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
