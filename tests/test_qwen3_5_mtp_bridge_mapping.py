import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def install_bridge_stubs():
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    models_mod = types.ModuleType("megatron.core.models")
    gpt_mod = types.ModuleType("megatron.core.models.gpt")
    gpt_layer_specs_mod = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    inference_mod = types.ModuleType("megatron.core.inference")
    inference_contexts_mod = types.ModuleType("megatron.core.inference.contexts")
    packed_seq_mod = types.ModuleType("megatron.core.packed_seq_params")
    transformer_mod = types.ModuleType("megatron.core.transformer")
    transformer_module_mod = types.ModuleType("megatron.core.transformer.module")
    spec_utils_mod = types.ModuleType("megatron.core.transformer.spec_utils")
    transformer_block_mod = types.ModuleType("megatron.core.transformer.transformer_block")
    transformer_layer_mod = types.ModuleType("megatron.core.transformer.transformer_layer")
    gpt_layer_specs_mod.get_gpt_mtp_block_spec = lambda _config, transformer_layer_spec, **_kwargs: (
        "mtp-spec",
        transformer_layer_spec,
    )
    gpt_layer_specs_mod.get_gpt_decoder_block_spec = lambda *args, **kwargs: None

    class PackedSeqParams:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class MegatronModule(torch.nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

    class ModuleSpec:
        def __init__(self, module=None, params=None):
            self.module = module
            self.params = params or {}

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_group=lambda: None,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
    )
    tensor_parallel_stub = types.SimpleNamespace(
        gather_from_sequence_parallel_region=lambda x, group=None: x,
        scatter_to_sequence_parallel_region=lambda x, group=None: x,
    )
    inference_contexts_mod.BaseInferenceContext = type("BaseInferenceContext", (), {})
    packed_seq_mod.PackedSeqParams = PackedSeqParams
    transformer_module_mod.MegatronModule = MegatronModule
    spec_utils_mod.ModuleSpec = ModuleSpec
    transformer_block_mod.get_num_layers_to_build = lambda *args, **kwargs: 0
    transformer_layer_mod.get_transformer_layer_offset = lambda *args, **kwargs: 0
    core_mod.mpu = mpu_stub
    core_mod.tensor_parallel = tensor_parallel_stub

    mbridge_mod = types.ModuleType("mbridge")
    mbridge_core_mod = types.ModuleType("mbridge.core")
    mbridge_models_mod = types.ModuleType("mbridge.models")

    def register_model(_names):
        def decorator(cls):
            return cls

        return decorator

    class Qwen2MoEBridge:
        _MLP_MAPPING = {
            "shared_experts.linear_fc1.weight": [
                "model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
                "model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
            ],
            "pre_mlp_layernorm": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
            "shared_experts.linear_fc2.weight": ["model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"],
            "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
            "shared_experts.gate_weight": ["model.layers.{layer_number}.mlp.shared_expert_gate.weight"],
            "mlp.experts.linear_fc1": [
                "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
                "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
            ],
            "mlp.experts.linear_fc2": ["model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"],
        }

        def _weight_name_mapping_mlp(self, name: str) -> list[str]:
            layer_number = name.split(".")[2]
            convert_names = []
            for keyword, mapping_names in self._MLP_MAPPING.items():
                if keyword in name:
                    if "{expert_id}" in mapping_names[0]:
                        expert_id = name.split("weight")[-1]
                        convert_names.extend(
                            [x.format(layer_number=layer_number, expert_id=expert_id) for x in mapping_names]
                        )
                    else:
                        convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                    break
            if len(convert_names) == 0:
                raise NotImplementedError(f"Unsupported parameter name: {name}")
            return convert_names

        def _weight_name_mapping_attention(self, name: str) -> list[str]:
            raise NotImplementedError(f"Unexpected attention mapping lookup: {name}")

        def _weight_name_mapping_mcore_to_hf(self, name: str) -> list[str]:
            return [name]

        def _get_transformer_layer_spec(self, vp_stage=None):
            return "REAL_LAYER_SPEC" if vp_stage is None else f"REAL_LAYER_SPEC_VP{vp_stage}"

        def _get_gptmodel_args(self) -> dict:
            return {"base": "ok"}

        def _model_provider(self, callbacks):
            def provider(pre_process, post_process, vp_stage=None):
                transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)
                gptmodel_args = self._get_gptmodel_args()
                return {"transformer_layer_spec": transformer_layer_spec, **gptmodel_args}

            return provider

        def _weight_to_mcore_format(self, _mcore_weights_name, hf_weights):
            assert len(hf_weights) == 1
            return hf_weights[0]

        def _weight_to_hf_format(self, mcore_weights_name, mcore_weights):
            return [mcore_weights_name], [mcore_weights]

        def _build_base_config(self, **kwargs):
            return kwargs

    mbridge_core_mod.register_model = register_model
    mbridge_models_mod.Qwen2MoEBridge = Qwen2MoEBridge

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.models"] = models_mod
    sys.modules["megatron.core.models.gpt"] = gpt_mod
    sys.modules["megatron.core.models.gpt.gpt_layer_specs"] = gpt_layer_specs_mod
    sys.modules["megatron.core.inference"] = inference_mod
    sys.modules["megatron.core.inference.contexts"] = inference_contexts_mod
    sys.modules["megatron.core.packed_seq_params"] = packed_seq_mod
    sys.modules["megatron.core.transformer"] = transformer_mod
    sys.modules["megatron.core.transformer.module"] = transformer_module_mod
    sys.modules["megatron.core.transformer.spec_utils"] = spec_utils_mod
    sys.modules["megatron.core.transformer.transformer_block"] = transformer_block_mod
    sys.modules["megatron.core.transformer.transformer_layer"] = transformer_layer_mod
    sys.modules["mbridge"] = mbridge_mod
    sys.modules["mbridge.core"] = mbridge_core_mod
    sys.modules["mbridge.models"] = mbridge_models_mod


def load_bridge_module():
    install_bridge_stubs()
    module_path = Path(__file__).resolve().parents[1] / "slime_plugins" / "mbridge" / "qwen3_5.py"
    module_name = "test_qwen3_5_bridge_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_raw_export_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "megatron_to_hf" / "qwen3_5.py"
    )
    module_name = "test_qwen3_5_raw_export_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_mtp_moe_expert_mapping_uses_individual_hf_weights():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)

    fc1_names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight42")
    fc2_names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.weight42")

    assert fc1_names == [
        "mtp.layers.0.mlp.experts.42.gate_proj.weight",
        "mtp.layers.0.mlp.experts.42.up_proj.weight",
    ]
    assert fc2_names == ["mtp.layers.0.mlp.experts.42.down_proj.weight"]


@pytest.mark.unit
def test_mtp_dense_mlp_mapping_still_uses_dense_hf_weights():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)

    fc1_names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.linear_fc1.weight")
    fc2_names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.linear_fc2.weight")

    assert fc1_names == ["mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight"]
    assert fc2_names == ["mtp.layers.0.mlp.down_proj.weight"]


@pytest.mark.unit
def test_mtp_block_spec_uses_current_transformer_layer_spec():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.config = "CONFIG_OBJECT"
    bridge.hf_config = types.SimpleNamespace(text_config=types.SimpleNamespace(mtp_num_hidden_layers=1))

    provider = bridge._model_provider([])
    result = provider(True, True, vp_stage=3)

    assert result["transformer_layer_spec"] == "REAL_LAYER_SPEC_VP3"
    assert result["mtp_block_spec"] == ("mtp-spec", "REAL_LAYER_SPEC_VP3")


@pytest.mark.unit
def test_eh_proj_keeps_column_order_when_loading_to_mcore():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)

    weight = torch.arange(24, dtype=torch.float32).view(3, 8)
    converted = bridge._weight_to_mcore_format("mtp.layers.0.eh_proj.weight", [weight])

    assert torch.equal(converted, weight)


@pytest.mark.unit
def test_build_config_enables_gated_attention_when_transformer_config_supports_it():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.hf_config = types.SimpleNamespace(text_config=types.SimpleNamespace(mtp_num_hidden_layers=1))
    bridge.TransformerConfigClass = types.SimpleNamespace(
        __dataclass_fields__={
            "mtp_num_layers": None,
            "attention_output_gate": None,
            "use_gated_attention": None,
        }
    )

    config = bridge._build_config()

    assert config["mtp_num_layers"] == 1
    assert config["attention_output_gate"] is True
    assert config["use_gated_attention"] is True


@pytest.mark.unit
def test_build_config_skips_gated_attention_when_transformer_config_does_not_support_it():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.hf_config = types.SimpleNamespace(text_config=types.SimpleNamespace(mtp_num_hidden_layers=1))
    bridge.TransformerConfigClass = types.SimpleNamespace(
        __dataclass_fields__={
            "mtp_num_layers": None,
            "attention_output_gate": None,
        }
    )

    config = bridge._build_config()

    assert config["mtp_num_layers"] == 1
    assert config["attention_output_gate"] is True
    assert "use_gated_attention" not in config


@pytest.mark.unit
def test_raw_qwen3_5_mtp_export_keeps_eh_proj_column_order():
    module = load_raw_export_module()

    weight = torch.arange(24, dtype=torch.float32).view(3, 8)
    converted = module.convert_qwen3_5_to_hf(
        types.SimpleNamespace(), "module.module.mtp.layers.0.eh_proj.weight", weight
    )

    assert converted == [("mtp.fc.weight", weight)]


def make_gdn_config():
    return types.SimpleNamespace(
        hidden_size=8,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        tensor_model_parallel_size=2,
        layernorm_zero_centered_gamma=True,
    )


@pytest.mark.unit
def test_qwen3_5_gdn_separate_in_proj_roundtrip_for_tp_packed_layout():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.config = make_gdn_config()

    qkv = torch.arange((2 * 8 + 16) * 8, dtype=torch.float32).view(32, 8)
    z = torch.arange(16 * 8, dtype=torch.float32).view(16, 8) + 1_000
    b = torch.arange(8 * 8, dtype=torch.float32).view(8, 8) + 2_000
    a = torch.arange(8 * 8, dtype=torch.float32).view(8, 8) + 3_000

    packed = bridge._weight_to_mcore_format("decoder.layers.0.self_attention.in_proj.weight", [qkv, z, b, a])
    qkv_out, z_out, b_out, a_out = module._split_gdn_linear_separate(bridge.config, packed)
    packed_blocks = packed.reshape(2, -1, 8)
    expected_block0 = torch.cat([qkv[0:4], qkv[8:12], qkv[16:24], z[0:8], b[0:4], a[0:4]], dim=0)
    expected_block1 = torch.cat([qkv[4:8], qkv[12:16], qkv[24:32], z[8:16], b[4:8], a[4:8]], dim=0)

    assert packed.shape == (64, 8)
    assert torch.equal(packed_blocks[0], expected_block0)
    assert torch.equal(packed_blocks[1], expected_block1)
    assert torch.equal(qkv_out, qkv)
    assert torch.equal(z_out, z)
    assert torch.equal(b_out, b)
    assert torch.equal(a_out, a)


@pytest.mark.unit
def test_qwen3_5_gdn_separate_in_proj_roundtrip_for_local_tp_shard():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.config = make_gdn_config()

    qkv = torch.arange((2 * 4 + 8) * 8, dtype=torch.float32).view(16, 8)
    z = torch.arange(8 * 8, dtype=torch.float32).view(8, 8) + 1_000
    b = torch.arange(4 * 8, dtype=torch.float32).view(4, 8) + 2_000
    a = torch.arange(4 * 8, dtype=torch.float32).view(4, 8) + 3_000

    packed = bridge._weight_to_mcore_format("decoder.layers.0.self_attention.in_proj.weight", [qkv, z, b, a])
    qkv_out, z_out, b_out, a_out = module._split_gdn_linear_separate(bridge.config, packed)

    assert packed.shape == (32, 8)
    assert torch.equal(qkv_out, qkv)
    assert torch.equal(z_out, z)
    assert torch.equal(b_out, b)
    assert torch.equal(a_out, a)


@pytest.mark.unit
def test_qwen3_5_gdn_conv1d_roundtrip_for_tp_packed_layout():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.config = make_gdn_config()

    conv = torch.arange((2 * 8 + 16) * 1 * 4, dtype=torch.float32).view(32, 1, 4)

    packed = bridge._weight_to_mcore_format("decoder.layers.0.self_attention.conv1d.weight", [conv])
    conv_out = module._split_gdn_conv1d_weight(bridge.config, packed)
    packed_blocks = packed.reshape(2, -1, 1, 4)
    expected_block0 = torch.cat([conv[0:4], conv[8:12], conv[16:24]], dim=0)
    expected_block1 = torch.cat([conv[4:8], conv[12:16], conv[24:32]], dim=0)

    assert packed.shape == conv.shape
    assert torch.equal(packed_blocks[0], expected_block0)
    assert torch.equal(packed_blocks[1], expected_block1)
    assert torch.equal(conv_out, conv)


@pytest.mark.unit
def test_qwen3_5_gdn_conv1d_roundtrip_for_local_tp_shard():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.config = make_gdn_config()

    conv = torch.arange((2 * 4 + 8) * 1 * 4, dtype=torch.float32).view(16, 1, 4)

    packed = bridge._weight_to_mcore_format("decoder.layers.0.self_attention.conv1d.weight", [conv])
    conv_out = module._split_gdn_conv1d_weight(bridge.config, packed)

    assert packed.shape == conv.shape
    assert torch.equal(conv_out, conv)


@pytest.mark.unit
def test_qwen3_5_gdn_out_norm_uses_zero_centered_rmsnorm_mapping():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.config = make_gdn_config()

    hf_weight = torch.tensor([0.87109375, 0.8671875, 0.88427734, 0.90332031])
    expected_mcore_weight = hf_weight - 1
    mcore_weight = bridge._weight_to_mcore_format("decoder.layers.0.self_attention.out_norm.weight", [hf_weight])
    _names, hf_weights = bridge._weight_to_hf_format("decoder.layers.0.self_attention.out_norm.weight", mcore_weight)

    torch.testing.assert_close(mcore_weight, expected_mcore_weight)
    torch.testing.assert_close(hf_weights[0], hf_weight)

    bridge.config.layernorm_zero_centered_gamma = False
    passthrough_weight = bridge._weight_to_mcore_format(
        "decoder.layers.0.self_attention.out_norm.weight", [hf_weight]
    )
    torch.testing.assert_close(passthrough_weight, hf_weight)


@pytest.mark.unit
def test_raw_qwen3_5_export_splits_new_gdn_in_proj_names():
    bridge_module = load_bridge_module()
    raw_module = load_raw_export_module()
    args = make_gdn_config()
    args.apply_layernorm_1p = True
    bridge = bridge_module.Qwen3_5Bridge.__new__(bridge_module.Qwen3_5Bridge)
    bridge.config = args

    qkv = torch.arange((2 * 8 + 16) * 8, dtype=torch.float32).view(32, 8)
    z = torch.arange(16 * 8, dtype=torch.float32).view(16, 8) + 1_000
    b = torch.arange(8 * 8, dtype=torch.float32).view(8, 8) + 2_000
    a = torch.arange(8 * 8, dtype=torch.float32).view(8, 8) + 3_000
    packed = bridge._weight_to_mcore_format("decoder.layers.0.self_attention.in_proj.weight", [qkv, z, b, a])

    converted = raw_module.convert_qwen3_5_to_hf(
        args,
        "module.module.decoder.layers.3.self_attention.in_proj.weight",
        packed,
    )

    assert [name for name, _ in converted] == [
        "model.language_model.layers.3.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.3.linear_attn.in_proj_z.weight",
        "model.language_model.layers.3.linear_attn.in_proj_b.weight",
        "model.language_model.layers.3.linear_attn.in_proj_a.weight",
    ]
    assert torch.equal(converted[0][1], qkv)
    assert torch.equal(converted[1][1], z)
    assert torch.equal(converted[2][1], b)
    assert torch.equal(converted[3][1], a)
