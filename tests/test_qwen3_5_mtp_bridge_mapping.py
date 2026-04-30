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
    gpt_layer_specs_mod.get_gpt_mtp_block_spec = lambda _config, transformer_layer_spec, **_kwargs: (
        "mtp-spec",
        transformer_layer_spec,
    )

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
def test_tied_qwen3_5_uses_language_embedding_for_output_layer():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.hf_config = types.SimpleNamespace(text_config=types.SimpleNamespace(tie_word_embeddings=True))

    bridge._adjust_mapping_for_shared_weights()

    assert bridge._DIRECT_MAPPING["output_layer.weight"] == "model.language_model.embed_tokens.weight"
    assert module.Qwen3_5Bridge._DIRECT_MAPPING["output_layer.weight"] == "lm_head.weight"


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
