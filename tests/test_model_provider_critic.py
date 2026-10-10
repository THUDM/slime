import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0

HIDDEN, VOCAB = 4, 10


def _load_model_provider(monkeypatch):
    modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.core": types.ModuleType("megatron.core"),
        "megatron.core.models": types.ModuleType("megatron.core.models"),
        "megatron.core.models.gpt": types.ModuleType("megatron.core.models.gpt"),
        "megatron.core.models.gpt.gpt_layer_specs": types.ModuleType("megatron.core.models.gpt.gpt_layer_specs"),
        "megatron.core.transformer": types.ModuleType("megatron.core.transformer"),
        "megatron.core.transformer.spec_utils": types.ModuleType("megatron.core.transformer.spec_utils"),
        "megatron.core.transformer.transformer_config": types.ModuleType(
            "megatron.core.transformer.transformer_config"
        ),
        "megatron.training": types.ModuleType("megatron.training"),
        "megatron.training.arguments": types.ModuleType("megatron.training.arguments"),
        "slime.utils.misc": types.ModuleType("slime.utils.misc"),
    }
    modules["megatron.core"].tensor_parallel = types.SimpleNamespace()
    modules["megatron.core.models.gpt"].GPTModel = torch.nn.Module
    layer_specs = modules["megatron.core.models.gpt.gpt_layer_specs"]
    layer_specs.get_gpt_decoder_block_spec = lambda *args, **kwargs: None
    layer_specs.get_gpt_layer_local_spec = lambda *args, **kwargs: None
    layer_specs.get_gpt_layer_with_transformer_engine_spec = lambda *args, **kwargs: None
    modules["megatron.core.transformer.spec_utils"].import_module = lambda value: value
    modules["megatron.core.transformer.transformer_config"].TransformerConfig = object
    config = types.SimpleNamespace(hidden_size=HIDDEN, sequence_parallel=False, init_method_std=0.02)
    modules["megatron.training.arguments"].core_transformer_config_from_args = lambda args: config
    modules["slime.utils.misc"].load_function = lambda value: value
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "model_provider.py"
    module_name = "test_model_provider_critic_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layer = torch.nn.Linear(HIDDEN, VOCAB)

    def forward(self, hidden):
        # LinearForLastLayer returns (logits, None), like Megatron's ColumnParallelLinear.
        out = self.output_layer(hidden)
        return out[0] if isinstance(out, tuple) else out


class _VLMWrapper(torch.nn.Module):
    """Like Qwen3_5VLModel: forward() delegates to language_model and its LM head."""

    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()

    def forward(self, hidden):
        return self.language_model(hidden)


def _critic_provider(model_provider, build):
    def spec(args, config, vp_stage):
        def provider(pre_process=True, post_process=True, vp_stage=None):
            return build()

        return provider

    args = types.SimpleNamespace(transformer_impl="local", spec=spec, freeze_indexer=False)
    return model_provider._get_model_provider_func(args, role="critic")


@pytest.mark.unit
def test_critic_value_head_goes_on_vlm_language_model(monkeypatch):
    model_provider = _load_model_provider(monkeypatch)

    model = _critic_provider(model_provider, _VLMWrapper)()

    assert isinstance(model.language_model.output_layer, model_provider.LinearForLastLayer)
    assert model.language_model.output_layer.out_features == 1
    assert not hasattr(model, "output_layer")
    assert model(torch.zeros(3, HIDDEN)).shape == (3, 1)


@pytest.mark.unit
def test_critic_value_head_replaces_plain_model_output_layer(monkeypatch):
    model_provider = _load_model_provider(monkeypatch)

    model = _critic_provider(model_provider, _LanguageModel)()

    assert isinstance(model.output_layer, model_provider.LinearForLastLayer)
    assert model.output_layer.out_features == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
