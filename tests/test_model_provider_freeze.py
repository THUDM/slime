import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0


class _FakeGPTModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs["config"]
        self.model_kwargs = kwargs


class _FakeCompactLogitsGPTModel(_FakeGPTModel):
    pass


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
        "slime.backends.megatron_utils.compact_logits": types.ModuleType(
            "slime.backends.megatron_utils.compact_logits"
        ),
        "slime.utils.misc": types.ModuleType("slime.utils.misc"),
    }
    modules["megatron.core"].tensor_parallel = types.SimpleNamespace()
    modules["megatron.core.models.gpt"].GPTModel = _FakeGPTModel
    layer_specs = modules["megatron.core.models.gpt.gpt_layer_specs"]
    layer_specs.get_gpt_decoder_block_spec = lambda *args, **kwargs: None
    layer_specs.get_gpt_layer_local_spec = lambda *args, **kwargs: None
    layer_specs.get_gpt_layer_with_transformer_engine_spec = lambda *args, **kwargs: None
    modules["megatron.core.transformer.spec_utils"].import_module = lambda value: value
    modules["megatron.core.transformer.transformer_config"].TransformerConfig = object
    modules["megatron.training.arguments"].core_transformer_config_from_args = lambda args: types.SimpleNamespace(
        hidden_size=8,
        sequence_parallel=False,
    )
    compact_logits = modules["slime.backends.megatron_utils.compact_logits"]
    compact_logits.CompactLogitsGPTModel = _FakeCompactLogitsGPTModel
    compact_logits.can_compact_actor_logits = lambda args: bool(
        getattr(args, "compact_actor_logits", False) and args.loss_type in {"policy_loss", "sft_loss"}
    )
    modules["slime.utils.misc"].load_function = lambda value: value
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "model_provider.py"
    module_name = "test_model_provider_freeze_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _provider_args(**overrides):
    values = {
        "compact_actor_logits": True,
        "custom_model_provider_path": None,
        "fp16_lm_cross_entropy": False,
        "fp8_param_gather": False,
        "loss_type": "policy_loss",
        "max_position_embeddings": 128,
        "moe_grouped_gemm": False,
        "moe_use_legacy_grouped_gemm": False,
        "mtp_num_layers": None,
        "multi_latent_attention": False,
        "num_experts": None,
        "padded_vocab_size": 32,
        "position_embedding_type": "rope",
        "qk_layernorm": False,
        "rotary_base": 10000,
        "rotary_percent": 1.0,
        "spec": None,
        "transformer_impl": "local",
        "untie_embeddings_and_output_weights": False,
        "use_rope_scaling": False,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class _GLMIndexerAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq_b = torch.nn.Linear(2, 2, bias=False)
        self.wk = torch.nn.Linear(2, 2, bias=False)
        self.k_norm = torch.nn.LayerNorm(2)
        self.weights_proj = torch.nn.Linear(2, 2, bias=False)
        self.linear_q_down_proj = torch.nn.Linear(2, 2, bias=False)


class _UpstreamDSAAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core_attention = torch.nn.Module()
        self.core_attention.indexer = torch.nn.Module()
        self.core_attention.indexer.linear_wq_b = torch.nn.Linear(2, 2, bias=False)
        self.core_attention.indexer.linear_wk = torch.nn.Linear(2, 2, bias=False)
        self.core_attention.indexer.k_norm = torch.nn.LayerNorm(2)
        self.core_attention.indexer.linear_weights_proj = torch.nn.Linear(2, 2, bias=False)
        self.core_attention.regular_projection = torch.nn.Linear(2, 2, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.self_attention = attention
        self.mlp = torch.nn.Linear(2, 2, bias=False)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(_GLMIndexerAttention()), _Layer(_UpstreamDSAAttention())])
        self.output_layer = torch.nn.Linear(2, 2, bias=False)


@pytest.mark.unit
def test_freeze_indexer_covers_glm_and_upstream_dsa_names(monkeypatch):
    model_provider = _load_model_provider(monkeypatch)
    model = _Model()
    args = types.SimpleNamespace(
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        freeze_indexer=True,
    )

    model_provider.freeze_model_params(model, args)

    frozen = set(model._slime_frozen_indexer_param_names)
    assert frozen
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert not parameter.requires_grad, name
        else:
            assert parameter.requires_grad, name
    assert "layers.0.self_attention.linear_q_down_proj.weight" not in frozen
    assert "layers.1.self_attention.core_attention.regular_projection.weight" not in frozen
    assert "layers.0.mlp.weight" not in frozen
    assert "output_layer.weight" not in frozen


@pytest.mark.unit
def test_freeze_indexer_rejects_unrecognized_attention(monkeypatch):
    model_provider = _load_model_provider(monkeypatch)
    model = _Layer(torch.nn.MultiheadAttention(2, 1))
    args = types.SimpleNamespace(
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        freeze_indexer=True,
    )

    with pytest.raises(RuntimeError, match="no recognized DSA indexer"):
        model_provider.freeze_model_params(model, args)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("role", "post_process", "compact_actor_logits", "loss_type", "expected_type"),
    [
        ("actor", True, True, "policy_loss", _FakeCompactLogitsGPTModel),
        ("actor", False, True, "policy_loss", _FakeGPTModel),
        ("critic", True, True, "policy_loss", _FakeGPTModel),
        ("actor", True, False, "policy_loss", _FakeGPTModel),
        ("actor", True, True, "custom_loss", _FakeGPTModel),
    ],
)
def test_compact_logits_subclass_is_owned_by_eligible_actor_postprocess_chunk(
    monkeypatch,
    role,
    post_process,
    compact_actor_logits,
    loss_type,
    expected_type,
):
    model_provider = _load_model_provider(monkeypatch)
    args = _provider_args(compact_actor_logits=compact_actor_logits, loss_type=loss_type)

    provider = model_provider._get_model_provider_func(args, role)
    model = provider(pre_process=True, post_process=post_process)

    assert type(model) is expected_type


@pytest.mark.unit
def test_compact_logits_does_not_replace_custom_model_providers(monkeypatch):
    model_provider = _load_model_provider(monkeypatch)
    custom_model = _FakeGPTModel(config=types.SimpleNamespace(hidden_size=8))
    model_provider.load_function = lambda _path: lambda **_kwargs: custom_model
    args = _provider_args(custom_model_provider_path="custom.provider")

    provider = model_provider._get_model_provider_func(args, "actor")

    assert provider() is custom_model


@pytest.mark.unit
def test_compact_logits_does_not_replace_callable_spec_model_provider(monkeypatch):
    model_provider = _load_model_provider(monkeypatch)
    custom_model = _FakeGPTModel(config=types.SimpleNamespace(hidden_size=8))

    def custom_provider(pre_process=True, post_process=True, vp_stage=None):
        return custom_model

    model_provider.import_module = lambda _path: lambda _args, _config, _vp_stage: custom_provider
    args = _provider_args(spec="custom.spec")

    provider = model_provider._get_model_provider_func(args, "actor")

    assert provider() is custom_model


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
