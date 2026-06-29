import importlib.util
import pathlib
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager


def _repo_path(*parts: str) -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2].joinpath(*parts)


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    if "." in name:
        parent_name, attr = name.rsplit(".", 1)
        parent = _ensure_module(parent_name)
        setattr(parent, attr, module)

    return module


def install_megatron_stubs() -> None:
    import torch

    class _SelfAttentionStub(torch.nn.Module):
        def get_query_key_value_tensors(self, *_args, **_kwargs):
            raise NotImplementedError

    _ensure_module("megatron")
    _ensure_module("megatron.core")
    fusions = _ensure_module("megatron.core.fusions")
    del fusions
    fused_bias_dropout = _ensure_module("megatron.core.fusions.fused_bias_dropout")
    fused_bias_dropout.get_bias_dropout_add = lambda *args, **kwargs: None

    _ensure_module("megatron.core.models")
    _ensure_module("megatron.core.models.gpt")
    gpt_model = _ensure_module("megatron.core.models.gpt.gpt_model")
    gpt_model.GPTModel = object

    _ensure_module("megatron.core.transformer")
    attention = _ensure_module("megatron.core.transformer.attention")
    attention.SelfAttention = _SelfAttentionStub
    attention.SelfAttentionSubmodules = type("SelfAttentionSubmodules", (), {})
    enums = _ensure_module("megatron.core.transformer.enums")
    enums.AttnMaskType = type("AttnMaskType", (), {"causal": "causal"})
    identity_op = _ensure_module("megatron.core.transformer.identity_op")
    identity_op.IdentityOp = type("IdentityOp", (), {})
    mlp = _ensure_module("megatron.core.transformer.mlp")
    mlp.MLP = type("MLP", (), {})
    mlp.MLPSubmodules = type("MLPSubmodules", (), {})
    moe_layer = _ensure_module("megatron.core.transformer.moe.moe_layer")
    moe_layer.BaseMoELayer = torch.nn.Module
    moe_layer.MoELayer = torch.nn.Module
    spec_utils = _ensure_module("megatron.core.transformer.spec_utils")
    spec_utils.import_module = lambda *args, **kwargs: None
    spec_utils.ModuleSpec = type("ModuleSpec", (), {})
    spec_utils.build_module = lambda *args, **kwargs: None
    transformer_layer = _ensure_module("megatron.core.transformer.transformer_layer")
    transformer_layer.TransformerLayer = object
    transformer_layer.TransformerLayerSubmodules = type("TransformerLayerSubmodules", (), {})
    transformer_layer.get_transformer_layer_offset = lambda config: 0
    utils = _ensure_module("megatron.core.utils")
    utils.make_viewless_tensor = lambda inp, **kwargs: inp

    training = _ensure_module("megatron.training")
    training.get_args = lambda: None
    arguments = _ensure_module("megatron.training.arguments")
    arguments.core_transformer_config_from_args = lambda *args, **kwargs: None


def install_mbridge_stubs() -> None:
    _ensure_module("mbridge")
    core = _ensure_module("mbridge.core")
    core.register_model = lambda *args, **kwargs: lambda cls: cls
    models = _ensure_module("mbridge.models")
    models.Gemma3Bridge = object
    gemma3_config = _ensure_module("mbridge.models.gemma3.transformer_config")
    gemma3_config.Gemma3TransformerConfig = type("Gemma3TransformerConfig", (), {})


@contextmanager
def _temporary_module(name: str, module: types.ModuleType) -> Iterator[None]:
    sentinel = object()
    original = sys.modules.get(name, sentinel)
    parent = sys.modules.get(name.rsplit(".", 1)[0]) if "." in name else None
    attr = name.rsplit(".", 1)[1] if "." in name else None
    original_attr = getattr(parent, attr, sentinel) if parent and attr else sentinel

    sys.modules[name] = module
    if parent and attr:
        setattr(parent, attr, module)
    try:
        yield
    finally:
        if original is sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

        if parent and attr:
            if original_attr is sentinel:
                if getattr(parent, attr, None) is module:
                    delattr(parent, attr)
            else:
                setattr(parent, attr, original_attr)


def load_gemma4_provider_module():
    install_megatron_stubs()
    gemma4_stub = types.ModuleType("slime_plugins.models.gemma4")
    gemma4_stub._load_hf_text_config = lambda path: None

    with _temporary_module("slime_plugins.models.gemma4", gemma4_stub):
        spec = importlib.util.spec_from_file_location(
            "_gemma4_provider_under_test",
            _repo_path("slime_plugins/models/gemma4_provider.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def load_gemma4_bridge_class():
    install_mbridge_stubs()
    gemma4_stub = types.ModuleType("slime_plugins.models.gemma4")
    gemma4_stub.get_rope_local_base_freq = lambda hf_text: None

    with _temporary_module("slime_plugins.models.gemma4", gemma4_stub):
        spec = importlib.util.spec_from_file_location(
            "_gemma4_bridge_under_test",
            _repo_path("slime_plugins/mbridge/gemma4.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Gemma4Bridge


def load_gemma4_model_module():
    install_megatron_stubs()
    install_mbridge_stubs()
    spec = importlib.util.spec_from_file_location(
        "_gemma4_model_under_test",
        _repo_path("slime_plugins/models/gemma4.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
