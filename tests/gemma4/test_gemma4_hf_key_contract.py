import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
import torch


def _load_convert_module():
    repo_path = pathlib.Path(__file__).resolve().parents[2] / (
        "slime/backends/megatron_utils/megatron_to_hf/gemma4.py"
    )
    spec = importlib.util.spec_from_file_location("_gemma4_key_contract_converter", repo_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mcore_keys_tiny_moe(num_experts: int = 2) -> list[str]:
    base = [
        "module.module.embedding.word_embeddings.weight",
        "module.module.decoder.final_layernorm.weight",
    ]
    base.append("module.module.output_layer.weight")
    for layer_idx in (0, 1):
        prefix = f"module.module.decoder.layers.{layer_idx}"
        base.extend(
            [
                f"{prefix}.self_attention.linear_qkv.weight",
                f"{prefix}.self_attention.linear_qkv.layer_norm_weight",
                f"{prefix}.self_attention.linear_proj.weight",
                f"{prefix}.self_attention.q_layernorm.weight",
                f"{prefix}.self_attention.k_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.layer_scalar",
                f"{prefix}.dense_mlp.linear_fc1.weight",
                f"{prefix}.dense_mlp.linear_fc1.layer_norm_weight",
                f"{prefix}.dense_mlp.linear_fc2.weight",
                f"{prefix}.pre_mlp_layernorm.weight",
                f"{prefix}.post_feedforward_layernorm.weight",
                f"{prefix}.post_feedforward_layernorm_1.weight",
                f"{prefix}.post_feedforward_layernorm_2.weight",
                f"{prefix}.mlp.pre_feedforward_layernorm_2.weight",
                f"{prefix}.mlp.router.proj.weight",
                f"{prefix}.mlp.router.scale",
                f"{prefix}.mlp.router.per_expert_scale",
            ]
        )
        for e in range(num_experts):
            base.extend(
                [
                    f"{prefix}.mlp.experts.linear_fc1.weight{e}",
                    f"{prefix}.mlp.experts.linear_fc2.weight{e}",
                ]
            )
    return base


def _build_tiny_hf_model():
    from transformers.models.gemma4 import configuration_gemma4 as C
    from transformers.models.gemma4 import modeling_gemma4 as M

    text_cfg = C.Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        sliding_window=64,
        rope_theta=10000.0,
        layer_types=["sliding_attention", "full_attention"],
        enable_moe_block=True,
        num_experts=2,
        moe_intermediate_size=48,
        top_k_experts=2,
        hidden_size_per_layer_input=0,
        attention_k_eq_v=True,
    )
    full_cfg = C.Gemma4Config(
        text_config=text_cfg.to_dict(),
        vision_config=None,
        audio_config=None,
    )
    hf_model = M.Gemma4ForConditionalGeneration(full_cfg)
    return set(k for k in hf_model.state_dict().keys() if "language_model" in k)


def test_converter_emits_every_hf_key():
    transformers_gemma4 = pytest.importorskip("transformers.models.gemma4")
    del transformers_gemma4  # only needed to gate

    conv = _load_convert_module()

    conv._config_cache["/nonexistent"] = {
        "global_attn_layers": {1},  # layer 1 is full_attention
        "local_head_dim": 16,
        "global_head_dim": 32,
        "num_attention_heads": 4,
        "local_num_kv_heads": 2,
        "global_num_kv_heads": 2,
        "hidden_size": 32,
        "num_experts": 2,
    }
    conv.reset_expert_buffers()

    args = SimpleNamespace(hf_checkpoint="/nonexistent")

    def _fake_tensor_for(name: str) -> torch.Tensor:
        if name.endswith("self_attention.linear_qkv.weight"):
            if "layers.1" in name:
                return torch.zeros(256, 32)
            return torch.zeros(128, 32)
        if name.endswith("self_attention.linear_proj.weight"):
            return torch.zeros(32, 64)
        if "dense_mlp.linear_fc1.weight" in name:
            return torch.zeros(128, 32)
        if "dense_mlp.linear_fc2.weight" in name:
            return torch.zeros(32, 64)
        if "mlp.router.proj.weight" in name:
            return torch.zeros(2, 32)
        if "mlp.router.scale" in name or "mlp.router.per_expert_scale" in name:
            return torch.zeros(2)
        if "experts.linear_fc1.weight" in name:
            return torch.zeros(96, 32)
        if "experts.linear_fc2.weight" in name:
            return torch.zeros(32, 48)
        if "embedding.word_embeddings" in name or "output_layer" in name:
            return torch.zeros(64, 32)
        if "layer_scalar" in name:
            return torch.tensor([1.0])
        return torch.zeros(32)

    emitted: set[str] = set()
    for mcore_name in _mcore_keys_tiny_moe(num_experts=2):
        t = _fake_tensor_for(mcore_name)
        out = conv.convert_gemma4_to_hf(args, mcore_name, t)
        for hf_name, _hf_param in out:
            emitted.add(hf_name)

    expected = _build_tiny_hf_model()

    missing = expected - emitted
    assert not missing, (
        f"HF expects {len(missing)} key(s) the converter never emits; this "
        f"would surface as a weight-load crash or silently-random weights in "
        f"sglang. Missing:\n  " + "\n  ".join(sorted(missing))
    )
