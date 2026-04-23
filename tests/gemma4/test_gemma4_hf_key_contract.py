"""Contract test: every HF Gemma4 state-dict key must be producible by our
megatron→HF converter.

Why this exists: ``convert_gemma4_to_hf`` emits HF-format tensors that are
then consumed by sglang (during rollout weight-update) and by
``convert_torch_dist_to_hf.py`` (for offline eval export). Both downstream
consumers walk HF's *expected* state-dict keys and look them up in the
converter output. If a future change to the Megatron side drops a key that
HF still wants, the tests in ``test_gemma4_bridge.py`` — which only exercise
specific mcore→HF mappings in isolation — won't notice; the first symptom is
a tensor-shape or missing-key crash at weight load, hours into a training
run.

This test pins the contract by:
  1. Instantiating a tiny 2-layer (1 sliding + 1 global) MoE Gemma4 via HF,
     snapshotting its state-dict key set (the "HF expected" set).
  2. Running a synthetic Megatron-side key list through
     ``convert_gemma4_to_hf``, gathering the emitted HF key set.
  3. Asserting the emitted set covers every HF key (modulo the known
     ``v_proj`` omission on K=V global layers, where HF itself sets
     ``v_proj = None`` and omits the parameter from its state_dict).

Anything that fails this contract would silently break a training run.
"""
from types import SimpleNamespace

import pytest
import torch


# Synthesized Megatron state-dict keys for the tiny 2-layer MoE config below.
# Matches what ``get_model(model_provider, ...)`` produces at load time —
# enumerated here rather than instantiated to keep this test CPU-only and
# independent of Megatron init.
def _mcore_keys_tiny_moe(num_experts: int = 2) -> list[str]:
    base = [
        "module.module.embedding.word_embeddings.weight",
        "module.module.decoder.final_layernorm.weight",
    ]
    # Output layer is tied to embedding and shares key in HF; Megatron saves
    # it separately.
    base.append("module.module.output_layer.weight")
    for layer_idx in (0, 1):
        prefix = f"module.module.decoder.layers.{layer_idx}"
        base.extend([
            # Attention
            f"{prefix}.self_attention.linear_qkv.weight",
            f"{prefix}.self_attention.linear_qkv.layer_norm_weight",
            f"{prefix}.self_attention.linear_proj.weight",
            f"{prefix}.self_attention.q_layernorm.weight",
            f"{prefix}.self_attention.k_layernorm.weight",
            # post_attention_layernorm lives outside TE-fused norm paths.
            f"{prefix}.post_attention_layernorm.weight",
            # Per-layer scalar buffer (loaded from HF ckpt via provider
            # hook; saved by Megatron alongside trainable params).
            f"{prefix}.layer_scalar",
            # Dense-MLP sibling (parallel to the MoE block in 26B-A4B).
            f"{prefix}.dense_mlp.linear_fc1.weight",
            f"{prefix}.dense_mlp.linear_fc1.layer_norm_weight",
            f"{prefix}.dense_mlp.linear_fc2.weight",
            # Fused pre/post FFN layernorms around the dense+MoE add.
            f"{prefix}.pre_mlp_layernorm.weight",
            f"{prefix}.post_feedforward_layernorm.weight",
            f"{prefix}.post_feedforward_layernorm_1.weight",
            f"{prefix}.post_feedforward_layernorm_2.weight",
            # MoE block internals — pre_feedforward_layernorm_2 moved inside
            # Gemma4MoELayer in the current code, so it lives under .mlp.*.
            f"{prefix}.mlp.pre_feedforward_layernorm_2.weight",
            # Router
            f"{prefix}.mlp.router.proj.weight",
            f"{prefix}.mlp.router.scale",
            f"{prefix}.mlp.router.per_expert_scale",
        ])
        # Per-expert weights (names use global expert indices; our converter
        # buffers + flushes to stacked 3D tensors).
        for e in range(num_experts):
            base.extend([
                f"{prefix}.mlp.experts.linear_fc1.weight{e}",
                f"{prefix}.mlp.experts.linear_fc2.weight{e}",
            ])
    return base


def _build_tiny_hf_model():
    """Build a 2-layer (1 sliding + 1 global) MoE Gemma4 via HF; return its
    ``model.language_model`` state-dict keys.

    Intentionally ``hidden_size_per_layer_input=0`` to disable the
    per-layer-input gate (which our plugin doesn't implement and our
    converter doesn't map); and ``attention_k_eq_v=True`` to match the
    real 26B/31B configs (so HF's global-layer v_proj is ``None`` and
    absent from state_dict).
    """
    from transformers.models.gemma4 import (
        configuration_gemma4 as C, modeling_gemma4 as M,
    )

    text_cfg = C.Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=16, global_head_dim=32,
        sliding_window=64, rope_theta=10000.0,
        layer_types=["sliding_attention", "full_attention"],
        enable_moe_block=True, num_experts=2, moe_intermediate_size=48,
        top_k_experts=2,
        hidden_size_per_layer_input=0,
        attention_k_eq_v=True,
    )
    full_cfg = C.Gemma4Config(
        text_config=text_cfg.to_dict(), vision_config=None, audio_config=None,
    )
    hf_model = M.Gemma4ForConditionalGeneration(full_cfg)
    return set(
        k for k in hf_model.state_dict().keys() if "language_model" in k
    )


def test_converter_emits_every_hf_key():
    """Run every synthesized Megatron key through ``convert_gemma4_to_hf``;
    assert the union of emitted HF keys covers HF's expected state_dict."""
    transformers_gemma4 = pytest.importorskip("transformers.models.gemma4")
    del transformers_gemma4  # only needed to gate

    from slime.backends.megatron_utils.megatron_to_hf import gemma4 as conv

    # Seed the converter's module-global config cache with the tiny config
    # directly (avoids instantiating AutoConfig, which needs a ckpt dir).
    # _get_config is the only reader; it returns early when "config" in
    # _config_cache.
    conv._config_cache["config"] = {
        "global_attn_layers": {1},  # layer 1 is full_attention
        "local_head_dim": 16,
        "global_head_dim": 32,
        "num_attention_heads": 4,
        "local_num_kv_heads": 2,
        "global_num_kv_heads": 2,
        "hidden_size": 32,
        "num_experts": 2,
    }
    # Clear expert-flush buffers so a prior test run doesn't leak state.
    conv.reset_expert_buffers()

    args = SimpleNamespace()
    # Dummy tensors sized to match the converter's view-reshape math. The
    # converter only cares about the *names* for this test, but its QKV path
    # calls .view() with real shapes, so we need plausible tensors.
    def _fake_tensor_for(name: str) -> torch.Tensor:
        if name.endswith("self_attention.linear_qkv.weight"):
            # Packed [num_kv_heads * (q_per_kv + 2) * head_dim, hidden_size]
            # For sliding: 2 kv_heads * (2 + 2) * 16 = 128, hidden=32.
            # For global: 2 kv_heads * (2 + 2) * 32 = 256, hidden=32.
            if "layers.1" in name:
                return torch.zeros(256, 32)
            return torch.zeros(128, 32)
        if name.endswith("self_attention.linear_proj.weight"):
            return torch.zeros(32, 64)  # o_proj [hidden, num_q_heads*head_dim]
        if "dense_mlp.linear_fc1.weight" in name:
            return torch.zeros(128, 32)  # gate||up packed: [2*inter, hidden]
        if "dense_mlp.linear_fc2.weight" in name:
            return torch.zeros(32, 64)
        if "mlp.router.proj.weight" in name:
            return torch.zeros(2, 32)
        if "mlp.router.scale" in name or "mlp.router.per_expert_scale" in name:
            return torch.zeros(2)
        if "experts.linear_fc1.weight" in name:
            return torch.zeros(96, 32)  # packed 2*moe_inter = 96
        if "experts.linear_fc2.weight" in name:
            return torch.zeros(32, 48)
        if "embedding.word_embeddings" in name or "output_layer" in name:
            return torch.zeros(64, 32)
        if "layer_scalar" in name:
            return torch.tensor([1.0])
        # Layernorm-family defaults.
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

    # Also surface extras (emitted but HF doesn't want) as a warning — not a
    # hard failure, since the converter may legitimately emit aliases (e.g.
    # tied embeddings maps output_layer → embed_tokens). Print them so a
    # reviewer can sanity-check.
    extras = emitted - expected
    if extras:
        print(
            f"[info] converter emits {len(extras)} key(s) HF doesn't have in "
            f"its state_dict (tied embeddings / aliases):\n  "
            + "\n  ".join(sorted(extras))
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
