"""HF ↔ Mcore QKV weight conversion roundtrip tests.

We already have forward-only tests for ``convert_gemma4_to_hf`` in
``test_gemma4_bridge.py``. This file closes the loop: for every attention
layer type (sliding local, global K=V), start from HF ``[q, k, v]`` tensors,
pack into the Mcore ``linear_qkv.weight`` layout via ``Gemma4Bridge.
_weight_to_mcore_format``, unpack via ``convert_gemma4_to_hf``, and confirm
bit-identity with the originals.

Guards against silent drift in either direction of the converter as we
evolve Gemma4 support.
"""

import importlib
import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.mbridge.gemma4 import Gemma4Bridge


def _load_convert_module():
    try:
        return importlib.import_module("slime.backends.megatron_utils.megatron_to_hf.gemma4")
    except ImportError:
        pass
    repo_path = pathlib.Path(__file__).resolve().parents[2] / (
        "slime/backends/megatron_utils/megatron_to_hf/gemma4.py"
    )
    if not repo_path.exists():
        pytest.skip(f"convert module not found at {repo_path}")
    spec = importlib.util.spec_from_file_location("_gemma4_conv_rt", repo_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Gemma4-31B canonical config.
CFG_31B = SimpleNamespace(
    hidden_size=5376,
    num_attention_heads=32,
    head_dim=256,
    num_key_value_heads=16,
    global_head_dim=512,
    num_global_key_value_heads=4,
    num_hidden_layers=60,
    attention_k_eq_v=True,
    # layer_types=["sliding_attention"] * 5 + ["full_attention"], repeated.
    layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 10,
)
_GLOBAL_LAYERS_31B = {i for i, t in enumerate(CFG_31B.layer_types) if t == "full_attention"}


def _build_bridge_stub(cfg):
    """Make a Gemma4Bridge instance without going through its real __init__
    (which expects an HF checkpoint on disk). We set just enough attributes
    for ``_weight_to_mcore_format`` to run."""
    b = object.__new__(Gemma4Bridge)
    b._GLOBAL_ATTN_LAYERS = {
        i for i, t in enumerate(cfg.layer_types) if t == "full_attention"
    }
    b.hf_config = SimpleNamespace(text_config=cfg)
    return b


def _prime_convert_config(conv):
    conv._config_cache["config"] = {
        "global_attn_layers": _GLOBAL_LAYERS_31B,
        "local_head_dim": CFG_31B.head_dim,
        "global_head_dim": CFG_31B.global_head_dim,
        "num_attention_heads": CFG_31B.num_attention_heads,
        "local_num_kv_heads": CFG_31B.num_key_value_heads,
        "global_num_kv_heads": CFG_31B.num_global_key_value_heads,
        "hidden_size": CFG_31B.hidden_size,
    }


def test_sliding_layer_qkv_roundtrip():
    """Start from HF [q, k, v] → pack into Mcore layout → unpack → same tensors."""
    torch.manual_seed(0)
    conv = _load_convert_module()
    _prime_convert_config(conv)
    bridge = _build_bridge_stub(CFG_31B)

    # Layer 0 is sliding in our config.
    layer_idx = 0
    q = torch.randn(CFG_31B.num_attention_heads * CFG_31B.head_dim, CFG_31B.hidden_size)
    k = torch.randn(CFG_31B.num_key_value_heads * CFG_31B.head_dim, CFG_31B.hidden_size)
    v = torch.randn(CFG_31B.num_key_value_heads * CFG_31B.head_dim, CFG_31B.hidden_size)

    mcore_name = f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"
    packed = bridge._weight_to_mcore_format(mcore_name, [q, k, v])
    assert packed.shape == (
        CFG_31B.num_attention_heads * CFG_31B.head_dim
        + 2 * CFG_31B.num_key_value_heads * CFG_31B.head_dim,
        CFG_31B.hidden_size,
    )

    args = SimpleNamespace(hf_checkpoint="/nonexistent")
    emitted = conv.convert_gemma4_to_hf(
        args, f"module.module.{mcore_name}", packed,
    )
    out = dict(emitted)
    assert set(out) == {
        f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight",
        f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight",
        f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight",
    }
    assert torch.allclose(
        out[f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight"], q
    )
    assert torch.allclose(
        out[f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight"], k
    )
    assert torch.allclose(
        out[f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight"], v
    )


def test_global_k_eq_v_layer_qkv_roundtrip():
    """K=V global layer: HF ships [q, k] only. The bridge reconstructs V
    from K during pack; convert_gemma4_to_hf unpacks and emits [q, k] only
    (no v_proj). Roundtrip must still recover q and k bit-identically."""
    torch.manual_seed(1)
    conv = _load_convert_module()
    _prime_convert_config(conv)
    bridge = _build_bridge_stub(CFG_31B)

    layer_idx = 5  # global
    assert layer_idx in _GLOBAL_LAYERS_31B

    q = torch.randn(CFG_31B.num_attention_heads * CFG_31B.global_head_dim, CFG_31B.hidden_size)
    k = torch.randn(CFG_31B.num_global_key_value_heads * CFG_31B.global_head_dim, CFG_31B.hidden_size)

    mcore_name = f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"
    packed = bridge._weight_to_mcore_format(mcore_name, [q, k])
    # Packed shape: num_kv * (q_per_kv + 2) * head_dim rows, with V=K slot.
    q_per_kv = CFG_31B.num_attention_heads // CFG_31B.num_global_key_value_heads
    expected_rows = (
        CFG_31B.num_global_key_value_heads
        * (q_per_kv + 2)
        * CFG_31B.global_head_dim
    )
    assert packed.shape == (expected_rows, CFG_31B.hidden_size)

    args = SimpleNamespace(hf_checkpoint="/nonexistent")
    emitted = conv.convert_gemma4_to_hf(
        args, f"module.module.{mcore_name}", packed,
    )
    out = dict(emitted)
    # Global K=V: only q and k are emitted.
    assert set(out) == {
        f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight",
        f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight",
    }
    assert torch.allclose(
        out[f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight"], q
    )
    assert torch.allclose(
        out[f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight"], k
    )


def test_sliding_layer_roundtrip_rejects_wrong_shape():
    """The new shape assertions (B6/B8) fire when q_proj rows don't match
    num_kv_heads * group_dim — a common symptom of a miscounted config."""
    bridge = _build_bridge_stub(CFG_31B)

    # Wrong: use GLOBAL head_dim on a SLIDING layer → q rows won't match.
    q_bad = torch.randn(CFG_31B.num_attention_heads * CFG_31B.global_head_dim, CFG_31B.hidden_size)
    k_bad = torch.randn(CFG_31B.num_key_value_heads * CFG_31B.head_dim, CFG_31B.hidden_size)
    v_bad = torch.randn(CFG_31B.num_key_value_heads * CFG_31B.head_dim, CFG_31B.hidden_size)

    with pytest.raises(AssertionError, match="q_proj rows"):
        bridge._weight_to_mcore_format(
            "decoder.layers.0.self_attention.linear_qkv.weight",
            [q_bad, k_bad, v_bad],
        )


def test_mlp_fc1_asserts_wrong_count():
    """linear_fc1.weight expects exactly [gate_proj, up_proj] from HF. Passing
    3 tensors hits the assert (passing 1 is short-circuited to a pass-through
    earlier in _weight_to_mcore_format for unrelated single-tensor cases)."""
    bridge = _build_bridge_stub(CFG_31B)
    with pytest.raises(AssertionError, match="linear_fc1.weight expects"):
        bridge._weight_to_mcore_format(
            "decoder.layers.0.mlp.linear_fc1.weight",
            [torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)],
        )


def test_mlp_fc1_pack_concatenates_gate_up():
    """linear_fc1.weight packs HF [gate_proj, up_proj] → [gate; up] along dim 0."""
    bridge = _build_bridge_stub(CFG_31B)
    gate = torch.randn(CFG_31B.hidden_size, CFG_31B.hidden_size)
    up = torch.randn(CFG_31B.hidden_size, CFG_31B.hidden_size)
    packed = bridge._weight_to_mcore_format(
        "decoder.layers.0.mlp.linear_fc1.weight", [gate, up],
    )
    assert packed.shape == (2 * CFG_31B.hidden_size, CFG_31B.hidden_size)
    assert torch.equal(packed[:CFG_31B.hidden_size], gate)
    assert torch.equal(packed[CFG_31B.hidden_size:], up)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
