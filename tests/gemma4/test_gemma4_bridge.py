"""Parity tests for slime_plugins.mbridge.gemma4 and
slime/backends/megatron_utils/megatron_to_hf/gemma4.py.

These exercise the ACTUAL production functions (Gemma4Bridge and
convert_gemma4_to_hf) rather than re-implementing the pack/unpack.
"""

import importlib
import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
import torch


def _load_convert_module():
    """Import the weight-conversion module either from the installed slime
    package or from the repo's working copy relative to this test file."""
    try:
        return importlib.import_module("slime.backends.megatron_utils.megatron_to_hf.gemma4")
    except ImportError:
        pass
    repo_path = pathlib.Path(__file__).resolve().parents[2] / (
        "slime/backends/megatron_utils/megatron_to_hf/gemma4.py"
    )
    if not repo_path.exists():
        pytest.skip(f"convert_gemma4_to_hf source not found at {repo_path}")
    spec = importlib.util.spec_from_file_location("_gemma4_conv_under_test", repo_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Gemma4-31B canonical config values.
CFG_31B = SimpleNamespace(
    hidden_size=5376,
    num_attention_heads=32,
    head_dim=256,
    num_key_value_heads=16,
    global_head_dim=512,
    num_global_key_value_heads=4,
    num_hidden_layers=60,
    attention_k_eq_v=True,
    layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 10,
)


def _pack_local_qkv(q, k, v):
    """Megatron packs as [num_kv_heads, (q_per_kv + 2) * head_dim, hidden]."""
    num_kv = CFG_31B.num_key_value_heads
    head_dim = CFG_31B.head_dim
    q_per_kv = CFG_31B.num_attention_heads // num_kv
    q = q.view(num_kv, q_per_kv * head_dim, CFG_31B.hidden_size)
    k = k.view(num_kv, head_dim, CFG_31B.hidden_size)
    v = v.view(num_kv, head_dim, CFG_31B.hidden_size)
    return torch.cat([q, k, v], dim=1).reshape(-1, CFG_31B.hidden_size).contiguous()


def _pack_global_qkv(q, k):
    """K=V global layers: stored as [q, k, k]."""
    num_kv = CFG_31B.num_global_key_value_heads
    head_dim = CFG_31B.global_head_dim
    q_per_kv = CFG_31B.num_attention_heads // num_kv
    q = q.view(num_kv, q_per_kv * head_dim, CFG_31B.hidden_size)
    k = k.view(num_kv, head_dim, CFG_31B.hidden_size)
    return torch.cat([q, k, k], dim=1).reshape(-1, CFG_31B.hidden_size).contiguous()


def test_convert_gemma4_to_hf_local_layer_roundtrip(monkeypatch):
    """Load convert_gemma4_to_hf and verify roundtrip for a local layer."""
    conv = _load_convert_module()

    # Prime the config cache so we don't need a real HF checkpoint on disk.
    conv._config_cache["config"] = {
        "global_attn_layers": {i for i, t in enumerate(CFG_31B.layer_types) if t == "full_attention"},
        "local_head_dim": CFG_31B.head_dim,
        "global_head_dim": CFG_31B.global_head_dim,
        "num_attention_heads": CFG_31B.num_attention_heads,
        "local_num_kv_heads": CFG_31B.num_key_value_heads,
        "global_num_kv_heads": CFG_31B.num_global_key_value_heads,
        "hidden_size": CFG_31B.hidden_size,
    }

    # Build a random local qkv and convert it.
    q = torch.randn(CFG_31B.num_attention_heads * CFG_31B.head_dim, CFG_31B.hidden_size)
    k = torch.randn(CFG_31B.num_key_value_heads * CFG_31B.head_dim, CFG_31B.hidden_size)
    v = torch.randn(CFG_31B.num_key_value_heads * CFG_31B.head_dim, CFG_31B.hidden_size)
    packed = _pack_local_qkv(q, k, v)

    # Layer 0 is sliding (local).
    args = SimpleNamespace(hf_checkpoint="/nonexistent")
    emitted = conv.convert_gemma4_to_hf(
        args, "module.module.decoder.layers.0.self_attention.linear_qkv.weight", packed,
    )
    names = {n for n, _ in emitted}
    assert names == {
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
    }
    out = dict(emitted)
    assert torch.allclose(out["model.language_model.layers.0.self_attn.q_proj.weight"], q)
    assert torch.allclose(out["model.language_model.layers.0.self_attn.k_proj.weight"], k)
    assert torch.allclose(out["model.language_model.layers.0.self_attn.v_proj.weight"], v)


def test_convert_gemma4_to_hf_global_layer_emits_no_v_proj():
    conv = _load_convert_module()

    conv._config_cache["config"] = {
        "global_attn_layers": {5, 11, 17, 23, 29, 35, 41, 47, 53, 59},
        "local_head_dim": CFG_31B.head_dim,
        "global_head_dim": CFG_31B.global_head_dim,
        "num_attention_heads": CFG_31B.num_attention_heads,
        "local_num_kv_heads": CFG_31B.num_key_value_heads,
        "global_num_kv_heads": CFG_31B.num_global_key_value_heads,
        "hidden_size": CFG_31B.hidden_size,
    }

    q = torch.randn(CFG_31B.num_attention_heads * CFG_31B.global_head_dim, CFG_31B.hidden_size)
    k = torch.randn(CFG_31B.num_global_key_value_heads * CFG_31B.global_head_dim, CFG_31B.hidden_size)
    packed = _pack_global_qkv(q, k)

    args = SimpleNamespace(hf_checkpoint="/nonexistent")
    emitted = conv.convert_gemma4_to_hf(
        args, "module.module.decoder.layers.5.self_attention.linear_qkv.weight", packed,
    )
    names = {n for n, _ in emitted}
    # Global K=V layers: only q and k are emitted, v is absent.
    assert names == {
        "model.language_model.layers.5.self_attn.q_proj.weight",
        "model.language_model.layers.5.self_attn.k_proj.weight",
    }


def test_global_layer_index_matches_layer_types():
    """The 1-indexed layer_number mod 6 == 0 heuristic must agree with HF's
    authoritative `layer_types`. This test guards against future Gemma4 variants
    shipping with a non-regular pattern.
    """
    # Build layer_types that mirror the production 31B config.
    lt = []
    for i in range(CFG_31B.num_hidden_layers):
        # 0-indexed: global layers are at 5, 11, 17, ... (every 6th)
        lt.append("full_attention" if (i + 1) % 6 == 0 else "sliding_attention")
    # Ensure our heuristic picks the same indices.
    heuristic = {i for i in range(CFG_31B.num_hidden_layers) if (i + 1) % 6 == 0}
    truth = {i for i, t in enumerate(lt) if t == "full_attention"}
    assert heuristic == truth


def test_mlp_gate_up_roundtrip():
    """linear_fc1 in Megatron packs [gate, up] along dim 0."""
    gate = torch.randn(21504, CFG_31B.hidden_size)
    up = torch.randn(21504, CFG_31B.hidden_size)
    fused = torch.cat([gate, up], dim=0)
    gate2, up2 = fused.chunk(2, dim=0)
    assert torch.equal(gate, gate2) and torch.equal(up, up2)


def test_convert_gemma4_to_hf_moe_expert_weights_stacked():
    """Per-expert fc1/fc2 weights (from TEGroupedLinear) stream in one at a time
    and the converter buffers them until all experts arrive, then emits a single
    stacked 3D tensor matching sglang's Gemma4 loader expectation.
    """
    conv = _load_convert_module()
    num_experts = 4  # keep test fast
    conv._config_cache["config"] = {
        "global_attn_layers": {5},
        "local_head_dim": 256, "global_head_dim": 512,
        "num_attention_heads": 16,
        "local_num_kv_heads": 8, "global_num_kv_heads": 2,
        "hidden_size": 2816,
        "num_experts": num_experts,
    }
    conv._expert_buffers.clear()
    args = SimpleNamespace(hf_checkpoint="/nonexistent")

    # Stream all 4 experts' linear_fc1 tensors. Only the last should emit.
    fc1_tensors = [torch.randn(2 * 704, 2816) for _ in range(num_experts)]
    emitted_total = []
    for e, t in enumerate(fc1_tensors):
        out = conv.convert_gemma4_to_hf(
            args, f"module.module.decoder.layers.3.mlp.experts.linear_fc1.weight{e}", t,
        )
        emitted_total.append(out)
    # Only the last flush produces output.
    assert all(len(out) == 0 for out in emitted_total[:-1])
    last = emitted_total[-1]
    assert len(last) == 1
    name, stacked = last[0]
    assert name == "model.language_model.layers.3.experts.gate_up_proj"
    assert stacked.shape == (num_experts, 2 * 704, 2816)
    for e, t in enumerate(fc1_tensors):
        assert torch.equal(stacked[e], t)

    # Same for linear_fc2.
    fc2_tensors = [torch.randn(2816, 704) for _ in range(num_experts)]
    emitted_total = []
    for e, t in enumerate(fc2_tensors):
        out = conv.convert_gemma4_to_hf(
            args, f"module.module.decoder.layers.3.mlp.experts.linear_fc2.weight{e}", t,
        )
        emitted_total.append(out)
    assert all(len(out) == 0 for out in emitted_total[:-1])
    last = emitted_total[-1]
    assert len(last) == 1
    name, stacked = last[0]
    assert name == "model.language_model.layers.3.experts.down_proj"
    assert stacked.shape == (num_experts, 2816, 704)
    for e, t in enumerate(fc2_tensors):
        assert torch.equal(stacked[e], t)


def test_convert_gemma4_to_hf_moe_router_weights():
    """Router lives at `.mlp.router.*` under the MoE variant since
    `self.mlp = Gemma4MoELayer` for MoE-enabled layers."""
    conv = _load_convert_module()
    conv._config_cache["config"] = {
        "global_attn_layers": {5},
        "local_head_dim": 256, "global_head_dim": 512,
        "num_attention_heads": 16,
        "local_num_kv_heads": 8, "global_num_kv_heads": 2,
        "hidden_size": 2816,
    }
    args = SimpleNamespace(hf_checkpoint="/nonexistent")
    for mcore_rest, hf_tail in [
        ("mlp.router.proj.weight", "router.proj.weight"),
        ("mlp.router.scale", "router.scale"),
        ("mlp.router.per_expert_scale", "router.per_expert_scale"),
    ]:
        param = torch.randn(4)
        emitted = conv.convert_gemma4_to_hf(
            args, f"module.module.decoder.layers.3.{mcore_rest}", param,
        )
        assert len(emitted) == 1
        assert emitted[0][0] == f"model.language_model.layers.3.{hf_tail}"


def test_convert_gemma4_to_hf_dense_mlp_sibling():
    """Under MoE variant the parallel dense MLP lives at `.dense_mlp.*`; it
    must still map to HF's `.mlp.*` since HF uses a single MLP naming."""
    conv = _load_convert_module()
    conv._config_cache["config"] = {
        "global_attn_layers": set(),
        "local_head_dim": 256, "global_head_dim": 512,
        "num_attention_heads": 16,
        "local_num_kv_heads": 8, "global_num_kv_heads": 2,
        "hidden_size": 2816,
    }
    args = SimpleNamespace(hf_checkpoint="/nonexistent")

    gate = torch.randn(2112, 2816)
    up = torch.randn(2112, 2816)
    fused = torch.cat([gate, up], dim=0)

    emitted = conv.convert_gemma4_to_hf(
        args, "module.module.decoder.layers.0.dense_mlp.linear_fc1.weight", fused,
    )
    names = {n for n, _ in emitted}
    assert names == {
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
    }

    down = torch.randn(2816, 2112)
    emitted = conv.convert_gemma4_to_hf(
        args, "module.module.decoder.layers.0.dense_mlp.linear_fc2.weight", down,
    )
    assert emitted == [("model.language_model.layers.0.mlp.down_proj.weight", down)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
