import importlib
import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
import torch

from tests.gemma4._standalone_imports import load_gemma4_bridge_class


def _load_convert_module():
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


def test_gemma4_bridge_dense_config_does_not_set_moe_kwargs():
    bridge = object.__new__(load_gemma4_bridge_class())
    bridge.hf_config = CFG_31B
    bridge._build_base_config = lambda **kwargs: kwargs

    cfg = bridge._build_config()

    assert cfg["text_config_key"] is None
    assert "num_moe_experts" not in cfg
    assert "moe_router_topk" not in cfg
    assert "moe_ffn_hidden_size" not in cfg


def test_gemma4_bridge_moe_config_sets_expert_parallel_kwargs():
    bridge = object.__new__(load_gemma4_bridge_class())
    bridge.hf_config = SimpleNamespace(
        text_config=SimpleNamespace(
            enable_moe_block=True,
            num_experts=128,
            top_k_experts=8,
            moe_intermediate_size=704,
            rope_parameters={"sliding_attention": {"rope_theta": 10000.0}},
        )
    )
    bridge._build_base_config = lambda **kwargs: kwargs

    cfg = bridge._build_config()

    assert cfg["text_config_key"] == "text_config"
    assert cfg["num_moe_experts"] == 128
    assert cfg["moe_router_topk"] == 8
    assert cfg["moe_ffn_hidden_size"] == 704
    assert cfg["moe_token_dispatcher_type"] == "alltoall"
    assert cfg["moe_grouped_gemm"] is True
    assert cfg["moe_aux_loss_coeff"] == 0.0
    assert cfg["moe_router_load_balancing_type"] == "none"
    assert cfg["moe_router_score_function"] == "softmax"
    assert cfg["moe_router_pre_softmax"] is False
    assert cfg["moe_router_dtype"] == "fp32"


def _pack_local_qkv(q, k, v):
    num_kv = CFG_31B.num_key_value_heads
    head_dim = CFG_31B.head_dim
    q_per_kv = CFG_31B.num_attention_heads // num_kv
    q = q.view(num_kv, q_per_kv * head_dim, CFG_31B.hidden_size)
    k = k.view(num_kv, head_dim, CFG_31B.hidden_size)
    v = v.view(num_kv, head_dim, CFG_31B.hidden_size)
    return torch.cat([q, k, v], dim=1).reshape(-1, CFG_31B.hidden_size).contiguous()


def _pack_global_qkv(q, k):
    num_kv = CFG_31B.num_global_key_value_heads
    head_dim = CFG_31B.global_head_dim
    q_per_kv = CFG_31B.num_attention_heads // num_kv
    q = q.view(num_kv, q_per_kv * head_dim, CFG_31B.hidden_size)
    k = k.view(num_kv, head_dim, CFG_31B.hidden_size)
    return torch.cat([q, k, k], dim=1).reshape(-1, CFG_31B.hidden_size).contiguous()


def test_convert_gemma4_to_hf_local_layer_roundtrip(monkeypatch):
    conv = _load_convert_module()

    conv._config_cache["/nonexistent"] = {
        "global_attn_layers": {i for i, t in enumerate(CFG_31B.layer_types) if t == "full_attention"},
        "local_head_dim": CFG_31B.head_dim,
        "global_head_dim": CFG_31B.global_head_dim,
        "num_attention_heads": CFG_31B.num_attention_heads,
        "local_num_kv_heads": CFG_31B.num_key_value_heads,
        "global_num_kv_heads": CFG_31B.num_global_key_value_heads,
        "hidden_size": CFG_31B.hidden_size,
    }

    q = torch.randn(CFG_31B.num_attention_heads * CFG_31B.head_dim, CFG_31B.hidden_size)
    k = torch.randn(CFG_31B.num_key_value_heads * CFG_31B.head_dim, CFG_31B.hidden_size)
    v = torch.randn(CFG_31B.num_key_value_heads * CFG_31B.head_dim, CFG_31B.hidden_size)
    packed = _pack_local_qkv(q, k, v)

    args = SimpleNamespace(hf_checkpoint="/nonexistent")
    emitted = conv.convert_gemma4_to_hf(
        args,
        "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
        packed,
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

    conv._config_cache["/nonexistent"] = {
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
        args,
        "module.module.decoder.layers.5.self_attention.linear_qkv.weight",
        packed,
    )
    names = {n for n, _ in emitted}
    assert names == {
        "model.language_model.layers.5.self_attn.q_proj.weight",
        "model.language_model.layers.5.self_attn.k_proj.weight",
    }


def test_convert_config_cache_is_checkpoint_scoped(monkeypatch):
    conv = _load_convert_module()
    conv._config_cache.clear()

    def fake_from_pretrained(path, trust_remote_code):
        hidden_size = 128 if path == "/ckpt-a" else 256
        text_config = SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            head_dim=16,
            global_head_dim=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            hidden_size=hidden_size,
        )
        return SimpleNamespace(text_config=text_config)

    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fake_from_pretrained)

    cfg_a = conv._get_config(SimpleNamespace(hf_checkpoint="/ckpt-a"))
    cfg_b = conv._get_config(SimpleNamespace(hf_checkpoint="/ckpt-b"))

    assert cfg_a["hidden_size"] == 128
    assert cfg_b["hidden_size"] == 256
    assert conv._get_config(SimpleNamespace(hf_checkpoint="/ckpt-a")) is cfg_a


def test_convert_gemma4_to_hf_moe_expert_weights_stacked():
    conv = _load_convert_module()
    num_experts = 4  # keep test fast
    conv._config_cache["/nonexistent"] = {
        "global_attn_layers": {5},
        "local_head_dim": 256,
        "global_head_dim": 512,
        "num_attention_heads": 16,
        "local_num_kv_heads": 8,
        "global_num_kv_heads": 2,
        "hidden_size": 2816,
        "num_experts": num_experts,
    }
    conv._expert_buffers.clear()
    args = SimpleNamespace(hf_checkpoint="/nonexistent")

    fc1_tensors = [torch.randn(2 * 704, 2816) for _ in range(num_experts)]
    emitted_total = []
    for e, t in enumerate(fc1_tensors):
        out = conv.convert_gemma4_to_hf(
            args,
            f"module.module.decoder.layers.3.mlp.experts.linear_fc1.weight{e}",
            t,
        )
        emitted_total.append(out)
    assert all(len(out) == 0 for out in emitted_total[:-1])
    last = emitted_total[-1]
    assert len(last) == 1
    name, stacked = last[0]
    assert name == "model.language_model.layers.3.experts.gate_up_proj"
    assert stacked.shape == (num_experts, 2 * 704, 2816)
    for e, t in enumerate(fc1_tensors):
        assert torch.equal(stacked[e], t)

    fc2_tensors = [torch.randn(2816, 704) for _ in range(num_experts)]
    emitted_total = []
    for e, t in enumerate(fc2_tensors):
        out = conv.convert_gemma4_to_hf(
            args,
            f"module.module.decoder.layers.3.mlp.experts.linear_fc2.weight{e}",
            t,
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
    conv = _load_convert_module()
    conv._config_cache["/nonexistent"] = {
        "global_attn_layers": {5},
        "local_head_dim": 256,
        "global_head_dim": 512,
        "num_attention_heads": 16,
        "local_num_kv_heads": 8,
        "global_num_kv_heads": 2,
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
            args,
            f"module.module.decoder.layers.3.{mcore_rest}",
            param,
        )
        assert len(emitted) == 1
        assert emitted[0][0] == f"model.language_model.layers.3.{hf_tail}"


def test_convert_gemma4_to_hf_dense_mlp_sibling():
    conv = _load_convert_module()
    conv._config_cache["/nonexistent"] = {
        "global_attn_layers": set(),
        "local_head_dim": 256,
        "global_head_dim": 512,
        "num_attention_heads": 16,
        "local_num_kv_heads": 8,
        "global_num_kv_heads": 2,
        "hidden_size": 2816,
    }
    args = SimpleNamespace(hf_checkpoint="/nonexistent")

    gate = torch.randn(2112, 2816)
    up = torch.randn(2112, 2816)
    fused = torch.cat([gate, up], dim=0)

    emitted = conv.convert_gemma4_to_hf(
        args,
        "module.module.decoder.layers.0.dense_mlp.linear_fc1.weight",
        fused,
    )
    names = {n for n, _ in emitted}
    assert names == {
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
    }

    down = torch.randn(2816, 2112)
    emitted = conv.convert_gemma4_to_hf(
        args,
        "module.module.decoder.layers.0.dense_mlp.linear_fc2.weight",
        down,
    )
    assert emitted == [("model.language_model.layers.0.mlp.down_proj.weight", down)]
