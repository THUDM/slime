"""CUDA-gated integration tests for ``Gemma4TransformerLayer``.

Builds a real layer through ``get_gemma4_layer_spec_te``, runs a small
forward, and checks output shape, finiteness, and that ``is_sliding``
matches the layer index. Numerical HF parity is covered outside slime.
"""

import os

import pytest
import torch


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Gemma4TransformerLayer requires CUDA + TE kernels",
)


def _init_single_rank_dist():
    """Initialize a world_size=1 process group (NCCL if CUDA else gloo) so
    Megatron's parallel_state helpers work without a real multi-GPU job.

    Megatron's ``parallel_state`` is module-global: if a prior test module
    left it initialized (e.g. test_gemma4_cp_attention) its
    ``initialize_model_parallel`` asserts on re-init. Clear it first.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    if mpu.model_parallel_is_initialized():
        mpu.destroy_model_parallel()
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29566")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=0, world_size=1)
    mpu.initialize_model_parallel()


@pytest.fixture(scope="module", autouse=True)
def _dist():
    _init_single_rank_dist()
    yield
    # Leave the PG initialized for the module; tearing down between modules
    # interacts badly with shared Megatron state.


def _build_layer_config(
    num_layers=6, hidden_size=128, ffn_hidden_size=256, num_heads=8,
    num_kv_heads=4, head_dim=128, global_head_dim=256, num_global_kv_heads=2,
    sliding_window=64,
):
    """Build a minimal Gemma4TransformerConfig suitable for a smoke test."""
    from slime_plugins.models.gemma4 import Gemma4TransformerConfig

    cfg = Gemma4TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_heads,
        num_query_groups=num_kv_heads,
        kv_channels=head_dim,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        pipeline_dtype=torch.bfloat16,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        add_qkv_bias=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.gelu,  # placeholder
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_softmax_in_fp32=True,
        persist_layer_norm=True,
        bias_activation_fusion=False,
        bias_dropout_fusion=True,
        apply_rope_fusion=False,
        qk_layernorm=True,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
    )
    # Gemma4-specific extensions.
    cfg.global_kv_channels = global_head_dim
    cfg.global_num_query_groups = num_global_kv_heads
    cfg.global_partial_rotary_factor = 0.25
    cfg.attention_k_eq_v = True
    cfg.final_logit_softcapping = 30.0
    cfg.enable_moe_block = False
    cfg.sliding_window = sliding_window
    cfg.sliding_window_pattern = 6  # every 6th layer is global
    cfg.softmax_scale = 1.0
    return cfg


@requires_cuda
def test_layer_builds_and_forwards_sliding():
    """Layer 1 (1-indexed mod 6 != 0) is a sliding layer. Build it, run
    forward, verify shape + finiteness."""
    from functools import partial
    import torch.nn.functional as F
    from megatron.core.transformer.spec_utils import build_module
    from slime_plugins.models.gemma4 import get_gemma4_layer_spec_te

    cfg = _build_layer_config()
    cfg.activation_func = partial(F.gelu, approximate="tanh")
    spec = get_gemma4_layer_spec_te(cfg)

    layer = build_module(spec, config=cfg, layer_number=1)
    layer = layer.cuda().to(torch.bfloat16)
    assert layer.is_sliding is True
    assert layer._is_global is False

    # Input: [seq, batch, hidden]
    seq, batch = 16, 1
    h = torch.randn(seq, batch, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

    # Minimal rotary_pos_emb — just provide a local-dim tensor.
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    rope = RotaryEmbedding(kv_channels=cfg.kv_channels, rotary_percent=1.0)
    rotary = rope(seq).cuda()

    out, _ctx = layer(h, rotary_pos_emb=rotary, attention_mask=None)
    assert out.shape == h.shape
    assert torch.isfinite(out).all()


@requires_cuda
def test_layer_global_path_builds_and_forwards():
    """Layer 6 (1-indexed mod 6 == 0) is a global layer with head_dim=256,
    num_kv_heads=2. Build it, run forward, verify shape + finiteness."""
    from functools import partial
    import torch.nn.functional as F
    from megatron.core.transformer.spec_utils import build_module
    from slime_plugins.models.gemma4 import get_gemma4_layer_spec_te

    cfg = _build_layer_config()
    cfg.activation_func = partial(F.gelu, approximate="tanh")
    spec = get_gemma4_layer_spec_te(cfg)

    # layer_number=6 → 1-indexed → global (mod 6 == 0)
    layer = build_module(spec, config=cfg, layer_number=6)
    layer = layer.cuda().to(torch.bfloat16)
    assert layer.is_sliding is False
    assert layer._is_global is True

    seq, batch = 16, 1
    h = torch.randn(seq, batch, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

    # Global layers use global_head_dim rotary.
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    rope = RotaryEmbedding(kv_channels=cfg.global_kv_channels, rotary_percent=1.0)
    rotary = rope(seq).cuda()

    out, _ctx = layer(h, rotary_pos_emb=rotary, attention_mask=None)
    assert out.shape == h.shape
    assert torch.isfinite(out).all()


@requires_cuda
def test_layer_does_not_mutate_shared_config():
    """Building a global layer must NOT leave the shared `config` with
    global-layer kv_channels/num_query_groups — that would break a
    subsequently-built sliding layer from the same spec."""
    from functools import partial
    import torch.nn.functional as F
    from megatron.core.transformer.spec_utils import build_module
    from slime_plugins.models.gemma4 import get_gemma4_layer_spec_te

    cfg = _build_layer_config()
    cfg.activation_func = partial(F.gelu, approximate="tanh")
    orig_kv = cfg.kv_channels
    orig_nqg = cfg.num_query_groups

    spec = get_gemma4_layer_spec_te(cfg)
    build_module(spec, config=cfg, layer_number=6).cuda()
    # Now shared config must still have the sliding-layer values.
    assert cfg.kv_channels == orig_kv, (
        f"building a global layer mutated shared config.kv_channels: "
        f"{orig_kv} -> {cfg.kv_channels}"
    )
    assert cfg.num_query_groups == orig_nqg, (
        f"building a global layer mutated shared config.num_query_groups: "
        f"{orig_nqg} -> {cfg.num_query_groups}"
    )


def test_layer_spec_builds_without_cuda():
    """Constructing the layer spec (no instantiation) must work on CPU —
    catches regressions like a typo'd submodule or a stray hard CUDA dep
    inside the spec factory itself."""
    from functools import partial
    import torch.nn.functional as F
    from slime_plugins.models.gemma4 import (
        Gemma4SelfAttention, Gemma4TransformerLayer, get_gemma4_layer_spec_te,
    )

    cfg = _build_layer_config()
    cfg.activation_func = partial(F.gelu, approximate="tanh")
    spec = get_gemma4_layer_spec_te(cfg)

    assert spec.module is Gemma4TransformerLayer
    assert spec.submodules.self_attention.module is Gemma4SelfAttention
    # post_attention_layernorm + post_feedforward_layernorm must be real
    # LayerNorms (not IdentityOp), since Gemma uses them.
    from megatron.core.transformer.identity_op import IdentityOp
    assert spec.submodules.post_attention_layernorm is not IdentityOp
    assert spec.submodules.post_feedforward_layernorm is not IdentityOp


def test_layer_spec_moe_variant_includes_dense_mlp_spec():
    """With enable_moe_block=True, the spec's `mlp` submodule must be a
    Gemma4MoELayer and `dense_mlp` must be set (not IdentityOp)."""
    from functools import partial
    import torch.nn.functional as F
    from megatron.core.transformer.identity_op import IdentityOp
    from slime_plugins.models.gemma4 import (
        Gemma4MoELayer, get_gemma4_layer_spec_te,
    )

    cfg = _build_layer_config()
    cfg.activation_func = partial(F.gelu, approximate="tanh")
    cfg.enable_moe_block = True
    cfg.num_moe_experts = 8
    cfg.moe_router_topk = 2
    cfg.moe_ffn_hidden_size = 128
    cfg.moe_token_dispatcher_type = "alltoall"
    cfg.moe_grouped_gemm = True
    cfg.moe_aux_loss_coeff = 0.0
    cfg.moe_router_load_balancing_type = "none"
    cfg.moe_router_score_function = "softmax"
    cfg.moe_router_topk_scaling_factor = 1.0
    cfg.moe_router_pre_softmax = False

    spec = get_gemma4_layer_spec_te(cfg)
    assert spec.submodules.mlp.module is Gemma4MoELayer
    assert spec.submodules.dense_mlp is not IdentityOp, (
        "dense_mlp must be a concrete spec when enable_moe_block=True"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
