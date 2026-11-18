from types import SimpleNamespace

import pytest

from slime.utils.flops_utils import calculate_fwd_flops

NUM_GPUS = 0


def _dense_args(**overrides):
    values = dict(
        hidden_size=5120,
        num_attention_heads=64,
        num_query_groups=8,
        kv_channels=128,
        ffn_hidden_size=25600,
        num_layers=1,
        vocab_size=151936,
        num_experts=None,
        multi_latent_attention=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _layer_flops(args, seqlen):
    # calculate_fwd_flops for one layer, without the LM head (the same model with no layers).
    return calculate_fwd_flops([seqlen], args) - calculate_fwd_flops(
        [seqlen], SimpleNamespace(**{**vars(args), "num_layers": 0})
    )


def _expected_layer_flops(args, seqlen, out_width):
    h, s = args.hidden_size, seqlen
    qkv = (
        2 * s * h * args.num_attention_heads * args.kv_channels
        + 2 * 2 * s * h * args.num_query_groups * args.kv_channels
    )
    attn = (
        2 * args.num_attention_heads * s * s * args.kv_channels / 2
        + args.num_attention_heads * s * s * args.kv_channels
    )
    out = 2 * s * out_width * h
    mlp = 2 * s * h * args.ffn_hidden_size * 3
    return qkv + attn + out + mlp


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {},  # Qwen3-32B: 64 heads x 128 = 8192, wider than hidden 5120
        dict(hidden_size=4096, num_attention_heads=32, num_query_groups=8, ffn_hidden_size=14336),  # Llama-3-8B: equal
    ],
)
def test_output_projection_uses_heads_times_head_dim(overrides):
    args = _dense_args(**overrides)
    seqlen = 1024
    out_width = args.num_attention_heads * args.kv_channels
    assert _layer_flops(args, seqlen) == pytest.approx(_expected_layer_flops(args, seqlen, out_width))


@pytest.mark.unit
def test_mla_output_projection_uses_v_head_dim():
    # DeepSeek-V3 attention: 128 heads x v_head_dim 128 = 16384 into hidden 7168.
    mla = dict(
        multi_latent_attention=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
    )
    args = _dense_args(hidden_size=7168, num_attention_heads=128, num_query_groups=128, ffn_hidden_size=18432, **mla)
    narrow = SimpleNamespace(**{**vars(args), "v_head_dim": 64})
    seqlen = 1024
    # Halving v_head_dim removes half of the output projection, of A*V and of the V up-projection.
    removed = _layer_flops(args, seqlen) - _layer_flops(narrow, seqlen)
    out_projection = 2 * seqlen * (128 * 64) * 7168
    a_times_v = 128 * seqlen * seqlen * 64
    kv_up_projection = 2 * seqlen * 512 * 128 * 64
    assert removed == pytest.approx(out_projection + a_times_v + kv_up_projection)
