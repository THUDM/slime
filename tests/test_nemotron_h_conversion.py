"""CPU tests for the Nemotron-H (text) HF <-> Megatron weight mapping.

Every test builds a tiny synthetic config, so no checkpoint, GPU or Megatron
install is needed. The properties pinned here are the ones that fail silently in
a real run: a wrong layout has the right shape, so nothing raises -- the engine
just serves different weights than the trainer holds.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

# These mapping tests also run in the CPU CI image, which does not install
# Megatron. In that environment, mount the source package without executing
# megatron_utils' runtime patch initialization.
try:
    _has_megatron = importlib.util.find_spec("megatron.core") is not None
except ModuleNotFoundError:
    _has_megatron = False
if not _has_megatron:
    _megatron_utils = types.ModuleType("slime.backends.megatron_utils")
    _megatron_utils.__path__ = [str(Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils")]
    sys.modules.setdefault("slime.backends.megatron_utils", _megatron_utils)

from slime.backends.megatron_utils.hf_to_megatron import nemotron_h as hf_to_megatron_nemotron_h
from slime.backends.megatron_utils.hf_to_megatron.nemotron_h import (
    _interleave_tp,
    mtp_pattern,
    nemotron_h_hf_tensor,
)
from slime.backends.megatron_utils.megatron_to_hf.nemotron_h import _deinterleave_tp, convert_nemotron_h_to_hf

NUM_GPUS = 0

# Tiny dimensions that still exercise every rule:
#   d_inner = expand * hidden = 16, groups_state = n_groups * ssm_state_size = 4,
#   nheads = 4, so in_proj packs [z 16 | x 16 | B 4 | C 4 | dt 4] = 44 rows and
#   conv1d packs [x 16 | B 4 | C 4] = 24 rows. Every component divides by TP2.
D_INNER, GROUPS_STATE, NHEADS = 16, 4, 4
IN_PROJ_ROWS = 2 * D_INNER + 2 * GROUPS_STATE + NHEADS
CONV_ROWS = D_INNER + 2 * GROUPS_STATE
HIDDEN = 8

# Main stack M, *, E; one MTP depth of attention + MoE.
HF_CONFIG = types.SimpleNamespace(
    hybrid_override_pattern="M*E",
    mtp_hybrid_override_pattern="*E",
    expand=2,
    hidden_size=HIDDEN,
    n_groups=2,
    ssm_state_size=2,
    mamba_num_heads=NHEADS,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=2,
    tie_word_embeddings=False,
)


def _export_args(tp=1, pattern="M*E/*E"):
    return types.SimpleNamespace(
        tensor_model_parallel_size=tp,
        hybrid_override_pattern=pattern,
        mamba_num_heads=NHEADS,
        mamba_head_dim=D_INNER // NHEADS,
        mamba_num_groups=2,
        mamba_state_dim=2,
        num_query_groups=2,
        num_attention_heads=4,
        kv_channels=2,
    )


class Reader:
    def __init__(self, **tensors):
        self.tensors = tensors

    def __contains__(self, name):
        return name in self.tensors

    def get_tensor(self, name):
        return self.tensors[name]


def _rows(n, cols=3, offset=0):
    return (torch.arange(n * cols, dtype=torch.float32) + offset).reshape(n, cols)


def _load(name, tensors, config=HF_CONFIG, tp=1, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(hf_to_megatron_nemotron_h, "_tp_size", lambda: tp)
    return nemotron_h_hf_tensor(f"module.module.{name}", Reader(**tensors), config)


def _export(name, param, tp=1, pattern="M*E/*E"):
    return convert_nemotron_h_to_hf(_export_args(tp, pattern), f"module.module.{name}", param)


# ---------------------------------------------------------------- Mamba layout


@pytest.mark.unit
def test_interleave_is_identity_at_tp1():
    components = [_rows(4), _rows(2), _rows(2)]
    assert torch.equal(_interleave_tp(components, 1), torch.cat(components, dim=0))


@pytest.mark.unit
def test_interleave_gives_every_tp_rank_a_slice_of_every_component():
    z, x, dt = _rows(8, offset=0), _rows(8, offset=100), _rows(4, offset=200)
    interleaved = _interleave_tp([z, x, dt], 2)

    # A plain contiguous chunk -- which is all slime's loader does -- must hand
    # rank r its half of z, its half of x and its half of dt.
    for rank, shard in enumerate(interleaved.chunk(2, dim=0)):
        expected = torch.cat([z.chunk(2)[rank], x.chunk(2)[rank], dt.chunk(2)[rank]])
        assert torch.equal(shard, expected)

    # A flat cat would give rank 0 all of z and nothing of dt.
    assert not torch.equal(interleaved, torch.cat([z, x, dt]))


@pytest.mark.unit
def test_interleave_rejects_a_component_that_does_not_divide():
    with pytest.raises(ValueError, match="does not divide across TP2"):
        _interleave_tp([_rows(3)], 2)


@pytest.mark.unit
def test_deinterleave_rejects_a_wrong_row_count():
    with pytest.raises(ValueError, match="expected 8 rows"):
        _deinterleave_tp(_rows(6), [4, 4], 2)


@pytest.mark.unit
@pytest.mark.parametrize("tp", [1, 2])
def test_mamba_in_proj_round_trips_through_the_tp_layout(tp, monkeypatch):
    hf = _rows(IN_PROJ_ROWS)
    loaded = _load(
        "decoder.layers.0.mixer.in_proj.weight",
        {"backbone.layers.0.mixer.in_proj.weight": hf},
        tp=tp,
        monkeypatch=monkeypatch,
    )

    # What the weight sync receives: the per-rank shards, concatenated again.
    gathered = torch.cat(loaded.chunk(tp, dim=0), dim=0)
    ((hf_name, restored),) = _export("decoder.layers.0.mixer.in_proj.weight", gathered, tp=tp)

    assert hf_name == "backbone.layers.0.mixer.in_proj.weight"
    assert torch.equal(restored, hf)


@pytest.mark.unit
def test_mamba_in_proj_tp2_shard_is_the_per_component_half(monkeypatch):
    hf = _rows(IN_PROJ_ROWS)
    loaded = _load(
        "decoder.layers.0.mixer.in_proj.weight",
        {"backbone.layers.0.mixer.in_proj.weight": hf},
        tp=2,
        monkeypatch=monkeypatch,
    )
    sizes = [D_INNER, D_INNER, GROUPS_STATE, GROUPS_STATE, NHEADS]
    components = hf.split(sizes, dim=0)
    rank1 = torch.cat([c.chunk(2, dim=0)[1] for c in components])
    assert torch.equal(loaded.chunk(2, dim=0)[1], rank1)


@pytest.mark.unit
@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("suffix", ["weight", "bias"])
def test_conv1d_is_renamed_to_the_flat_mcore_parameter_and_round_trips(tp, suffix, monkeypatch):
    hf = _rows(CONV_ROWS, cols=1) if suffix == "bias" else torch.randn(CONV_ROWS, 1, 4)
    loaded = _load(
        f"decoder.layers.0.mixer.conv1d_{suffix}",
        {f"backbone.layers.0.mixer.conv1d.{suffix}": hf},
        tp=tp,
        monkeypatch=monkeypatch,
    )
    gathered = torch.cat(loaded.chunk(tp, dim=0), dim=0)
    ((hf_name, restored),) = _export(f"decoder.layers.0.mixer.conv1d_{suffix}", gathered, tp=tp)

    assert hf_name == f"backbone.layers.0.mixer.conv1d.{suffix}"
    assert torch.equal(restored, hf)


@pytest.mark.unit
@pytest.mark.parametrize("rest", ["A_log", "D", "dt_bias", "norm.weight", "out_proj.weight"])
def test_per_head_mamba_parameters_pass_through(rest):
    tensor = _rows(NHEADS)
    loaded = _load(f"decoder.layers.0.mixer.{rest}", {f"backbone.layers.0.mixer.{rest}": tensor})
    assert loaded is tensor
    assert _export(f"decoder.layers.0.mixer.{rest}", tensor) == [(f"backbone.layers.0.mixer.{rest}", tensor)]


# ------------------------------------------------------------------- attention


@pytest.mark.unit
def test_qkv_is_grouped_per_kv_head_and_round_trips():
    # 4 heads over 2 KV groups, head_dim 2: q is 8 rows, k and v are 4 rows each.
    q, k, v = _rows(8, offset=0), _rows(4, offset=100), _rows(4, offset=200)
    loaded = _load(
        "decoder.layers.1.self_attention.linear_qkv.weight",
        {
            "backbone.layers.1.mixer.q_proj.weight": q,
            "backbone.layers.1.mixer.k_proj.weight": k,
            "backbone.layers.1.mixer.v_proj.weight": v,
        },
    )

    # Per KV group: that group's two query heads (4 rows), then its k, then its v.
    assert torch.equal(loaded[0:4], q[0:4])
    assert torch.equal(loaded[4:6], k[0:2])
    assert torch.equal(loaded[6:8], v[0:2])
    assert torch.equal(loaded[8:12], q[4:8])
    assert torch.equal(loaded[12:14], k[2:4])
    assert torch.equal(loaded[14:16], v[2:4])

    exported = dict(_export("decoder.layers.1.self_attention.linear_qkv.weight", loaded))
    assert torch.equal(exported["backbone.layers.1.mixer.q_proj.weight"], q)
    assert torch.equal(exported["backbone.layers.1.mixer.k_proj.weight"], k)
    assert torch.equal(exported["backbone.layers.1.mixer.v_proj.weight"], v)


@pytest.mark.unit
def test_attention_output_projection_maps_to_o_proj():
    tensor = _rows(HIDDEN)
    loaded = _load(
        "decoder.layers.1.self_attention.linear_proj.weight", {"backbone.layers.1.mixer.o_proj.weight": tensor}
    )
    assert loaded is tensor
    assert _export("decoder.layers.1.self_attention.linear_proj.weight", tensor) == [
        ("backbone.layers.1.mixer.o_proj.weight", tensor)
    ]


# ------------------------------------------------------------------- pre-norms


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layer", "rest"),
    [
        (0, "mixer.in_proj.layer_norm_weight"),  # M: folded into in_proj
        (1, "self_attention.linear_qkv.layer_norm_weight"),  # *: folded into linear_qkv
        (2, "pre_mlp_layernorm.weight"),  # E: kept separate
    ],
)
def test_every_layer_type_reaches_the_same_hf_norm(layer, rest):
    norm = torch.randn(HIDDEN)
    name = f"decoder.layers.{layer}.{rest}"
    loaded = _load(name, {f"backbone.layers.{layer}.norm.weight": norm})
    assert loaded is norm
    assert _export(name, norm) == [(f"backbone.layers.{layer}.norm.weight", norm)]


# -------------------------------------------------------------------- MoE layer


@pytest.mark.unit
@pytest.mark.parametrize(
    ("megatron_rest", "hf_rest"),
    [
        ("mlp.router.weight", "mixer.gate.weight"),
        ("mlp.router.expert_bias", "mixer.gate.e_score_correction_bias"),
        ("mlp.fc1_latent_proj.weight", "mixer.fc1_latent_proj.weight"),
        ("mlp.fc2_latent_proj.weight", "mixer.fc2_latent_proj.weight"),
        ("mlp.shared_experts.linear_fc1.weight", "mixer.shared_experts.up_proj.weight"),
        ("mlp.shared_experts.linear_fc2.weight", "mixer.shared_experts.down_proj.weight"),
        # Non-gated experts: linear_fc1 is up_proj alone. There is no gate_proj to
        # concatenate, which is what distinguishes this family from Qwen.
        ("mlp.experts.linear_fc1.weight7", "mixer.experts.7.up_proj.weight"),
        ("mlp.experts.linear_fc2.weight7", "mixer.experts.7.down_proj.weight"),
    ],
)
def test_moe_layer_names_round_trip(megatron_rest, hf_rest):
    tensor = _rows(4)
    name = f"decoder.layers.2.{megatron_rest}"
    loaded = _load(name, {f"backbone.layers.2.{hf_rest}": tensor})
    assert loaded is tensor
    assert _export(name, tensor) == [(f"backbone.layers.2.{hf_rest}", tensor)]


@pytest.mark.unit
def test_whole_expert_stack_is_rejected_in_both_directions():
    name = "decoder.layers.2.mlp.experts.linear_fc1.weight"
    with pytest.raises(KeyError, match="whole expert stack"):
        _load(name, {})
    with pytest.raises(ValueError, match="whole expert stack"):
        _export(name, _rows(2))


# ---------------------------------------------------------------- top level


@pytest.mark.unit
def test_embedding_norm_and_head_map_to_backbone_names():
    for megatron, hf in [
        ("embedding.word_embeddings.weight", "backbone.embeddings.weight"),
        ("decoder.final_norm.weight", "backbone.norm_f.weight"),
        ("decoder.final_layernorm.weight", "backbone.norm_f.weight"),
        ("output_layer.weight", "lm_head.weight"),
    ]:
        tensor = _rows(2)
        assert _load(megatron, {hf: tensor}) is tensor
        assert _export(megatron, tensor) == [(hf, tensor)]


@pytest.mark.unit
def test_tied_embeddings_read_the_embedding_for_the_head():
    tied = types.SimpleNamespace(**{**vars(HF_CONFIG), "tie_word_embeddings": True})
    tensor = _rows(2)
    assert _load("output_layer.weight", {"backbone.embeddings.weight": tensor}, config=tied) is tensor


@pytest.mark.unit
def test_layer_past_the_end_of_the_pattern_is_an_error():
    with pytest.raises(KeyError, match="past the end"):
        _load("decoder.layers.3.mixer.in_proj.weight", {})


@pytest.mark.unit
def test_unknown_parameters_are_rejected():
    with pytest.raises(KeyError, match="Unsupported"):
        _load("decoder.layers.0.mixer.surprise", {})
    with pytest.raises(ValueError, match="Unsupported"):
        _export("decoder.layers.0.mixer.surprise", _rows(2))
    with pytest.raises(KeyError, match="Unsupported"):
        _load("not_a_decoder.weight", {})


# ------------------------------------------------------------------------ MTP


@pytest.mark.unit
def test_mtp_pattern_accepts_both_config_spellings():
    assert mtp_pattern(types.SimpleNamespace(mtp_hybrid_override_pattern="*E")) == "*E"
    assert mtp_pattern(types.SimpleNamespace(mtp_layers_block_type=["attention", "moe"])) == "*E"
    with pytest.raises(KeyError, match="neither"):
        mtp_pattern(types.SimpleNamespace())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("megatron_rest", "hf_name"),
    [
        # Depth-level pieces live on the first / last layer of the checkpoint's depth.
        ("enorm.weight", "mtp.layers.0.enorm.weight"),
        ("hnorm.weight", "mtp.layers.0.hnorm.weight"),
        ("eh_proj.weight", "mtp.layers.0.eh_proj.weight"),
        ("final_layernorm.weight", "mtp.layers.1.final_layernorm.weight"),
        # Inner layers follow the MTP pattern: '*' at 0, 'E' at 1.
        ("mtp_model_layer.layers.0.self_attention.linear_proj.weight", "mtp.layers.0.mixer.o_proj.weight"),
        ("mtp_model_layer.layers.1.mlp.router.weight", "mtp.layers.1.mixer.gate.weight"),
        ("mtp_model_layer.layers.1.mlp.experts.linear_fc1.weight3", "mtp.layers.1.mixer.experts.3.up_proj.weight"),
        ("mtp_model_layer.layers.1.pre_mlp_layernorm.weight", "mtp.layers.1.norm.weight"),
    ],
)
def test_mtp_parameters_round_trip(megatron_rest, hf_name):
    tensor = _rows(4)
    name = f"mtp.layers.0.{megatron_rest}"
    assert _load(name, {hf_name: tensor}) is tensor
    assert _export(name, tensor) == [(hf_name, tensor)]


@pytest.mark.unit
def test_mtp_depth_beyond_the_checkpoints_single_depth_is_rejected():
    with pytest.raises(ValueError, match="no checkpoint counterpart"):
        _export("mtp.layers.1.enorm.weight", _rows(2))


@pytest.mark.unit
def test_mtp_export_needs_an_mtp_segment_in_the_pattern():
    with pytest.raises(ValueError, match="no MTP segment"):
        _export("mtp.layers.0.enorm.weight", _rows(2), pattern="M*E")


@pytest.mark.unit
def test_mtp_inner_layer_past_the_pattern_is_rejected():
    with pytest.raises(KeyError, match="past the end"):
        _load("mtp.layers.0.mtp_model_layer.layers.5.mlp.router.weight", {})


@pytest.mark.unit
def test_main_stack_ignores_the_mtp_segment_of_the_pattern():
    # MCore's spelling appends the MTP depths after '/'; layer indices address
    # the main stack only.
    config = types.SimpleNamespace(**{**vars(HF_CONFIG), "hybrid_override_pattern": "M*E/*E"})
    tensor = _rows(HIDDEN)
    assert (
        _load("decoder.layers.2.pre_mlp_layernorm.weight", {"backbone.layers.2.norm.weight": tensor}, config=config)
        is tensor
    )
