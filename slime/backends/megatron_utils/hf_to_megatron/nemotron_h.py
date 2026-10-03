"""HuggingFace ``nemotron_h`` -> Megatron ``MambaModel`` weight mapping.

The loader contract is one function (``common.py::load_model_hf_weights``):
given a *Megatron* parameter name, return the full, unsharded tensor it should
hold. The caller then does vocab padding, tensor-parallel sharding and the shape
check. So this file is a name map plus exactly one real transform (QKV fusion).

The model it maps onto is ``slime_plugins.models.nemotron_h``, which builds
megatron.core's generic hybrid stack -- 88 layers of Mamba-2 / attention /
latent MoE for Nemotron-3 Super, and the same file unchanged for the Nano and
Ultra configs, which differ only in arguments.

WHERE THE MAP COMES FROM
------------------------
Not from reading the two implementations side by side. Megatron-Bridge carries a
verified bidirectional mapping for this architecture
(``src/megatron/bridge/models/nemotronh/nemotron_h_bridge.py::mapping_registry``)
and it is what converted this checkpoint independently. Every entry below is the
Megatron-side name of one of its mappings, restricted to the subset
``MambaModel`` actually creates. Two of its groups are dropped:

* **MTP.** The released checkpoints ship a Multi-Token-Prediction block
  (``mtp.layers.0.*``, 512 more experts). ``MambaModel`` has no MTP, so those
  names are never requested and the tensors are not read. A model loaded through
  here is the backbone without its speculative-decoding head, which matters for
  inference throughput and not for training.
* **Quantisation-spec aliases** (``decoder.layers.*.input_layernorm.weight`` and
  friends). The stock ``mamba_stack_spec`` fuses every input norm into the
  following TE linear, so only the fused names occur. The unfused aliases are
  still handled below, because they cost one dict entry and their absence would
  otherwise surface as a ``KeyError`` on some future spec.

TENSOR PARALLELISM
------------------
This loader refuses TP > 1 and ETP > 1, on purpose. Three separate things in the
generic sharding path are wrong for this model at TP > 1:

1. ``_tensor_parallel_shard`` special-cases any ``linear_fc1.weight`` as a fused
   gate+up tensor and splits it in halves first. Nemotron's experts are
   **squared-ReLU, not gated** -- there is no gate half -- so that split silently
   interleaves garbage. It hits the shared expert at TP and the routed experts at
   ETP.
2. Mamba ``in_proj`` is the concatenation ``[z, x, B, C, dt]``, and Megatron
   expects each TP rank to hold a slice of *every* component, not a contiguous
   block of the whole. A naive chunk gives rank 0 all of ``z`` and nothing else.
3. ``conv1d`` has the same problem over ``[x, B, C]``.

Fixing those three in ``common.py`` -- by discriminating ungated MLPs on
``config.gated_linear_unit`` rather than on the parameter name, and by slicing
the two Mamba tensors per component -- is what would lift this restriction and
let ``--load`` point straight at an HF snapshot at any TP. Until then the
supported route is ``tools/convert_hf_to_torch_dist.py`` at TP1/ETP1/EP1 once,
after which megatron.core's own dist-checkpoint reader reshards to any TP/EP
correctly, because ``MambaMixer`` and the MoE modules implement
``sharded_state_dict``. Converting once is also cheaper: the HF side of
Nemotron-3 Super is 42,683 separate tensors.
"""


from __future__ import annotations

import re

import torch

from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader, merge_qkv, strip_mcore_wrappers

# Megatron name (after wrapper stripping) -> HF name, for the handful of
# parameters that live outside a layer.
_TOP_LEVEL = {
    "embedding.word_embeddings.weight": "backbone.embeddings.weight",
    # MambaStack names its trailing norm `final_norm` (ssm/mamba_block.py:166);
    # `final_layernorm` is the GPTModel spelling and is accepted so a future
    # spec swap does not land here as a KeyError.
    "decoder.final_norm.weight": "backbone.norm_f.weight",
    "decoder.final_layernorm.weight": "backbone.norm_f.weight",
    "output_layer.weight": "lm_head.weight",
}

# Per-layer Megatron suffix -> HF suffix, for everything that is a plain rename.
# `{i}` is filled in by the caller. Ordered by layer type for readability; the
# lookup is a single dict because the layer type is implied by which names the
# model actually asks for.
_PER_LAYER = {
    # -- Mamba-2 mixer ------------------------------------------------------
    "mixer.A_log": "mixer.A_log",
    "mixer.D": "mixer.D",
    "mixer.dt_bias": "mixer.dt_bias",
    # The mixer's internal gated RMSNorm (8192-wide), NOT the layer input norm.
    "mixer.norm.weight": "mixer.norm.weight",
    "mixer.in_proj.weight": "mixer.in_proj.weight",
    "mixer.out_proj.weight": "mixer.out_proj.weight",
    # megatron.core has spelled the convolution both ways across releases; the
    # Bridge registry carries both for the same reason.
    "mixer.conv1d.weight": "mixer.conv1d.weight",
    "mixer.conv1d.bias": "mixer.conv1d.bias",
    "mixer.conv1d_weight": "mixer.conv1d.weight",
    "mixer.conv1d_bias": "mixer.conv1d.bias",
    # -- attention ----------------------------------------------------------
    "self_attention.linear_proj.weight": "mixer.o_proj.weight",
    # -- latent MoE ---------------------------------------------------------
    "mlp.router.weight": "mixer.gate.weight",
    "mlp.router.expert_bias": "mixer.gate.e_score_correction_bias",
    "mlp.fc1_latent_proj.weight": "mixer.fc1_latent_proj.weight",
    "mlp.fc2_latent_proj.weight": "mixer.fc2_latent_proj.weight",
    # Non-gated: linear_fc1 is up_proj alone, with no gate half to concatenate.
    "mlp.shared_experts.linear_fc1.weight": "mixer.shared_experts.up_proj.weight",
    "mlp.shared_experts.linear_fc2.weight": "mixer.shared_experts.down_proj.weight",
    # -- the layer input norm, under each of its four spellings -------------
    # Every layer type in this stack fuses its input norm into the first TE
    # linear, so which of these occurs depends only on the layer type; all four
    # resolve to the same HF tensor.
    "mixer.in_proj.layer_norm_weight": "norm.weight",
    "self_attention.linear_qkv.layer_norm_weight": "norm.weight",
    "mlp.linear_fc1.layer_norm_weight": "norm.weight",
    "pre_mlp_layernorm.weight": "norm.weight",
    "input_layernorm.weight": "norm.weight",
    "norm.weight": "norm.weight",
    # -- dense MLP layers ('-' in the pattern). Nemotron-3 Super has none, but
    # the sibling Nano/Ultra configs do, and these two lines make this file
    # work for them unchanged.
    "mlp.linear_fc1.weight": "mixer.up_proj.weight",
    "mlp.linear_fc2.weight": "mixer.down_proj.weight",
}

# TEGroupedMLP stores the routed experts as `...linear_fc1.weight{global_index}`;
# slime's named_params_and_buffers() has already added the expert-parallel
# offset, so the index here is the global one and indexes HF directly.
_EXPERT_RE = re.compile(r"mlp\.experts\.linear_fc(1|2)\.weight(\d+)")
_EXPERT_HF = {"1": "up_proj", "2": "down_proj"}


def _assert_no_tensor_parallel() -> None:
    """Refuse the layouts the generic sharding path gets wrong for this model."""
    from megatron.core import mpu

    tp = mpu.get_tensor_model_parallel_world_size()
    etp = mpu.get_expert_tensor_parallel_world_size()
    if tp != 1 or etp != 1:
        raise ValueError(
            f"nemotron_h: loading HF weights needs TP=1 and ETP=1 (got TP={tp}, ETP={etp}). "
            "Convert once with tools/convert_hf_to_torch_dist.py and train from the "
            "torch_dist checkpoint instead -- see this module's docstring for the three "
            "transforms that would otherwise be silently wrong."
        )


def nemotron_h_hf_tensor(name: str, reader: SafetensorReader, hf_config) -> torch.Tensor:
    """Return the full, unsharded MCore tensor for a Nemotron-3 parameter name."""
    _assert_no_tensor_parallel()

    name = strip_mcore_wrappers(name)

    if name in _TOP_LEVEL:
        return reader.get_tensor(_TOP_LEVEL[name])

    layer_match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not layer_match:
        raise KeyError(f"Unsupported Nemotron-3 Megatron parameter {name!r}")
    layer_idx, rest = layer_match.groups()
    hf_layer = f"backbone.layers.{layer_idx}"

    expert_match = _EXPERT_RE.fullmatch(rest)
    if expert_match:
        which, expert_idx = expert_match.groups()
        return reader.get_tensor(f"{hf_layer}.mixer.experts.{expert_idx}.{_EXPERT_HF[which]}.weight")

    # The only transform. HF keeps q/k/v apart; Megatron wants one interleaved
    # [q k v] per KV group. merge_qkv reads num_key_value_heads / head_dim off
    # the HF config, which for this checkpoint is 2 groups of 128 -- so the
    # result is [32*128 + 2*128 + 2*128, 4096] = [4608, 4096].
    #
    # Unlike Qwen3.5 there is no RoPE-pair interleave to undo: this model is
    # NoPE, so the plain merge is the right one.
    if rest == "self_attention.linear_qkv.weight":
        return merge_qkv(
            reader.get_tensor(f"{hf_layer}.mixer.q_proj.weight"),
            reader.get_tensor(f"{hf_layer}.mixer.k_proj.weight"),
            reader.get_tensor(f"{hf_layer}.mixer.v_proj.weight"),
            hf_config,
        )

    if rest in _PER_LAYER:
        return reader.get_tensor(f"{hf_layer}.{_PER_LAYER[rest]}")

    raise KeyError(f"Unsupported Nemotron-3 Megatron parameter {name!r} (layer suffix {rest!r})")
