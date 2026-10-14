"""HuggingFace -> Megatron weight mapping for NVIDIA Nemotron-H (`nemotron_h`).

Nemotron 3 Super is an 88-layer hybrid built from three layer types, chosen per
layer by ``config.hybrid_override_pattern``:

    'M'  Mamba2      40 layers
    'E'  MoE         40 layers   (512 routed experts in a 1024-wide latent space)
    '*'  attention    8 layers   (GQA, 32 heads / 2 KV heads, head_dim 128)

MCore's HybridModel builds exactly those three, in the same order, so layer
index i on the HF side is layer index i on the MCore side. The pattern is what
tells this module which mapping to apply.

HF names everything under a layer ``mixer``, whatever the layer type, and puts
the pre-mixer norm at ``backbone.layers.{i}.norm``. MCore instead folds that
norm into whichever linear comes first (TELayerNormColumnParallelLinear) for
Mamba and attention layers, and keeps it as ``pre_mlp_layernorm`` for MoE
layers -- which is why one HF tensor is reached by three different MCore names
below.

The Mamba tensors are the only ones that are not a rename. See _interleave_tp.
"""

from __future__ import annotations

import re

import torch

from .common import SafetensorReader, strip_mcore_wrappers


def _tp_size() -> int:
    """Tensor-parallel width, or 1 outside an initialised MCore process."""
    try:
        from megatron.core import mpu

        return mpu.get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _interleave_tp(components: list[torch.Tensor], tp_size: int) -> torch.Tensor:
    """Lay components out so a plain TP chunk gives each rank all of them.

    This is the one place the mapping is not a rename, and MCore says why at
    ssm/mamba_mixer.py:300-303:

        in_proj packs [z, x, B, C, dt] into one ColumnParallelLinear.  Each
        component is independently TP-sharded but with different sizes.  When
        resharding across different TP sizes the planner must interleave
        per-component blocks rather than doing a contiguous concat.

    MCore records that as a ``partition_sizes`` attribute on the parameter and
    its own checkpoint planner honours it. slime's loader does not: it shards
    with a generic ``torch.chunk`` over partition_dim
    (hf_to_megatron/common.py:_tensor_parallel_shard). So a flat HF
    ``[z | x | B | C | dt]`` would hand rank 0 all of z and none of dt.

    Building the interleaved layout here instead means the generic chunk is
    correct by construction:

        [z_0, x_0, B_0, C_0, dt_0, z_1, x_1, B_1, C_1, dt_1, ...]

    At tp_size == 1 this is the identity, which is why a TP1 run cannot tell
    the two layouts apart -- and why this needs a TP>1 test.
    """
    if tp_size == 1:
        return torch.cat(components, dim=0).contiguous()

    blocks = []
    for rank in range(tp_size):
        for component in components:
            if component.shape[0] % tp_size != 0:
                raise ValueError(
                    f"Mamba component of size {component.shape[0]} does not divide across TP{tp_size}"
                )
            blocks.append(component.chunk(tp_size, dim=0)[rank])
    return torch.cat(blocks, dim=0).contiguous()


def _mamba_dims(hf_config) -> tuple[int, int, int]:
    """(d_inner, groups_state, nheads) -- the component sizes of in_proj."""
    d_inner = hf_config.expand * hf_config.hidden_size
    groups_state = hf_config.n_groups * hf_config.ssm_state_size
    return d_inner, groups_state, hf_config.mamba_num_heads


def _split_in_proj(tensor: torch.Tensor, hf_config) -> list[torch.Tensor]:
    """HF in_proj is [z | xBC | dt]; return [z, x, B, C, dt].

    Both sides order the components the same way -- HF at
    modeling_nemotron_h.py:400 (`[d_mlp, d_mlp, intermediate_size, conv_dim,
    num_heads]`, with d_mlp 0 for this checkpoint) and MCore at
    ssm/mamba_mixer.py:591 (`z, xBC, dt`) -- so only the TP layout differs.
    """
    d_inner, groups_state, nheads = _mamba_dims(hf_config)
    z, x, b, c, dt = torch.split(tensor, [d_inner, d_inner, groups_state, groups_state, nheads], dim=0)
    return [z, x, b, c, dt]


def _split_conv1d(tensor: torch.Tensor, hf_config) -> list[torch.Tensor]:
    """conv1d covers [x | B | C] -- d_inner + 2 * n_groups * d_state rows."""
    d_inner, groups_state, _ = _mamba_dims(hf_config)
    x, b, c = torch.split(tensor, [d_inner, groups_state, groups_state], dim=0)
    return [x, b, c]


def _merge_qkv(reader: SafetensorReader, prefix: str, hf_config) -> torch.Tensor:
    """HF keeps q/k/v apart; MCore's linear_qkv wants them grouped per KV head."""
    q = reader.get_tensor(f"{prefix}.q_proj.weight")
    k = reader.get_tensor(f"{prefix}.k_proj.weight")
    v = reader.get_tensor(f"{prefix}.v_proj.weight")
    num_groups = hf_config.num_key_value_heads
    queries_per_group = hf_config.num_attention_heads // num_groups
    head_dim = hf_config.head_dim

    trailing = q.shape[1:]
    q = q.reshape(num_groups, queries_per_group * head_dim, *trailing)
    k = k.reshape(num_groups, head_dim, *trailing)
    v = v.reshape(num_groups, head_dim, *trailing)
    return torch.cat((q, k, v), dim=1).reshape(-1, *trailing).contiguous()


_BLOCK_TYPE_SYMBOLS = {"mamba": "M", "moe": "E", "attention": "*"}


def mtp_pattern(hf_config) -> str:
    """The per-depth MTP layer pattern, e.g. '*E' for Nemotron 3 / 3.5 Super.

    Nemotron 3's config spells it ``mtp_hybrid_override_pattern``; 3.5 spells it
    as an ``mtp_layers_block_type`` list. Either way it is ONE depth's worth of
    layers -- the checkpoint stores a single MTP depth (num_nextn_predict_layers
    1) at ``mtp.layers.{k}``, k indexing the layers of that depth.
    """
    pattern = getattr(hf_config, "mtp_hybrid_override_pattern", None)
    if pattern:
        return pattern
    block_types = getattr(hf_config, "mtp_layers_block_type", None)
    if not block_types:
        raise KeyError("Nemotron-H config has neither mtp_hybrid_override_pattern nor mtp_layers_block_type")
    return "".join(_BLOCK_TYPE_SYMBOLS[block] for block in block_types)


def nemotron_h_hf_tensor(name: str, reader: SafetensorReader, hf_config) -> torch.Tensor:
    """Return the full, unsharded MCore tensor for a Nemotron-H parameter name."""

    name = strip_mcore_wrappers(name)

    direct = {
        "embedding.word_embeddings.weight": "backbone.embeddings.weight",
        "decoder.final_norm.weight": "backbone.norm_f.weight",
        # HybridModel inherits GPTModel's name for the post-decoder norm in some
        # configurations; accept both rather than depend on which.
        "decoder.final_layernorm.weight": "backbone.norm_f.weight",
        "output_layer.weight": (
            "backbone.embeddings.weight"
            if getattr(hf_config, "tie_word_embeddings", False)
            else "lm_head.weight"
        ),
    }
    if name in direct:
        return reader.get_tensor(direct[name])

    mtp_match = re.fullmatch(r"mtp\.layers\.(\d+)\.(.+)", name)
    if mtp_match:
        return _mtp_tensor(mtp_match.group(2), reader, hf_config, name)

    layer_match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not layer_match:
        raise KeyError(f"Unsupported Nemotron-H Megatron parameter {name!r}")
    layer_idx, rest = layer_match.groups()
    layer_idx = int(layer_idx)

    pattern = hf_config.hybrid_override_pattern
    # The MTP pattern is appended after a separator in MCore's spelling; the
    # main stack is what layer indices address.
    pattern = pattern.split("/")[0]
    if layer_idx >= len(pattern):
        raise KeyError(f"Layer {layer_idx} is past the end of hybrid_override_pattern ({len(pattern)} layers)")
    return _layer_tensor(pattern[layer_idx], f"backbone.layers.{layer_idx}", rest, reader, hf_config, name)


def _mtp_tensor(rest: str, reader: SafetensorReader, hf_config, name: str) -> torch.Tensor:
    """One MCore MTP parameter, from the checkpoint's single MTP depth.

    MCore's hybrid MTP layer (transformer/multi_token_prediction.py) wraps the
    depth's layers in a nested HybridStack and keeps the depth-level pieces
    outside it; the checkpoint flattens the same thing into ``mtp.layers.{k}``,
    with the depth-level pieces on the first and last layer:

        MCore  mtp.layers.{d}.<rest>                  HF  mtp.layers.{k}.<...>
        ------------------------------------------    -------------------------------
        enorm / hnorm / eh_proj                       0.enorm / 0.hnorm / 0.eh_proj
        final_layernorm                               {P-1}.final_layernorm
        mtp_model_layer.layers.{k}.<layer param>      {k}.<same rules as a decoder layer>

    (P = layers per depth.) eh_proj needs no reordering: MCore concatenates
    [enorm(embedding), hnorm(hidden)] (_concat_embeddings), the same order the
    checkpoint's weight was trained for and SGLang's nemotron_h_mtp.py uses.

    Every depth d reads the checkpoint's one depth. With --mtp-use-repeated-layer
    (the Nemotron recipe: one MTP layer applied once per depth) MCore builds
    d = 0 only; without it, every depth starts from the same weights.
    """
    pattern = mtp_pattern(hf_config)
    if rest in {"enorm.weight", "hnorm.weight", "eh_proj.weight"}:
        return reader.get_tensor(f"mtp.layers.0.{rest}")
    if rest == "final_layernorm.weight":
        return reader.get_tensor(f"mtp.layers.{len(pattern) - 1}.final_layernorm.weight")

    inner = re.fullmatch(r"mtp_model_layer\.layers\.(\d+)\.(.+)", rest)
    if not inner:
        raise KeyError(f"Unsupported Nemotron-H MTP parameter {name!r}")
    k, inner_rest = int(inner.group(1)), inner.group(2)
    if k >= len(pattern):
        raise KeyError(f"MTP inner layer {k} is past the end of the MTP pattern {pattern!r} ({name!r})")
    return _layer_tensor(pattern[k], f"mtp.layers.{k}", inner_rest, reader, hf_config, name)


def _layer_tensor(symbol: str, hf: str, rest: str, reader: SafetensorReader, hf_config, name: str) -> torch.Tensor:
    """One parameter of one M / E / * layer; ``hf`` is that layer's checkpoint prefix."""
    # Every layer type has the same pre-mixer norm on the HF side; MCore reaches
    # it under three different names depending on what the layer is.
    pre_norm_names = {
        "mixer.in_proj.layer_norm_weight",            # M: folded into in_proj
        "self_attention.linear_qkv.layer_norm_weight",  # *: folded into linear_qkv
        "pre_mlp_layernorm.weight",                   # E: kept separate
        "input_layernorm.weight",
    }
    if rest in pre_norm_names:
        return reader.get_tensor(f"{hf}.norm.weight")

    if symbol == "M":
        tp = _tp_size()
        if rest == "mixer.in_proj.weight":
            components = _split_in_proj(reader.get_tensor(f"{hf}.mixer.in_proj.weight"), hf_config)
            return _interleave_tp(components, tp)
        # MCore registers these as FLAT parameters -- conv1d_weight, not a
        # conv1d submodule with a .weight (ssm/mamba_mixer.py:329,338) -- while
        # HF has a real nn.Conv1d. Same shapes either way, [conv_dim, 1, d_conv]
        # and [conv_dim]; only the name differs.
        if rest in {"mixer.conv1d_weight", "mixer.conv1d_bias"}:
            suffix = "weight" if rest.endswith("_weight") else "bias"
            tensor = reader.get_tensor(f"{hf}.mixer.conv1d.{suffix}")
            return _interleave_tp(_split_conv1d(tensor, hf_config), tp)
        # A_log, D and dt_bias are one value per Mamba head, so a plain chunk
        # over heads is already the right split and no interleaving applies.
        if rest in {"mixer.A_log", "mixer.D", "mixer.dt_bias"}:
            return reader.get_tensor(f"{hf}.{rest}")
        if rest in {"mixer.norm.weight", "mixer.out_proj.weight"}:
            return reader.get_tensor(f"{hf}.{rest}")
        raise KeyError(f"Unsupported Nemotron-H Mamba parameter {name!r}")

    if symbol == "*":
        if rest == "self_attention.linear_qkv.weight":
            return _merge_qkv(reader, f"{hf}.mixer", hf_config)
        if rest == "self_attention.linear_proj.weight":
            return reader.get_tensor(f"{hf}.mixer.o_proj.weight")
        raise KeyError(f"Unsupported Nemotron-H attention parameter {name!r}")

    if symbol == "E":
        if rest == "mlp.router.weight":
            return reader.get_tensor(f"{hf}.mixer.gate.weight")
        if rest == "mlp.router.expert_bias":
            return reader.get_tensor(f"{hf}.mixer.gate.e_score_correction_bias")
        if rest in {"mlp.fc1_latent_proj.weight", "mlp.fc2_latent_proj.weight"}:
            suffix = rest.removeprefix("mlp.")
            return reader.get_tensor(f"{hf}.mixer.{suffix}")

        # Routed experts. MCore's grouped GEMM exposes one tensor per expert as
        # linear_fc1.weight{e}; the un-suffixed form is the whole stack.
        # Nemotron's experts are NOT gated -- `mlp_hidden_act: relu2`, and the
        # checkpoint has no gate_proj -- so linear_fc1 is up_proj alone rather
        # than cat(gate, up) the way every gated model in this directory does.
        expert_match = re.fullmatch(r"mlp\.experts\.linear_fc([12])(?:\.weight)?(\d+)?", rest)
        if expert_match:
            projection, expert_idx = expert_match.groups()
            hf_proj = "up_proj" if projection == "1" else "down_proj"
            if expert_idx is None:
                raise KeyError(
                    f"{name!r} addresses the whole expert stack; Nemotron-H stores experts "
                    "individually, so --moe-grouped-gemm must expose per-expert weights"
                )
            return reader.get_tensor(f"{hf}.mixer.experts.{int(expert_idx)}.{hf_proj}.weight")

        shared = {
            "mlp.shared_experts.linear_fc1.weight": "up_proj",
            "mlp.shared_experts.linear_fc2.weight": "down_proj",
        }
        if rest in shared:
            return reader.get_tensor(f"{hf}.mixer.shared_experts.{shared[rest]}.weight")
        raise KeyError(f"Unsupported Nemotron-H MoE parameter {name!r}")

    raise KeyError(f"{hf} has unknown hybrid symbol {symbol!r} for parameter {name!r}")
