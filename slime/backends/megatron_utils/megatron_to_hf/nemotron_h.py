"""Megatron -> HuggingFace name mapping for NVIDIA Nemotron-H (`nemotron_h`).

The inverse of hf_to_megatron/nemotron_h.py, used by the RL weight sync: GRPO
pushes Megatron's parameters into the colocated SGLang engine on every rollout,
and SGLang wants them under the checkpoint's own names.

Two asymmetries with the loading direction are worth stating, because they are
where this file can be wrong in a way that does not raise:

  * Tensors arrive TP-gathered, and the gather is a plain concat of per-rank
    shards along partition_dim (update_weight/common.py:41-55). For the Mamba
    in_proj and conv1d that means the incoming layout is the *interleaved* one
    hf_to_megatron builds -- [z_0, x_0, B_0, C_0, dt_0, z_1, ...] -- and it has
    to be flattened back to the checkpoint's [z | x | B | C | dt] before
    SGLang sees it. _deinterleave_tp is exactly the inverse of
    hf_to_megatron/nemotron_h.py::_interleave_tp, and like it, it is the
    identity at TP1.

  * One HF tensor is produced by three different Megatron names, because MCore
    folds the pre-mixer norm into whichever linear comes first. Emitting it
    from all three is correct -- they hold the same values -- but it does mean
    the same HF key is written more than once per sync.
"""

from __future__ import annotations

import re

import torch


def _tp_size(args) -> int:
    return max(1, int(getattr(args, "tensor_model_parallel_size", 1) or 1))


def _mamba_dims(args) -> tuple[int, int, int]:
    """(d_inner, groups_state, nheads) -- component sizes inside in_proj."""
    nheads = args.mamba_num_heads
    d_inner = nheads * args.mamba_head_dim
    groups_state = args.mamba_num_groups * args.mamba_state_dim
    return d_inner, groups_state, nheads


def _deinterleave_tp(tensor: torch.Tensor, sizes: list[int], tp_size: int) -> torch.Tensor:
    """Undo the per-rank interleaving, returning components concatenated whole.

    ``sizes`` are the FULL (all-rank) sizes of each component, in order.
    """
    if tp_size == 1:
        return tensor.contiguous()

    total = sum(sizes)
    if tensor.shape[0] != total:
        raise ValueError(f"expected {total} rows to de-interleave, got {tensor.shape[0]}")

    local_sizes = []
    for size in sizes:
        if size % tp_size != 0:
            raise ValueError(f"component of size {size} does not divide across TP{tp_size}")
        local_sizes.append(size // tp_size)

    per_rank = sum(local_sizes)
    parts: list[list[torch.Tensor]] = [[] for _ in sizes]
    for rank in range(tp_size):
        offset = rank * per_rank
        for index, local in enumerate(local_sizes):
            parts[index].append(tensor[offset : offset + local])
            offset += local
    return torch.cat([torch.cat(blocks, dim=0) for blocks in parts], dim=0).contiguous()


def _split_qkv(param: torch.Tensor, args) -> list[tuple[str, torch.Tensor]]:
    """linear_qkv is grouped per KV head; the checkpoint wants q/k/v apart."""
    num_groups = args.num_query_groups
    head_dim = args.kv_channels
    queries_per_group = args.num_attention_heads // num_groups

    trailing = param.shape[1:]
    grouped = param.reshape(num_groups, (queries_per_group + 2) * head_dim, *trailing)
    q = grouped[:, : queries_per_group * head_dim].reshape(-1, *trailing)
    k = grouped[:, queries_per_group * head_dim : (queries_per_group + 1) * head_dim].reshape(-1, *trailing)
    v = grouped[:, (queries_per_group + 1) * head_dim :].reshape(-1, *trailing)
    return [q.contiguous(), k.contiguous(), v.contiguous()]


def convert_nemotron_h_to_hf(args, name, param):
    """Convert one Nemotron-H Megatron parameter to its HuggingFace form."""

    while name.startswith("module."):
        name = name.removeprefix("module.")
    name = name.removeprefix("language_model.")

    direct = {
        "embedding.word_embeddings.weight": "backbone.embeddings.weight",
        "decoder.final_norm.weight": "backbone.norm_f.weight",
        "decoder.final_layernorm.weight": "backbone.norm_f.weight",
        "output_layer.weight": "lm_head.weight",
    }
    if name in direct:
        return [(direct[name], param)]

    mtp_match = re.fullmatch(r"mtp\.layers\.(\d+)\.(.+)", name)
    if mtp_match:
        return _convert_mtp(int(mtp_match.group(1)), mtp_match.group(2), param, args, name)

    layer_match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not layer_match:
        raise ValueError(f"Unsupported Nemotron-H Megatron parameter {name!r}")
    layer_idx, rest = layer_match.groups()
    layer_idx = int(layer_idx)

    pattern = args.hybrid_override_pattern.split("/")[0]
    return _convert_layer(pattern[layer_idx], f"backbone.layers.{layer_idx}", rest, param, args, name)


def _convert_mtp(depth: int, rest: str, param, args, name: str):
    """One MCore MTP parameter -> the checkpoint's single MTP depth.

    The inverse of hf_to_megatron/nemotron_h.py::_mtp_tensor (see its table).
    SGLang's draft (models/nemotron_h_mtp.py, loaded with is_mtp=True) takes
    exactly these ``mtp.layers.{k}.*`` names, and the RL weight sync hands it
    the same tensors as the target (speculative/eagle_worker_v2.py
    update_weights_from_tensor), so this is what keeps the draft on the
    trained MTP weights.

    The checkpoint -- and therefore the draft -- has ONE MTP depth, which the
    engine applies once per speculative step. Only a repeated MCore layer
    (--mtp-use-repeated-layer, depth 0 only) maps onto it one-to-one; distinct
    per-depth weights would all have to land on the same names.
    """
    if depth != 0:
        raise ValueError(
            f"{name!r}: MTP depth {depth} has no checkpoint counterpart. The Nemotron-H checkpoint and "
            "SGLang's draft hold one MTP depth; train it with --mtp-use-repeated-layer."
        )
    parts = args.hybrid_override_pattern.split("/")
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"{name!r}: --hybrid-override-pattern has no MTP segment ('<main>/<mtp>/...')")
    pattern = parts[1]

    if rest in {"enorm.weight", "hnorm.weight", "eh_proj.weight"}:
        return [(f"mtp.layers.0.{rest}", param)]
    if rest == "final_layernorm.weight":
        return [(f"mtp.layers.{len(pattern) - 1}.final_layernorm.weight", param)]

    inner = re.fullmatch(r"mtp_model_layer\.layers\.(\d+)\.(.+)", rest)
    if not inner:
        raise ValueError(f"Unsupported Nemotron-H MTP parameter {name!r}")
    k, inner_rest = int(inner.group(1)), inner.group(2)
    return _convert_layer(pattern[k], f"mtp.layers.{k}", inner_rest, param, args, name)


def _convert_layer(symbol: str, hf: str, rest: str, param, args, name: str):
    """One parameter of one M / E / * layer; ``hf`` is that layer's checkpoint prefix."""
    # The pre-mixer norm, whichever linear MCore folded it into.
    if rest in {
        "mixer.in_proj.layer_norm_weight",
        "self_attention.linear_qkv.layer_norm_weight",
        "pre_mlp_layernorm.weight",
        "input_layernorm.weight",
    }:
        return [(f"{hf}.norm.weight", param)]

    if symbol == "M":
        tp = _tp_size(args)
        d_inner, groups_state, nheads = _mamba_dims(args)
        if rest == "mixer.in_proj.weight":
            flat = _deinterleave_tp(param, [d_inner, d_inner, groups_state, groups_state, nheads], tp)
            return [(f"{hf}.mixer.in_proj.weight", flat)]
        # Flat on the MCore side (conv1d_weight), a real nn.Conv1d on the HF
        # side (conv1d.weight). Shapes match; only the name differs.
        if rest in {"mixer.conv1d_weight", "mixer.conv1d_bias"}:
            suffix = "weight" if rest.endswith("_weight") else "bias"
            flat = _deinterleave_tp(param, [d_inner, groups_state, groups_state], tp)
            return [(f"{hf}.mixer.conv1d.{suffix}", flat)]
        if rest in {"mixer.A_log", "mixer.D", "mixer.dt_bias", "mixer.norm.weight", "mixer.out_proj.weight"}:
            return [(f"{hf}.{rest}", param)]
        raise ValueError(f"Unsupported Nemotron-H Mamba parameter {name!r}")

    if symbol == "*":
        if rest == "self_attention.linear_qkv.weight":
            q, k, v = _split_qkv(param, args)
            return [
                (f"{hf}.mixer.q_proj.weight", q),
                (f"{hf}.mixer.k_proj.weight", k),
                (f"{hf}.mixer.v_proj.weight", v),
            ]
        if rest == "self_attention.linear_proj.weight":
            return [(f"{hf}.mixer.o_proj.weight", param)]
        raise ValueError(f"Unsupported Nemotron-H attention parameter {name!r}")

    if symbol == "E":
        if rest == "mlp.router.weight":
            return [(f"{hf}.mixer.gate.weight", param)]
        if rest == "mlp.router.expert_bias":
            return [(f"{hf}.mixer.gate.e_score_correction_bias", param)]
        if rest in {"mlp.fc1_latent_proj.weight", "mlp.fc2_latent_proj.weight"}:
            return [(f"{hf}.mixer.{rest.removeprefix('mlp.')}", param)]

        # Non-gated experts: linear_fc1 is up_proj alone, not cat(gate, up).
        expert_match = re.fullmatch(r"mlp\.experts\.linear_fc([12])(?:\.weight)?(\d+)?", rest)
        if expert_match:
            projection, expert_idx = expert_match.groups()
            hf_proj = "up_proj" if projection == "1" else "down_proj"
            if expert_idx is None:
                raise ValueError(f"{name!r} addresses the whole expert stack; expected a per-expert weight")
            return [(f"{hf}.mixer.experts.{int(expert_idx)}.{hf_proj}.weight", param)]

        shared = {
            "mlp.shared_experts.linear_fc1.weight": "up_proj",
            "mlp.shared_experts.linear_fc2.weight": "down_proj",
        }
        if rest in shared:
            return [(f"{hf}.mixer.shared_experts.{shared[rest]}.weight", param)]
        raise ValueError(f"Unsupported Nemotron-H MoE parameter {name!r}")

    raise ValueError(f"{hf} has unknown hybrid symbol {symbol!r} for parameter {name!r}")
