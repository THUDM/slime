from __future__ import annotations

import re

import torch

from .common import SafetensorReader, strip_mcore_wrappers
from .qwen import _attention_tensor, _direct_tensor, _qwen_moe_layer_tensor

_DIRECT_ATTENTION = {
    "input_layernorm.weight",
    "linear_attn.A_log",
    "linear_attn.conv1d.weight",
    "linear_attn.dt_bias",
    "linear_attn.in_proj_ba.weight",
    "linear_attn.in_proj_qkvz.weight",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
}


def _qwen3_next_layer_tensor(
    rest: str,
    prefix: str,
    reader: SafetensorReader,
    config,
) -> torch.Tensor:
    direct = rest.removeprefix("self_attention.")
    if rest.startswith("self_attention.") and direct in _DIRECT_ATTENTION:
        return reader.get_tensor(f"{prefix}.{direct}")
    if rest in {"self_attention.linear_qkv.weight", "self_attention.linear_qkv.bias"}:
        suffix = rest.rsplit(".", 1)[-1]
        q, k, v = (reader.get_tensor(f"{prefix}.self_attn.{projection}_proj.{suffix}") for projection in "qkv")
        text = getattr(config, "text_config", config)
        groups = text.num_key_value_heads
        queries_per_group = text.num_attention_heads // groups
        head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
        trailing = q.shape[1:]
        q = q.reshape(groups, queries_per_group, 2, head_dim, *trailing).transpose(1, 2).flatten(1, 3)
        k = k.reshape(groups, head_dim, *trailing)
        v = v.reshape(groups, head_dim, *trailing)
        return torch.cat((q, k, v), dim=1).reshape(-1, *trailing).contiguous()
    if (tensor := _attention_tensor(rest, prefix, reader, config)) is not None:
        return tensor
    if (tensor := _qwen_moe_layer_tensor(rest, prefix, reader)) is not None:
        return tensor
    raise KeyError(f"Unsupported Qwen3-Next Megatron layer parameter {rest!r}")


def qwen3_next_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    if (tensor := _direct_tensor(name, reader, config)) is not None:
        return tensor

    mtp = re.fullmatch(r"mtp\.layers\.(\d+)\.(.+)", name)
    if mtp:
        layer, rest = mtp.groups()
        mapping = {
            "eh_proj.weight": "mtp.fc.weight",
            "enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "final_layernorm.weight": "mtp.norm.weight",
        }
        if rest in mapping:
            tensor = reader.get_tensor(mapping[rest])
            if rest == "eh_proj.weight":
                tensor = torch.cat(tensor.chunk(2, dim=1)[::-1], dim=1)
            return tensor
        prefix = f"mtp.layers.{layer}"
        rest = rest.removeprefix("transformer_layer.")
        return _qwen3_next_layer_tensor(rest, prefix, reader, config)

    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise KeyError(f"Unsupported Qwen3-Next Megatron parameter {name!r}")
    layer, rest = match.groups()
    return _qwen3_next_layer_tensor(rest, f"model.layers.{layer}", reader, config)
