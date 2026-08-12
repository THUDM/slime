from __future__ import annotations

import re

import torch

from .common import SafetensorReader, merge_gate_up, strip_mcore_wrappers
from .qwen import _attention_tensor, _direct_tensor, _qwen_moe_layer_tensor


def glm4_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    if (tensor := _direct_tensor(name, reader, config)) is not None:
        return tensor
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise KeyError(f"Unsupported GLM-4 Megatron parameter {name!r}")
    layer, rest = match.groups()
    prefix = f"model.layers.{layer}"
    if (tensor := _attention_tensor(rest, prefix, reader, config)) is not None:
        return tensor
    mapping = {
        "mlp.linear_fc1.weight": "mlp.gate_up_proj.weight",
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "mlp.linear_fc2.weight": "mlp.down_proj.weight",
        "post_self_attn_layernorm.weight": "post_self_attn_layernorm.weight",
        "post_mlp_layernorm.weight": "post_mlp_layernorm.weight",
    }
    if rest in mapping:
        return reader.get_tensor(f"{prefix}.{mapping[rest]}")
    raise KeyError(f"Unsupported GLM-4 Megatron parameter {name!r}")


def _glm4_moe_layer_tensor(rest: str, prefix: str, reader: SafetensorReader, config) -> torch.Tensor:
    if (tensor := _attention_tensor(rest, prefix, reader, config)) is not None:
        return tensor
    if rest == "mlp.shared_experts.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.shared_experts.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.shared_experts.up_proj.weight"),
        )
    if rest == "mlp.shared_experts.linear_fc2.weight":
        return reader.get_tensor(f"{prefix}.mlp.shared_experts.down_proj.weight")
    if (tensor := _qwen_moe_layer_tensor(rest, prefix, reader)) is not None:
        return tensor
    mapping = {
        "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "mlp.linear_fc2.weight": "mlp.down_proj.weight",
    }
    if rest in mapping:
        return reader.get_tensor(f"{prefix}.{mapping[rest]}")
    if rest == "mlp.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.up_proj.weight"),
        )
    raise KeyError(f"Unsupported GLM-4 MoE Megatron layer parameter {rest!r}")


def glm4_moe_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    if (tensor := _direct_tensor(name, reader, config)) is not None:
        return tensor
    mtp = re.fullmatch(r"mtp\.layers\.(\d+)\.(.+)", name)
    if mtp:
        mtp_layer, rest = mtp.groups()
        layer = config.num_hidden_layers + int(mtp_layer)
        mapping = {
            "enorm.weight": "enorm.weight",
            "hnorm.weight": "hnorm.weight",
            "eh_proj.weight": "eh_proj.weight",
            "final_layernorm.weight": "shared_head.norm.weight",
        }
        if rest in mapping:
            return reader.get_tensor(f"model.layers.{layer}.{mapping[rest]}")
        return _glm4_moe_layer_tensor(
            rest.removeprefix("transformer_layer."),
            f"model.layers.{layer}",
            reader,
            config,
        )
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise KeyError(f"Unsupported GLM-4 MoE Megatron parameter {name!r}")
    layer, rest = match.groups()
    return _glm4_moe_layer_tensor(rest, f"model.layers.{layer}", reader, config)
