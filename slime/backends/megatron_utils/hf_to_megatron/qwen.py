from __future__ import annotations

import re

import torch

from .common import SafetensorReader, merge_gate_up, merge_qkv, strip_mcore_wrappers


def _direct_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor | None:
    mapping = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": (
            "model.embed_tokens.weight"
            if getattr(config, "tie_word_embeddings", False) or "lm_head.weight" not in reader
            else "lm_head.weight"
        ),
    }
    return reader.get_tensor(mapping[name]) if name in mapping else None


def _layer(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise KeyError(f"Unsupported Megatron parameter {name!r}")
    return int(match.group(1)), match.group(2)


def _attention_tensor(
    rest: str,
    prefix: str,
    reader: SafetensorReader,
    config,
) -> torch.Tensor | None:
    mapping = {
        "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
        "self_attention.linear_proj.bias": "self_attn.o_proj.bias",
        "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
        "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
        "self_attention.core_attention.softmax_offset": "self_attn.sinks",
    }
    if rest in mapping:
        return reader.get_tensor(f"{prefix}.{mapping[rest]}")
    match = re.fullmatch(r"self_attention\.linear_qkv\.(weight|bias)", rest)
    if match:
        suffix = match.group(1)
        return merge_qkv(
            *(reader.get_tensor(f"{prefix}.self_attn.{projection}_proj.{suffix}") for projection in "qkv"),
            config,
        )
    return None


def qwen_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    if (tensor := _direct_tensor(name, reader, config)) is not None:
        return tensor

    layer, rest = _layer(name)
    prefix = f"model.layers.{layer}"
    if (tensor := _attention_tensor(rest, prefix, reader, config)) is not None:
        return tensor

    mapping = {
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "mlp.linear_fc2.weight": "mlp.down_proj.weight",
    }
    if rest in mapping:
        return reader.get_tensor(f"{prefix}.{mapping[rest]}")
    if rest == "mlp.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.up_proj.weight"),
        )
    raise KeyError(f"Unsupported Qwen/Llama Megatron parameter {name!r}")


def _qwen_moe_layer_tensor(
    rest: str,
    prefix: str,
    reader: SafetensorReader,
) -> torch.Tensor | None:
    mapping = {
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "mlp.router.weight": "mlp.gate.weight",
        "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
        "mlp.shared_experts.linear_fc2.weight": "mlp.shared_expert.down_proj.weight",
        "mlp.shared_experts.gate_weight": "mlp.shared_expert_gate.weight",
    }
    if rest in mapping:
        return reader.get_tensor(f"{prefix}.{mapping[rest]}")
    if rest in {"mlp.shared_experts.linear_fc1.weight", "shared_experts.linear_fc1.weight"}:
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.shared_expert.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.shared_expert.up_proj.weight"),
        )
    if rest == "shared_experts.linear_fc2.weight":
        return reader.get_tensor(f"{prefix}.mlp.shared_expert.down_proj.weight")
    if rest == "shared_experts.gate_weight":
        return reader.get_tensor(f"{prefix}.mlp.shared_expert_gate.weight")

    match = re.fullmatch(r"mlp\.experts\.linear_fc([12])\.(weight|bias)(\d+)", rest)
    if match:
        projection, kind, expert = match.groups()
        if projection == "1":
            return merge_gate_up(
                reader.get_tensor(f"{prefix}.mlp.experts.{expert}.gate_proj.{kind}"),
                reader.get_tensor(f"{prefix}.mlp.experts.{expert}.up_proj.{kind}"),
            )
        return reader.get_tensor(f"{prefix}.mlp.experts.{expert}.down_proj.{kind}")
    return None


def qwen_moe_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    if (tensor := _direct_tensor(name, reader, config)) is not None:
        return tensor

    layer, rest = _layer(name)
    prefix = f"model.layers.{layer}"
    if (tensor := _attention_tensor(rest, prefix, reader, config)) is not None:
        return tensor
    if (tensor := _qwen_moe_layer_tensor(rest, prefix, reader)) is not None:
        return tensor
    return qwen_hf_tensor(name, reader, config)


def mimo_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    match = re.fullmatch(r"mtp\.layers\.(\d+)\.(.+)", name)
    if not match:
        return qwen_hf_tensor(name, reader, config)

    layer, rest = match.groups()
    mapping = {
        "enorm.weight": "token_layernorm.weight",
        "hnorm.weight": "hidden_layernorm.weight",
        "eh_proj.weight": "input_proj.weight",
        "final_layernorm.weight": "final_layernorm.weight",
    }
    if rest in mapping:
        tensor = reader.get_tensor(f"model.mtp_layers.{layer}.{mapping[rest]}")
        if rest == "eh_proj.weight":
            tensor = torch.cat(tensor.chunk(2, dim=1)[::-1], dim=1)
        return tensor

    proxy = f"decoder.layers.{layer}." + rest.removeprefix("transformer_layer.")
    hf_prefix = f"model.mtp_layers.{layer}"
    _, proxy_rest = _layer(proxy)
    if (tensor := _attention_tensor(proxy_rest, hf_prefix, reader, config)) is not None:
        return tensor
    mapping = {
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "mlp.linear_fc2.weight": "mlp.down_proj.weight",
    }
    if proxy_rest in mapping:
        return reader.get_tensor(f"{hf_prefix}.{mapping[proxy_rest]}")
    if proxy_rest == "mlp.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{hf_prefix}.mlp.gate_proj.weight"),
            reader.get_tensor(f"{hf_prefix}.mlp.up_proj.weight"),
        )
    raise KeyError(f"Unsupported MiMo Megatron parameter {name!r}")


def minimax_m2_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    if (tensor := _direct_tensor(name, reader, config)) is not None:
        return tensor
    layer, rest = _layer(name)
    prefix = f"model.layers.{layer}"
    if (tensor := _attention_tensor(rest, prefix, reader, config)) is not None:
        return tensor
    mapping = {
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "mlp.router.weight": "block_sparse_moe.gate.weight",
        "mlp.router.expert_bias": "block_sparse_moe.e_score_correction_bias",
        "self_attention.q_norm.weight": "self_attn.q_norm.weight",
        "self_attention.k_norm.weight": "self_attn.k_norm.weight",
    }
    if rest in mapping:
        return reader.get_tensor(f"{prefix}.{mapping[rest]}")
    match = re.fullmatch(r"mlp\.experts\.linear_fc([12])\.weight(\d+)", rest)
    if match:
        projection, expert = match.groups()
        if projection == "1":
            return merge_gate_up(
                reader.get_tensor(f"{prefix}.block_sparse_moe.experts.{expert}.w1.weight"),
                reader.get_tensor(f"{prefix}.block_sparse_moe.experts.{expert}.w3.weight"),
            )
        return reader.get_tensor(f"{prefix}.block_sparse_moe.experts.{expert}.w2.weight")
    raise KeyError(f"Unsupported MiniMax-M2 Megatron parameter {name!r}")
