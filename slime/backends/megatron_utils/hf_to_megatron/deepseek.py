from __future__ import annotations

import re

import torch

from .common import SafetensorReader, merge_gate_up, strip_mcore_wrappers


def _direct(name: str, reader: SafetensorReader, config) -> torch.Tensor | None:
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


def _dsa_reorder(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if name == "self_attention.wq_b.weight":
        tensor = tensor.view(-1, 128, tensor.shape[-1])
        return torch.cat((tensor[:, 64:], tensor[:, :64]), dim=1).flatten(0, 1)
    if name in {
        "self_attention.wk.weight",
        "self_attention.k_norm.weight",
        "self_attention.k_norm.bias",
    }:
        return torch.cat((tensor[64:], tensor[:64]), dim=0)
    return tensor


def _layer_tensor(
    layer: int,
    rest: str,
    reader: SafetensorReader,
    *,
    dsa: bool,
) -> torch.Tensor:
    prefix = f"model.layers.{layer}"
    mapping = {
        "input_layernorm.weight": "input_layernorm.weight",
        "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
        "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
        "self_attention.linear_q_proj.weight": "self_attn.q_proj.weight",
        "self_attention.linear_kv_down_proj.weight": "self_attn.kv_a_proj_with_mqa.weight",
        "self_attention.linear_kv_up_proj.layer_norm_weight": "self_attn.kv_a_layernorm.weight",
        "self_attention.linear_kv_up_proj.weight": "self_attn.kv_b_proj.weight",
        "self_attention.linear_q_down_proj.weight": "self_attn.q_a_proj.weight",
        "self_attention.linear_q_up_proj.weight": "self_attn.q_b_proj.weight",
        "self_attention.linear_q_up_proj.layer_norm_weight": "self_attn.q_a_layernorm.weight",
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "mlp.linear_fc2.weight": "mlp.down_proj.weight",
        "mlp.shared_experts.linear_fc2.weight": "mlp.shared_experts.down_proj.weight",
        "mlp.router.weight": "mlp.gate.weight",
        "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
    }
    if dsa:
        mapping |= {
            "self_attention.wq_b.weight": "self_attn.indexer.wq_b.weight",
            "self_attention.wk.weight": "self_attn.indexer.wk.weight",
            "self_attention.weights_proj.weight": "self_attn.indexer.weights_proj.weight",
            "self_attention.k_norm.weight": "self_attn.indexer.k_norm.weight",
            "self_attention.k_norm.bias": "self_attn.indexer.k_norm.bias",
        }
    if rest in mapping:
        return (
            _dsa_reorder(rest, reader.get_tensor(f"{prefix}.{mapping[rest]}"))
            if dsa
            else reader.get_tensor(f"{prefix}.{mapping[rest]}")
        )
    if rest == "mlp.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.up_proj.weight"),
        )
    if rest == "mlp.shared_experts.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.shared_experts.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.shared_experts.up_proj.weight"),
        )
    match = re.fullmatch(r"mlp\.experts\.linear_fc([12])\.weight(\d+)", rest)
    if match:
        projection, expert = match.groups()
        if projection == "1":
            return merge_gate_up(
                reader.get_tensor(f"{prefix}.mlp.experts.{expert}.gate_proj.weight"),
                reader.get_tensor(f"{prefix}.mlp.experts.{expert}.up_proj.weight"),
            )
        return reader.get_tensor(f"{prefix}.mlp.experts.{expert}.down_proj.weight")
    raise KeyError(f"Unsupported DeepSeek Megatron layer parameter {rest!r}")


def deepseek_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    if (tensor := _direct(name, reader, config)) is not None:
        return tensor

    mtp = re.fullmatch(r"mtp\.layers\.(\d+)\.(.+)", name)
    if mtp:
        mtp_layer, rest = mtp.groups()
        layer = config.num_hidden_layers + int(mtp_layer)
        mapping = {
            "eh_proj.weight": "eh_proj.weight",
            "enorm.weight": "enorm.weight",
            "hnorm.weight": "hnorm.weight",
            "final_layernorm.weight": "shared_head.norm.weight",
        }
        if rest in mapping:
            return reader.get_tensor(f"model.layers.{layer}.{mapping[rest]}")
        rest = rest.removeprefix("transformer_layer.")
    else:
        match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
        if not match:
            raise KeyError(f"Unsupported DeepSeek Megatron parameter {name!r}")
        layer, rest = int(match.group(1)), match.group(2)

    return _layer_tensor(
        int(layer),
        rest,
        reader,
        dsa=config.model_type in {"deepseek_v32", "glm_moe_dsa"},
    )
